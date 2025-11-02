"""
Cached LLM wrapper for persistent caching across sessions.
Integrates LMDB-based caching from serve/utils_llm.py with litellm.
"""
import json
import logging
import os
import threading
from typing import List, Dict, Any, Optional
import concurrent.futures

import lmdb
from litellm import completion
import litellm

# Optional per-model routing via environment configuration
_MODEL_ALIAS_MAP: Dict[str, Dict[str, str] | str] = {}
_MODEL_CONFIG_MAP: Dict[str, Dict[str, str]] = {}

_alias_env = os.getenv("LITELLM_MODEL_ALIAS")
if _alias_env:
    try:
        _MODEL_ALIAS_MAP = json.loads(_alias_env)
        if hasattr(litellm, "set_model_aliases"):
            # Newer litellm exposes helper; use it when available
            litellm.set_model_aliases(_MODEL_ALIAS_MAP)  # type: ignore[attr-defined]
    except Exception as alias_err:  # pragma: no cover - logging only
        logging.warning(f"Could not apply LITELLM_MODEL_ALIAS: {alias_err}")
        _MODEL_ALIAS_MAP = {}

_config_env = os.getenv("LITELLM_CONFIG")
if _config_env:
    try:
        _MODEL_CONFIG_MAP = json.loads(_config_env)
    except Exception as config_err:  # pragma: no cover - logging only
        logging.warning(f"Could not parse LITELLM_CONFIG: {config_err}")
        _MODEL_CONFIG_MAP = {}


def _apply_model_routing(model: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Apply alias/config routing for the requested model."""
    effective_model = model
    updated_kwargs = dict(kwargs)  # shallow copy to avoid mutating caller

    alias_entry = _MODEL_ALIAS_MAP.get(model)
    if not alias_entry:
        alias_env = os.getenv("LITELLM_MODEL_ALIAS")
        if alias_env:
            try:
                alias_map = json.loads(alias_env)
                alias_entry = alias_map.get(model)
            except Exception:
                alias_entry = None
    provider_hint = None

    if alias_entry:
        if isinstance(alias_entry, str):
            effective_model = alias_entry
        elif isinstance(alias_entry, dict):
            effective_model = alias_entry.get("model", effective_model)
            for key in ("api_base", "api_key", "api_type", "api_version"):
                if alias_entry.get(key) and key not in updated_kwargs:
                    updated_kwargs[key] = alias_entry[key]
            provider_hint = alias_entry.get("custom_llm_provider") or alias_entry.get("litellm_provider")

    config_entry = _MODEL_CONFIG_MAP.get(model) or _MODEL_CONFIG_MAP.get(effective_model)
    if not config_entry:
        config_env = os.getenv("LITELLM_CONFIG")
        if config_env:
            try:
                config_map = json.loads(config_env)
                config_entry = config_map.get(model) or config_map.get(effective_model)
            except Exception:
                config_entry = None
    if config_entry:
        for key in ("api_base", "api_key", "api_type", "api_version"):
            if config_entry.get(key) and key not in updated_kwargs:
                updated_kwargs[key] = config_entry[key]
        provider_hint = provider_hint or config_entry.get("custom_llm_provider") or config_entry.get("litellm_provider")

    if provider_hint:
        updated_kwargs.setdefault("custom_llm_provider", provider_hint)
        updated_kwargs.setdefault("litellm_provider", provider_hint)
    if "api_base" in updated_kwargs and "custom_llm_provider" not in updated_kwargs:
        # default to openai-compatible when routing via custom base
        updated_kwargs["custom_llm_provider"] = "openai"
        updated_kwargs.setdefault("litellm_provider", "openai")

    return effective_model, updated_kwargs

# Import from serve utilities
from serve.utils_general import (
    get_from_cache,
    save_to_cache,
    hash_key
)

# Enable litellm cache as well (in-memory)
if not hasattr(litellm, 'cache') or litellm.cache is None:
    litellm.cache = litellm.Cache()

# Setup persistent cache directories
if not os.path.exists("cache"):
    os.makedirs("cache")
if not os.path.exists("cache/llm_cache"):
    os.makedirs("cache/llm_cache")

# Initialize LMDB cache for persistent storage
llm_cache = lmdb.open("cache/llm_cache", map_size=int(1e11))  # 100GB limit
cache_lock = threading.Lock()

logging.basicConfig(level=logging.ERROR)


class CachedLLM:
    """Cached wrapper around litellm completion with persistent LMDB storage."""

    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache

    def create_cache_key(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Create a deterministic cache key from model and messages."""
        # Include key parameters that affect output
        cache_data = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p"),
            "api_base": kwargs.get("api_base"),
            # Add other relevant parameters that affect output
        }
        # Remove None values
        cache_data = {k: v for k, v in cache_data.items() if v is not None}
        return json.dumps(cache_data, sort_keys=True)

    def completion(self, model: str, messages: List[Dict], **kwargs) -> Any:
        """
        Cached version of litellm.completion().

        Args:
            model: Model identifier
            messages: List of message dictionaries
            **kwargs: Additional parameters passed to litellm

        Returns:
            Completion response (same format as litellm.completion)
        """
        effective_model, routed_kwargs = _apply_model_routing(model, kwargs)
        kwargs = routed_kwargs

        if not self.enable_cache:
            return completion(model=effective_model, messages=messages, **kwargs)

        # Create cache key
        cache_key = self.create_cache_key(effective_model, messages, **kwargs)

        # Try to get from persistent cache first
        with cache_lock:
            cached_response = get_from_cache(cache_key, llm_cache)

        if cached_response is not None:
            logging.debug("LLM Cache Hit (persistent)")
            # Return the cached response (parse back from JSON)
            try:
                cached_data = json.loads(cached_response)
                # Create a proper mock response object that matches litellm structure
                class MockMessage:
                    def __init__(self, content, role='assistant'):
                        self.content = content
                        self.role = role

                class MockChoice:
                    def __init__(self, content, role='assistant'):
                        self.message = MockMessage(content, role)

                class MockResponse:
                    def __init__(self, choices_data):
                        if isinstance(choices_data, list) and 'message' in choices_data[0]:
                            # New format with full structure
                            self.choices = [MockChoice(
                                choice['message']['content'],
                                choice['message'].get('role', 'assistant')
                            ) for choice in choices_data]
                        else:
                            # Legacy format - just content string
                            content = choices_data if isinstance(choices_data, str) else str(choices_data)
                            self.choices = [MockChoice(content)]

                return MockResponse(cached_data.get('choices', cached_data))
            except json.JSONDecodeError:
                # Handle legacy cache entries that might just be strings
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                        self.role = 'assistant'

                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)

                class MockResponse:
                    def __init__(self, content):
                        self.choices = [MockChoice(content)]

                return MockResponse(cached_response)

        logging.debug("LLM Cache Miss")

        # Make the actual API call
        response = completion(model=effective_model, messages=messages, **kwargs)

        # Cache the response
        with cache_lock:
            # Store the full response object as JSON
            try:
                response_json = json.dumps({
                    "choices": [
                        {
                            "message": {
                                "content": response.choices[0].message.content,
                                "role": getattr(response.choices[0].message, 'role', 'assistant')
                            }
                        }
                    ]
                })
                save_to_cache(cache_key, response_json, llm_cache)
            except Exception as e:
                logging.warning(f"Failed to cache response: {e}")

        return response

    def batch_completion(self, model: str, batch_messages: List[List[Dict]], **kwargs) -> List[Any]:
        """
        Process multiple completions in parallel with caching.

        Args:
            model: Model identifier
            batch_messages: List of message lists
            **kwargs: Additional parameters

        Returns:
            List of completion responses
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(self.completion, model, messages, **kwargs)
                for messages in batch_messages
            ]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]


# Global cached LLM instance
cached_llm = CachedLLM()


def cached_completion(model: str, messages: List[Dict], enable_cache: bool = True, **kwargs) -> Any:
    """
    Drop-in replacement for litellm.completion() with persistent caching.

    Args:
        model: Model identifier
        messages: List of message dictionaries
        enable_cache: Whether to use caching (default True)
        **kwargs: Additional parameters passed to litellm

    Returns:
        Completion response (same format as litellm.completion)
    """
    if enable_cache:
        return cached_llm.completion(model=model, messages=messages, **kwargs)

    effective_model, routed_kwargs = _apply_model_routing(model, kwargs)
    return completion(model=effective_model, messages=messages, **routed_kwargs)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        with llm_cache.begin(write=False) as txn:
            stats = txn.stat()
            return {
                "cache_entries": stats['entries'],
                "cache_size_mb": stats['psize'] * stats['leaf_pages'] / (1024 * 1024),
                "cache_path": llm_cache.path()
            }
    except Exception as e:
        return {"error": str(e)}


def clear_cache():
    """Clear all cached entries."""
    with cache_lock:
        with llm_cache.begin(write=True) as txn:
            txn.drop(llm_cache.open_db())


if __name__ == "__main__":
    # Test the cached completion
    print("Testing cached completion...")

    test_messages = [{"role": "user", "content": "What is 2+2?"}]

    # First call (should miss cache)
    print("First call...")
    response1 = cached_completion("openai/gpt-3.5-turbo", test_messages)
    print(f"Response: {response1.choices[0].message.content[:50]}...")

    # Second call (should hit cache)
    print("Second call...")
    response2 = cached_completion("openai/gpt-3.5-turbo", test_messages)
    print(f"Response: {response2.choices[0].message.content[:50]}...")

    # Print cache stats
    print("Cache stats:", get_cache_stats())
