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
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p"),
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
        if not self.enable_cache:
            return completion(model=model, messages=messages, **kwargs)

        # Create cache key
        cache_key = self.create_cache_key(model, messages, **kwargs)

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
        response = completion(model=model, messages=messages, **kwargs)

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
    else:
        return completion(model=model, messages=messages, **kwargs)


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