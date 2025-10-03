import base64
import json
import logging
import os
import lmdb
import requests
import litellm
from typing import Dict

from .utils_general import get_from_cache, save_to_cache
from .global_vars import IDEFICS_URL

logging.basicConfig(level=logging.ERROR)
if not os.path.exists("cache/vlm_cache"):
    os.makedirs("cache/vlm_cache")
vlm_cache = lmdb.open("cache/vlm_cache", map_size=int(1e11))

litellm.set_verbose = False
os.environ["LITELLM_LOG_LEVEL"] = "ERROR"
litellm.suppress_debug_info = True

vllm_api_base = "http://localhost:8000/v1"


def get_image_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_vlm_output(image: str, prompt: str, model: str, cache: bool = True) -> str:
    if "idefics" in model:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ]
        key = json.dumps([model, messages])
        cached_value = get_from_cache(key, vlm_cache) if cache else None
        if cached_value is not None:
            return cached_value
        payload = {"model": "idefics2-8b-chat", "messages": messages, "max_tokens": 512, "temperature": 0.0}
        resp = requests.post(f"{IDEFICS_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]
        save_to_cache(key, response_text, vlm_cache)
        return response_text
    else:
        raise NotImplementedError("Only idefics path retained for legacy")

