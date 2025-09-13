import json
import logging
import os
import threading
from typing import List
import concurrent.futures

import lmdb
from openai import OpenAI
import anthropic
import datetime
import numpy as np

from .utils_general import (
    get_from_cache,
    save_to_cache,
    save_emb_to_cache,
    get_emb_from_cache,
)

from .global_vars import LLAMA_URL

logging.basicConfig(level=logging.ERROR)

if not os.path.exists("cache/llm_cache"):
    os.makedirs("cache/llm_cache")

if not os.path.exists("cache/llm_embed_cache"):
    os.makedirs("cache/llm_embed_cache")

llm_cache = lmdb.open("cache/llm_cache", map_size=int(1e11))
llm_embed_cache = lmdb.open("cache/llm_embed_cache", map_size=int(1e11))

cache_lock = threading.Lock()


def get_llm_output(
    prompt: str | List[str], model: str, cache=True, system_prompt=None, history=[], max_tokens=1024
) -> str | List[str]:
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    get_llm_output, p, model, cache, system_prompt, history, max_tokens
                )
                for p in prompt
            ]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]

    client = OpenAI(base_url=("https://api.openai.com/v1" if model != "llama-3-8b" else LLAMA_URL))
    systems_prompt = ("You are a helpful assistant." if not system_prompt else system_prompt)
    messages = (
        [{"role": "system", "content": systems_prompt}] + history + [{"role": "user", "content": prompt}]
    )
    key = json.dumps([model, messages])

    with cache_lock:
        cached_value = get_from_cache(key, llm_cache) if cache else None

    if cached_value is not None:
        logging.debug("LLM Cache Hit")
        return cached_value

    logging.debug("LLM Cache Miss")
    completion = client.chat.completions.create(
        model=("meta-llama/Meta-Llama-3-8B-Instruct" if model == "llama-3-8b" else model),
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"stop_token_ids": [128009]} if model == "llama-3-8b" else None,
    )
    response = completion.choices[0].message.content.strip()
    with cache_lock:
        save_to_cache(key, response, llm_cache)
    return response


def get_llm_embedding(prompt: str | List[str], model: str, cache=True):
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(get_llm_embedding, p, model, cache) for p in prompt]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]

    MAX_CHARS = 32768
    if len(str(prompt)) > MAX_CHARS:
        prompt = str(prompt)[:MAX_CHARS]

    client = OpenAI(base_url="https://api.openai.com/v1")
    key = json.dumps([model, prompt])

    with cache_lock:
        cached_value = get_emb_from_cache(key, llm_embed_cache) if cache else None

    if cached_value is not None:
        return cached_value

    text = str(prompt).replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    with cache_lock:
        save_emb_to_cache(key, embedding, llm_embed_cache)
    embedding = np.array(embedding)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

