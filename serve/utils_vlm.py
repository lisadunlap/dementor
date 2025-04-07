import base64
import json
import logging
import threading
import time

import openai
from openai import OpenAI

import litellm
import wandb

logging.basicConfig(level=logging.ERROR)

import os
from typing import Dict, List

import lmdb
import requests

from serve.utils_general import get_from_cache, save_to_cache
from serve.global_vars import IDEFICS_URL

if not os.path.exists("cache/vlm_cache"):
    os.makedirs("cache/vlm_cache")

vlm_cache = lmdb.open("cache/vlm_cache", map_size=int(1e11))

litellm.set_verbose = False
os.environ["LITELLM_LOG_LEVEL"] = "ERROR"
litellm.suppress_debug_info = True

vllm_api_base = "http://localhost:8000/v1"  # Default VLLM endpoint

def get_image_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_vlm_output(image: str, prompt: str, model: str, cache: bool = True) -> str:
    key = json.dumps([model, image, prompt])
    cached_value = get_from_cache(key, vlm_cache)
    if cached_value is not None and cache:
        logging.debug(f"VLM Cache Hit")
        return cached_value
    else:
        logging.debug("VLM Cache Miss")

    image_url = f"data:image/jpeg;base64,{get_image_base64(image)}"
    try:
        if model == "idefics":
            client = OpenAI(api_key="EMPTY", base_url= IDEFICS_URL)
            chat_response = client.chat.completions.create(
                model="HuggingFaceM4/Idefics3-8B-Llama3",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            )
            output = chat_response.choices[0].message.content
            save_to_cache(key, output, vlm_cache)
        elif model == "gpt-4o":
            response = litellm.completion(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}],
                }],
                max_tokens=512
            )
            output = response.choices[0].message.content
            # print(f"Response cost: {response._hidden_params['response_cost']}")
            save_to_cache(key, output, vlm_cache)
        else:
            chat_response = litellm.completion(
                model="HuggingFaceM4/Idefics3-8B-Llama3",  # Model name for routing
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                api_base=vllm_api_base,  # Your VLLM endpoint
                custom_llm_provider="openai"  # Using OpenAI-compatible API format
            )
            output = chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"VLM Error: {e}")
        return None
    return output


def captioning(image: str, model: str, cache: bool = True) -> str:
    caption = get_vlm_output(image, "Describe this image in detail.", model, cache=cache)
    return caption


def vqa(image: str, question: str, model: str, cache: bool = True) -> str:
    answer = get_vlm_output(image, question, model, cache=cache)
    return answer


def test_get_vlm_output():
    image = "../data/ballet_dancer_waterfall.png"

    model = "gpt-4o"
    caption = captioning(image, model, cache=False)
    print(f"{caption=}")
    question = "So whats up fam?"
    answer = vqa(image, question, model)
    print(f"{answer=}")

    model = "idefics"
    caption = captioning(image, model, cache=False)
    print(f"{caption=}")
    question = "So whats up fam?"
    answer = vqa(image, question, model, cache=False)
    print(f"{answer=}")


def test_get_vlm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_vlm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    test_get_vlm_output()
    # test_get_vlm_output_parallel()
