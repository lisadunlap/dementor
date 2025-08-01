import pandas as pd
import asyncio
import os
import nest_asyncio
import tiktoken  # For token counting
from dotenv import load_dotenv
from openai import OpenAI
import requests

"""
Run this script **after** your local vLLM server is up, e.g.

    vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype half --tensor-parallel-size 4 \
        --max-model-len 8192 --max-num-seqs 128 --gpu-memory-utilization 0.90

The script will:
  1. Read *model-responses/molmo-7b-o.csv* which must contain two columns:
        - prompt          – the original question
        - model_response  – molmo‑7b‑o’s answer
  2. Sample **five** rows every time it talks to the model and build a
     system prompt so Llama‑3‑8B mimics molmo’s style.
  3. Call the *local* vLLM server (OpenAI‑compatible endpoint
     http://localhost:8000/v1/chat/completions) and save the new answers in
     */home/ethanliu/dementor/disguising/comparisons/llama-3-8b_as_molmo-7b-o.csv*.
"""

load_dotenv()  # read .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")  # not used when is_vllm=True

TOKEN_LIMIT = 8192
MODEL_NAME = "qwen/qwen3-32B"  # must match what vLLM serves
VLLM_URL = "http://localhost:8000"
IS_VLLM = True  # we are calling our own llama‑3 server

LOAD_PATH = "disguising/model-responses/base_500_all_models/Qwen_Qwen2.5-VL-7B-Instruct.csv"
SAVE_PATH = "disguising/model-responses/disguised/random_sample_5_examples/qwen-qwen3-32b_as_qwen-qwen2.5-vl-7b-instruct.csv"

PROMPT_COLUMN = "prompt"
RESPONSE_COLUMN = "model_response"  # column holding "ground truth" original model answers in the csv
OUTPUT_COLUMN = "llama3_as_gpt-4o"

class LLMHandler:
    def __init__(self,
                 model_name: str = MODEL_NAME,
                 batch_size: int = 128,
                 example_size: int = 5,
                 token_limit: int = TOKEN_LIMIT,
                 load_path: str = LOAD_PATH,
                 save_path: str = SAVE_PATH,
                 prompt_column: str = PROMPT_COLUMN,
                 response_column: str = RESPONSE_COLUMN,
                 output_column: str = OUTPUT_COLUMN,
                 is_vllm: bool = IS_VLLM):

        self.model_name = model_name
        self.batch_size = batch_size
        self.example_size = example_size
        self.token_limit = token_limit
        self.load_path = load_path
        self.save_path = save_path
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.output_column = output_column
        self.is_vllm = is_vllm

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        # fall back to base encoding if model not in tiktoken list
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.df = pd.read_csv(self.load_path, nrows=1000)
        # ensure output column exists and empty
        self.df[self.output_column] = pd.NA
        
    def _build_system_prompt(self, examples: pd.DataFrame) -> str:
        """Return the system prompt that makes llama imitate molmo."""
        s = [
            "You are a helpful AI assistant. Answer **only** in the style "
            "shown in the examples below. Your goal is to mimic formatting, "
            "tone, level of detail, and phrasing – not to copy content.\n\n"
        ]
        for i, row in examples.iterrows():
            s.append(f"Example {i + 1}:\n"              # 1‑indexed for readability
                     f"prompt: {row[self.prompt_column]}\n"
                     f"response: {row[self.response_column]}\n\n")
        s.append("Here is the question to answer:")
        return "".join(s)

    async def _call_openai_or_vllm(self, full_prompt: str) -> str:
        if not self.is_vllm:
            # cloud OpenAI (not used in this workflow)
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return resp.choices[0].message.content
        else:
            # local vLLM OpenAI‑compatible endpoint
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": full_prompt}],
            }
            r = await asyncio.to_thread(requests.post, f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def _generate_one(self, prompt: str) -> str:
        examples = self.df[[self.prompt_column, self.response_column]].sample(
            n=self.example_size, random_state=None)  # true randomness per call
        full_prompt = f"{self._build_system_prompt(examples)}\n{prompt}"

        if len(self.encoder.encode(full_prompt)) > self.token_limit:
            return "N/A (prompt too long)"

        try:
            return await self._call_openai_or_vllm(full_prompt)
        except Exception as e:
            return f"ERROR: {e}"

    async def process_all(self):
        prompts = self.df[self.prompt_column].tolist()
        total = len(prompts)
        print(f"Processing {total} prompts in batches of {self.batch_size}…")

        for start in range(0, total, self.batch_size):
            batch_prompts = prompts[start:start + self.batch_size]
            print(f" ▸ Batch {start // self.batch_size + 1} / {(total - 1) // self.batch_size + 1}")
            try:
                responses = await asyncio.gather(*[self._generate_one(p) for p in batch_prompts])
            except Exception as e:
                print("Batch failed:", e)
                responses = [f"ERROR: {e}"] * len(batch_prompts)

            # write results back to dataframe
            for p, ans in zip(batch_prompts, responses):
                self.df.loc[self.df[self.prompt_column] == p, self.output_column] = ans

            self.df.to_csv(self.save_path, index=False)
            print(f"   Saved up to row {start + len(batch_prompts)} → {os.path.basename(self.save_path)}")

        print("✓ All done – file written to", self.save_path)

if __name__ == "__main__":
    nest_asyncio.apply()
    handler = LLMHandler()
    asyncio.run(handler.process_all())