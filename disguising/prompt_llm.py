import pandas as pd
import asyncio
import os
import nest_asyncio
from openai import OpenAI
import tiktoken  # For token counting
from dotenv import load_dotenv
import requests
from transformers import AutoTokenizer  # Using transformers to load the correct tokenizer for Meta LLaMA models
import aiohttp

# Allow nested event loops (needed for Jupyter Notebooks)
nest_asyncio.apply()

# Load environment variables
load_dotenv()  # Load environment variables from .env file

# Get the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")  
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set constants
TOKEN_LIMIT = 8192 # Max token limit
LOAD_PATH = "/home/ethanliu/dementor/disguising/minimodel_responses.csv"
SAVE_PATH = "/home/ethanliu/dementor/disguising/model-responses/gemma-3-4b_responses.csv"
#VLLM_URL = "http://0.0.0.0:8000"  # URL of your local VLLM server
VLLM_URL = "http://localhost:8000"  # URL of your local VLLM server

IS_VLLM = True  # Whether or not the model being called is a VLLM

class LLMHandler:
    def __init__(self, model_name="gpt-3.5-turbo", batch_size=10000, token_limit=TOKEN_LIMIT, 
                 save_path=SAVE_PATH, load_path=LOAD_PATH, prompt_column="prompt", output_column="model_response"):
        '''
        Initialize the LLMHandler.

        Parameters: 
        - model_name: The name of the model to use (e.g., "gpt-3.5-turbo").
        - batch_size: The number of prompts to process at once (default: 10000).
        - token_limit: The maximum token limit for the model (default: 8192).
        - load_path: The load path which contains the prompts.
        - save_path: The path to save the CSV file (default: a specified location).
        - prompt_column: The column name that contains the prompts (default: "prompt").
        - output_column: The column name where the new responses will be saved (default: "model_response").
        '''
        self.model_name = model_name
        self.batch_size = batch_size  # Number of prompts to batch for the LLM
        self.token_limit = token_limit
        self.load_path = load_path
        self.save_path = save_path
        self.client = OpenAI(api_key=openai_api_key)
        
        # Use Hugging Face tokenizer for LLaMA models
        if 'Meta-Llama' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Using transformers for LLaMA tokenizer
        else:
            self.tokenizer = tiktoken.get_encoding("gpt2")  # Default to GPT-2 encoding for GPT models

        self.df = pd.read_csv(self.load_path)
        self.df[output_column] = pd.NA
        self.prompt_column = prompt_column
        self.output_column = output_column

    async def fetch_response(self, prompt: str, is_vllm=False) -> str:
        """
        Fetches a response from either OpenAI's API or a local vLLM server based on the provided prompt.

        Parameters:
        - prompt: The prompt to send to the model.
        - is_vllm: Boolean flag indicating whether to use the local vLLM server (default: False for OpenAI API).

        Returns:
        - The response from the model.
        """
        # Token check
        num_tokens = len(self.tokenizer.encode(prompt))
        if num_tokens > self.token_limit:
            return "N/A"  # Skip this prompt if it's too long

        if not is_vllm:
            try:
                response = await asyncio.to_thread(self.client.chat.completions.create, 
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content 
            except Exception as e:
                return f"ERROR: {str(e)}"

        # Make HTTP request to local vLLM server
        else: 
            try:
                print(f"Sending prompt to vLLM server: {prompt[:50]}...")
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model_name, 
                        "prompt": prompt, 
                        "max_tokens": 2048,
                        #"top_k": -1
                    }
                    print(f"Payload sent: {payload}")
                    async with session.post(
                        f"{VLLM_URL}/v1/completions", 
                        json=payload
                    ) as response:
                        print(f"Received status: {response.status}")
                        if response.status == 200:
                            response_json = await response.json()
                            print(f"Response JSON: {response_json}")
                            choices = response_json.get("choices", [])
                            if choices:
                                return choices[0].get("text", "ERROR: No response field found.")
                            else:
                                print("No choices found in the response.")
                                return "ERROR: No choices found"
                        else:
                            print(f"Error: {response.status}")
                            return f"ERROR: {response.status}"
            except Exception as e:
                print(f"Error during HTTP request to vLLM server: {str(e)}")
                return f"ERROR: {str(e)}"
            
    async def process_batch(self, batch: list) -> list:
        responses = await asyncio.gather(*[self.fetch_response(prompt, IS_VLLM) for prompt in batch])
        return responses

    async def process_all_prompts(self):
        prompts = self.df[self.prompt_column].tolist()
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        print(f"Starting processing of {len(prompts)} prompts in {total_batches} batches")

        processed_count = 0
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            print(f"Processing batch {i // self.batch_size + 1}/{total_batches}...")

            try:
                responses = await self.process_batch(batch)
                for j, prompt in enumerate(batch):
                    self.df.loc[self.df[self.prompt_column] == prompt, self.output_column] = responses[j]
                    processed_count += 1

                self.df[[self.prompt_column, self.output_column]].to_csv(self.save_path, index=False, escapechar='\\')
                print(f"Processed {processed_count}/{len(prompts)} responses")

            except Exception as e:
                print(f"Error in batch {i // self.batch_size + 1}: {e}")
                continue 

        print(f"Processing complete. Total responses: {processed_count}/{len(prompts)}")
        assert processed_count == len(prompts), f"Expected {len(prompts)} responses, but got {processed_count}"

# Run event loop
if __name__ == "__main__":
    handler = LLMHandler(model_name="google/gemma-3-4b-it", batch_size=16, save_path=SAVE_PATH, 
                         prompt_column="prompt", output_column="model_response")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handler.process_all_prompts())
