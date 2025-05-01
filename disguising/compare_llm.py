import pandas as pd
import asyncio
import os
import nest_asyncio
from openai import OpenAI
import tiktoken  # For token counting
from dotenv import load_dotenv
from vllm import VLLM  # Placeholder for vLLM import
import requests

# Allow nested event loops (needed for Jupyter Notebooks)
nest_asyncio.apply()

# Load environment variables
load_dotenv()  # Load environment variables from .env file

# Get the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")  
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set constants
TOKEN_LIMIT = 8192  # Max token limit for GPT-3.5-turbo. 
LOAD_PATH = "/home/ethanliu/dementor/disguising/minimodel_responses.csv"
SAVE_PATH = "/home/ethanliu/dementor/disguising/test_responses.csv"
VLLM_URL = "http://localhost:5000" # URL of your local VLLM server
IS_VLLM = False # Whether or not the model being called is a VLLM

class LLMHandler:
    def __init__(self, model_name="gpt-3.5-turbo", batch_size=10000, example_size = 5, 
                 token_limit=TOKEN_LIMIT, save_path=SAVE_PATH, load_path = LOAD_PATH,
                 prompt_column="prompt", response_column="gpt4omini_response", output_column="gpt35_reprompted"):
        '''
        Initialize the LLMHandler.
        
        Parameters: 
        - model_name: The name of the model to use (e.g., "gpt-3.5-turbo").
        - is_vllm: Whether or not the model is an LLM
        - batch_size: The number of prompts to process at once (default: 10000).
        - example_size: The number of examples to sample from the dataframe for the system prompt (default: 5).
        - token_limit: The maximum token limit for the model (default: 8192).
        - load_path: The load path which contains the prompts as well as the LLM outputs that you would like the model to "act" as. 
        - save_path: The path to save the CSV file (default: a specified location).
        - prompt_column: The column name that contains the prompts (default: "prompt").
        - response_column: The column name that contains the model responses (default: "gpt4omini_response").
        - output_column: The column name where the new responses will be saved (default: "gpt35_reprompted").
        '''
        self.model_name = model_name
        self.is_vllm = IS_VLLM
        self.batch_size = batch_size # Number of prompts to batch for the LLM
        self.example_size = example_size # Number of examples for the system prompt
        self.token_limit = token_limit
        self.load_path = load_path
        self.save_path = save_path
        self.client = OpenAI(api_key=openai_api_key)
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.df = pd.read_csv(self.load_path)
        # reset the reprompted column to NA
        self.df[self.output_column] = pd.NA

    def get_system_prompt(self, examples: pd.DataFrame) -> str:
        '''
        Generates the system prompt string based on sampled examples from the dataframe.

        Parameters:
        - examples: A DataFrame containing the examples to use in the system prompt.

        Returns:
        - A formatted system prompt string.
        '''
        
        system_prompt = f'''You are a helpful AI assistant. You answer the questions provided in the style defined by the example question and responses below. Note that your task is to match the style of the responses only.\n\n'''
        
        for i, row in examples.iterrows():
            # Replace gpt4omini_response
            system_prompt += f"Example {i+1}:\nprompt: {row['self.prompt_column']}\nresponse: {row['self.response_column']}\n\n"

        system_prompt += "Here is the question to answer: "
        return system_prompt

    async def fetch_response(self, prompt: str, is_vllm = False) -> str:
        """
        Fetches a response from either OpenAI's API or a local vLLM server based on the provided prompt.

        Parameters:
        - prompt: The prompt to send to the model.
        - is_vllm: Boolean flag indicating whether to use the local vLLM server (default: False for OpenAI API).

        Returns:
        - The response from the model.
        """
        
        # Make sure that the LLM_response for your column is not "I cannot assist with your request"
        sampled_examples = self.df[[self.prompt_column, self.response_column]].sample(n=self.example_size, random_state=42) 
        full_prompt = self.get_system_prompt(sampled_examples) + "\n" + prompt

        # Token check
        num_tokens = len(self.encoder.encode(full_prompt))
        if num_tokens > self.token_limit:
            return "N/A"  # Skip this prompt if it's too long

        if not is_vllm:
            try:
                response = await asyncio.to_thread(self.client.chat.completions.create, 
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.choices[0].message.content 
        
            except Exception as e:
                return f"ERROR: {str(e)}"
        
        # Make HTTP request to local vLLM server
        else: 
            try:
                response = requests.post(
                    f"{VLLM_URL}/predict",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt
                    }
                )
                response.raise_for_status()  # Raise an exception for HTTP errors
                return response.json().get("response", "ERROR: No response field found.")
            except requests.exceptions.RequestException as e:
                return f"ERROR: {str(e)}"

    async def process_batch(self, batch: list) -> list:
        responses = await asyncio.gather(*[self.fetch_response(prompt, self.is_vllm) for prompt in batch])
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

                self.df.to_csv(self.save_path, index=False, escapechar='\\')
                print(f"Processed {processed_count}/{len(prompts)} responses")

            except Exception as e:
                print(f"Error in batch {i // self.batch_size + 1}: {e}")
                continue

        print(f"Processing complete. Total responses: {processed_count}/{len(prompts)}")
        assert processed_count == len(prompts), f"Expected {len(prompts)} responses, but got {processed_count}"

# Run event loop
if __name__ == "__main__":
    handler = LLMHandler(model_name="gpt-3.5-turbo", batch_size=10000, example_size=5, 
                         save_path=SAVE_PATH, 
                         prompt_column="prompt", response_column="gpt4omini_response", output_column="gpt35_reprompted")
    
    # model_name is the LLM you have. 
    # The response_column should be what you want the model_name LLM to act as, namely some other LLM. 
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handler.process_all_prompts())
