import pandas as pd
from sklearn.metrics import cohen_kappa_score
import wandb
import numpy as np
from vllm import LLM, SamplingParams
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Simplified prompt to reduce token count
JUDGE_PROMPT = """Compare these two AI responses and choose the better one:

Question: {question}
A: {response_a}
B: {response_b}

Which response (A or B) is better? Explain briefly and end with exactly one line stating just A, B, or tie.
"""

def truncate_text(text, max_length=400):
    """Truncate text to max_length characters"""
    if not isinstance(text, str):
        text = str(text)
    return text[:max_length] + "..." if len(text) > max_length else text

def setup_llm():
    """Initialize the vLLM model with appropriate parameters"""
    try:
        model = LLM(
            model="meta-llama/Llama-2-7b-chat-hf",  # Using 7B model instead of 8B
            dtype="float16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            trust_remote_code=True
        )
        return model
    except Exception as e:
        logging.error(f"Error setting up LLM: {e}")
        raise

def get_model_preference(llm, question, response_a, response_b):
    """Get model preference with proper error handling and token management"""
    try:
        # Truncate inputs
        question = truncate_text(question, 200)
        response_a = truncate_text(response_a, 400)
        response_b = truncate_text(response_b, 400)
        
        # Format prompt
        prompt = JUDGE_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=128
        )
        
        # Get response
        outputs = llm.generate([prompt], sampling_params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return "Error: No output generated"
        
    except Exception as e:
        logging.error(f"Error in get_model_preference: {e}")
        return "Error: " + str(e)

def extract_preference(output):
    """Extract A/B/tie preference from model output"""
    output = output.lower().strip()
    if "a" in output.split()[-1]:
        return "A"
    elif "b" in output.split()[-1]:
        return "B"
    else:
        return "tie"

def main():
    # Parse arguments (keep your existing argument parsing code)
    
    try:
        # Initialize wandb
        run = wandb.init(
            project="model-comparison",
            name="llama-judge-comparison",
            config={
                "model": "llama-2-7b-chat"
            }
        )

        # Load data
        df = pd.read_csv(args.data_path)
        if args.test:
            df = df.head(3)
            logging.info("Test mode: using first 3 rows")

        # Setup LLM
        llm = setup_llm()
        
        # Process each row
        results = []
        for idx, row in df.iterrows():
            logging.info(f"Processing row {idx+1}/{len(df)}")
            
            # Get normal order preference
            normal_pref = get_model_preference(
                llm,
                row['question'],
                row[args.models[0]],
                row[args.models[1]]
            )
            
            # Get reversed order preference
            reversed_pref = get_model_preference(
                llm,
                row['question'],
                row[args.models[1]],
                row[args.models[0]]
            )
            
            # Extract preferences
            normal_result = extract_preference(normal_pref)
            reversed_result = extract_preference(reversed_pref)
            
            results.append({
                'normal_pref': normal_result,
                'reversed_pref': reversed_result
            })
            
            # Log progress
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1} rows")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        df = pd.concat([df, results_df], axis=1)
        
        # Save results
        df.to_csv(args.output_path, index=False)
        logging.info(f"Results saved to {args.output_path}")
        
        # Log to wandb (keep your existing wandb logging code)
        
        wandb.finish()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        if wandb.run is not None:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()