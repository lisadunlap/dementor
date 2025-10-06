#!/usr/bin/env python3
"""
Few-Shot LLM Classifier

This script uses a few-shot prompting approach to classify which model generated a response.
It samples 5 examples from the dataset to create a few-shot prompt, then uses GPT-4o-mini
to classify the remaining responses.

Input: CSV with columns including 'prompt', 'target_response', 'source_response'
       (or 'prompt', 'model_a_response', 'model_b_response')

Output: Classification accuracy and detailed results

Usage:
    python scripts/few_shot.py \
        --input data/results/gsm8k/comparisons/disguised_vs_target/random_sampling/llama_gpt_4.1.csv \
        --model_a_col target_response \
        --model_b_col source_response \
        --model_a_name "gpt-4.1" \
        --model_b_name "llama-3-8b" \
        --output data/results/classifier_output.csv \
        --use_wandb \
        --wandb_project "few-shot-classifier"
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import json
import wandb
import plotly.graph_objects as go
from datetime import datetime

from cached_llm import cached_completion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str, prompt_col: str = "prompt", 
                 model_a_col: str = "target_response", 
                 model_b_col: str = "source_response") -> pd.DataFrame:
    """
    Load dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        prompt_col: Column name for prompts
        model_a_col: Column name for model A responses
        model_b_col: Column name for model B responses
        
    Returns:
        DataFrame with required columns
    """
    df = pd.read_csv(csv_path)
    
    required_cols = [prompt_col, model_a_col, model_b_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
    
    # filter out rows with missing values
    df = df[required_cols].dropna()
    
    logger.info(f"Loaded {len(df)} examples from {csv_path}")
    return df


def create_few_shot_prompt(examples: List[Dict], model_a_name: str, model_b_name: str) -> str:
    """
    Create few-shot system prompt with examples.
    
    Args:
        examples: List of dicts with keys 'prompt', 'response_a', 'response_b'
        model_a_name: Name of model A
        model_b_name: Name of model B
        
    Returns:
        System prompt string
    """
    prompt = f"""You are tasked with determining which model a given response is from: {model_a_name} or {model_b_name}.

To help you learn the differences between these models, here are 5 examples of each model answering the same prompt:

"""
    
    for i, ex in enumerate(examples, 1):
        prompt += f"""--- Example {i} ---
Prompt: {ex['prompt']}

{model_a_name} Response:
{ex['response_a']}

{model_b_name} Response:
{ex['response_b']}

"""
    
    prompt += f"""Now, given a new prompt and response, you must identify whether it came from {model_a_name} or {model_b_name}.

Instructions:
1. Carefully analyze the style, formatting, verbosity, and approach of the response
2. Compare it to the patterns you observed in the examples above
3. Output ONLY "{model_a_name}" or "{model_b_name}" - nothing else

Your classification:"""
    
    return prompt


def classify_response(system_prompt: str, prompt: str, response: str, 
                     model: str = "gpt-5-mini", temperature: float = 0.0) -> str:
    """
    Classify a single response using the LLM classifier.
    
    Args:
        system_prompt: Few-shot system prompt
        prompt: The input prompt
        response: The response to classify
        model: Classifier model to use
        temperature: Temperature for generation
        
    Returns:
        Classification result (model name)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Prompt: {prompt}\n\nResponse:\n{response}"}
    ]
    
    try:
        result = cached_completion(
            model=model,
            messages=messages,
            #temperature=temperature,
            #max_tokens=50
        )
        classification = result.choices[0].message.content.strip()
        return classification
    except Exception as e:
        logger.error(f"Error classifying response: {e}")
        return "ERROR"


def run_classification(df: pd.DataFrame, 
                      prompt_col: str,
                      model_a_col: str,
                      model_b_col: str,
                      model_a_name: str,
                      model_b_name: str,
                      num_few_shot: int = 5,
                      classifier_model: str = "gpt-5-mini",
                      temperature: float = 0.0,
                      seed: int = 42,
                      num_samples: int = None,
                      use_wandb: bool = False,
                      wandb_project: str = "few-shot-classifier",
                      wandb_run_name: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full classification pipeline.
    
    Args:
        df: Input dataframe
        prompt_col: Prompt column name
        model_a_col: Model A response column
        model_b_col: Model B response column
        model_a_name: Name of model A
        model_b_name: Name of model B
        num_few_shot: Number of few-shot examples
        classifier_model: Model to use for classification
        temperature: Temperature for generation
        seed: Random seed
        num_samples: Number of samples to use for testing (None for all)
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        
    Returns:
        Tuple of (results dataframe, metrics dict)
    """
    if use_wandb:
        if wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"few_shot_classifier_{timestamp}"
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "num_few_shot": num_few_shot,
                "classifier_model": classifier_model,
                "temperature": temperature,
                "seed": seed,
                "num_samples": num_samples,
                "total_samples": len(df)
            }
        )
    
    random.seed(seed)
    np.random.seed(seed)
    
    # limit dataset size if num_samples
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
        logger.info(f"Limited dataset to {num_samples} samples")
    
    # sampling the few-shot examples
    few_shot_indices = random.sample(range(len(df)), min(num_few_shot, len(df)))
    few_shot_df = df.iloc[few_shot_indices]
    test_df = df.drop(few_shot_indices).reset_index(drop=True)
    
    logger.info(f"Using {len(few_shot_df)} few-shot examples")
    logger.info(f"Testing on {len(test_df) * 2} responses ({len(test_df)} from each model)")
    
    # creating few-shot examples
    few_shot_examples = []
    for _, row in few_shot_df.iterrows():
        few_shot_examples.append({
            'prompt': row[prompt_col],
            'response_a': row[model_a_col],
            'response_b': row[model_b_col]
        })
    
    system_prompt = create_few_shot_prompt(few_shot_examples, model_a_name, model_b_name)
    
    logger.info(f"System prompt created ({len(system_prompt)} chars)")
    
    # test response classification
    results = []
    correct_a = 0
    correct_b = 0
    total_a = 0
    total_b = 0
    
    logger.info("Classifying responses...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        prompt = row[prompt_col]
        response_a = row[model_a_col]
        response_b = row[model_b_col]
        
        # classify model_A_response
        pred_a = classify_response(system_prompt, prompt, response_a, classifier_model, temperature)
        is_correct_a = model_a_name.lower() in pred_a.lower()
        
        results.append({
            'prompt': prompt,
            'response': response_a,
            'true_model': model_a_name,
            'predicted_model': pred_a,
            'correct': is_correct_a
        })
        
        total_a += 1
        if is_correct_a:
            correct_a += 1
        
        # classify model_B_response
        pred_b = classify_response(system_prompt, prompt, response_b, classifier_model, temperature)
        is_correct_b = model_b_name.lower() in pred_b.lower()
        
        results.append({
            'prompt': prompt,
            'response': response_b,
            'true_model': model_b_name,
            'predicted_model': pred_b,
            'correct': is_correct_b
        })
        
        total_b += 1
        if is_correct_b:
            correct_b += 1
    
    # metrics
    accuracy_a = correct_a / total_a if total_a > 0 else 0
    accuracy_b = correct_b / total_b if total_b > 0 else 0
    overall_accuracy = (correct_a + correct_b) / (total_a + total_b) if (total_a + total_b) > 0 else 0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        f'{model_a_name}_accuracy': accuracy_a,
        f'{model_b_name}_accuracy': accuracy_b,
        f'{model_a_name}_correct': correct_a,
        f'{model_a_name}_total': total_a,
        f'{model_b_name}_correct': correct_b,
        f'{model_b_name}_total': total_b,
        'num_few_shot_examples': num_few_shot,
        'classifier_model': classifier_model
    }
    
    results_df = pd.DataFrame(results)
    
    if use_wandb:
        
        wandb.summary["overall_accuracy"] = float(overall_accuracy)
        wandb.summary[f"{model_a_name}_accuracy"] = float(accuracy_a)
        wandb.summary[f"{model_b_name}_accuracy"] = float(accuracy_b)
        wandb.summary[f"{model_a_name}_correct"] = int(correct_a)
        wandb.summary[f"{model_a_name}_total"] = int(total_a)
        wandb.summary[f"{model_b_name}_correct"] = int(correct_b)
        wandb.summary[f"{model_b_name}_total"] = int(total_b)
        wandb.summary["num_few_shot_examples"] = int(num_few_shot)
        wandb.summary["total_test_samples"] = int(len(test_df))
        
        
        table_df = results_df.copy()
        
        # Convert all columns to strings for wandb compatibility
        for col in table_df.columns:
            table_df[col] = table_df[col].astype(str)
        
        table = wandb.Table(dataframe=table_df)
        wandb.log({"classification_results": table})
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Accuracy",
            x=[model_a_name, model_b_name, "Overall"],
            y=[accuracy_a, accuracy_b, overall_accuracy],
            text=[f"{accuracy_a:.2%}", f"{accuracy_b:.2%}", f"{overall_accuracy:.2%}"],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f'Classification Accuracy: {model_a_name} vs {model_b_name}',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        
        wandb.log({"accuracy_plot": fig})
        
        confusion_data = [
            [correct_a, total_a - correct_a],  # Model A: correct, incorrect
            [total_b - correct_b, correct_b]   # Model B: incorrect, correct
        ]
        
        fig_confusion = go.Figure(data=go.Heatmap(
            z=confusion_data,
            x=[f"Predicted {model_a_name}", f"Predicted {model_b_name}"],
            y=[f"True {model_a_name}", f"True {model_b_name}"],
            text=[[f"{correct_a}", f"{total_a - correct_a}"], 
                  [f"{total_b - correct_b}", f"{correct_b}"]],
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues'
        ))
        
        fig_confusion.update_layout(
            title='Classification Confusion Matrix',
            xaxis_title='Predicted Model',
            yaxis_title='True Model'
        )
        
        wandb.log({"confusion_matrix": fig_confusion})
        
        few_shot_table_df = few_shot_df.copy()
        for col in few_shot_table_df.columns:
            few_shot_table_df[col] = few_shot_table_df[col].astype(str)
        
        few_shot_table = wandb.Table(dataframe=few_shot_table_df)
        wandb.log({"few_shot_examples": few_shot_table})
        
        wandb.finish()
    
    return results_df, metrics


def main():
    parser = argparse.ArgumentParser(description="Few-shot LLM response classifier")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output CSV file (default: input_classified.csv)")
    parser.add_argument("--prompt_col", type=str, default="prompt",
                       help="Column name for prompts")
    parser.add_argument("--model_a_col", type=str, default="target_response",
                       help="Column name for model A responses")
    parser.add_argument("--model_b_col", type=str, default="source_response",
                       help="Column name for model B responses")
    parser.add_argument("--model_a_name", type=str, default="Model A",
                       help="Name of model A")
    parser.add_argument("--model_b_name", type=str, default="Model B",
                       help="Name of model B")
    parser.add_argument("--num_few_shot", type=int, default=5,
                       help="Number of few-shot examples")
    parser.add_argument("--classifier_model", type=str, default="gpt-5-mini",
                       help="Model to use for classification")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for classification")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use for testing (default: use all available)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                       help="Tensor parallel size")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="few-shot-classifier",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # FIXED: Create directory if it doesn't exist
    if args.output is None:
        input_path = Path(args.input)
        
        # Create output directory in the same parent as input
        output_dir = input_path.parent / "classifier_results"
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{input_path.stem}_classified.csv")
    else:
       
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {args.input}")
    df = load_dataset(
        args.input,
        prompt_col=args.prompt_col,
        model_a_col=args.model_a_col,
        model_b_col=args.model_b_col
    )
    
    # Run classification
    results_df, metrics = run_classification(
        df=df,
        prompt_col=args.prompt_col,
        model_a_col=args.model_a_col,
        model_b_col=args.model_b_col,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        num_few_shot=args.num_few_shot,
        classifier_model=args.classifier_model,
        temperature=args.temperature,
        seed=args.seed,
        num_samples=args.num_samples,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    
    # add weave WANDB
    
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    metrics_path = Path(args.output).parent / f"{Path(args.output).stem}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"{args.model_a_name} Accuracy: {metrics[f'{args.model_a_name}_accuracy']:.2%} "
          f"({metrics[f'{args.model_a_name}_correct']}/{metrics[f'{args.model_a_name}_total']})")
    print(f"{args.model_b_name} Accuracy: {metrics[f'{args.model_b_name}_accuracy']:.2%} "
          f"({metrics[f'{args.model_b_name}_correct']}/{metrics[f'{args.model_b_name}_total']})")
    print(f"\nFew-shot examples: {metrics['num_few_shot_examples']}")
    print(f"Classifier model: {metrics['classifier_model']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()