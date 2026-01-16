#!/usr/bin/env python3
"""
Prepare MMLU-Pro dataset for Dementor pipeline.

This script:
1. Downloads MMLU-Pro from HuggingFace
2. Randomly samples 500 questions (300 train + 200 eval) using seed 42
3. Creates prompt CSVs in Dementor format with a single 'prompt' column
4. Formats questions as multiple-choice with options

Output files:
- data/datasets/mmlu_pro/mmlu_pro_prompts_train_300_seed42.csv
- data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv

Usage:
    python scripts/prepare_mmlu_pro.py
    python scripts/prepare_mmlu_pro.py --train_size 300 --eval_size 200 --seed 42
"""

import argparse
import os
import pandas as pd
from datasets import load_dataset
from typing import List, Dict

# Choice mappings for multiple choice questions
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def format_question_as_prompt(question_data: Dict) -> str:
    """
    Format a MMLU-Pro question into a structured prompt string.
    
    Args:
        question_data: Dictionary with 'question', 'options', and 'answer' keys
        
    Returns:
        Formatted prompt string with structure:
        Question:
        <question_text>
        
        Options:
        A. <option_1>
        B. <option_2>
        ...
        
        Select the best option from the ones given.
    """
    prompt = "Question:\n"
    prompt += question_data["question"] + "\n\n"
    prompt += "Options:\n"
    
    # Filter out N/A options
    options = [opt for opt in question_data["options"] if opt != "N/A"]
    
    for i, option in enumerate(options):
        prompt += f"{CHOICES[i]}. {option}\n"
    
    prompt += "\nSelect the best option from the ones given.\n"
    
    return prompt


def download_and_prepare_mmlu_pro(
    train_size: int = 300,
    eval_size: int = 200,
    seed: int = 42,
    output_dir: str = "data/datasets/mmlu_pro"
) -> None:
    """
    Download MMLU-Pro and create train/eval splits.
    
    Args:
        train_size: Number of questions for training split
        eval_size: Number of questions for evaluation split
        seed: Random seed for reproducibility
        output_dir: Directory to save output CSVs
    """
    print("="*60)
    print("MMLU-Pro Dataset Preparation for Dementor")
    print("="*60)
    
    # Download dataset from HuggingFace
    print("\n📚 Downloading MMLU-Pro dataset from HuggingFace...")
    print("   (This may take a few minutes on first run)")
    
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_data = list(dataset["test"])
    
    print(f"✅ Downloaded {len(test_data)} test questions")
    
    # Convert to DataFrame for easier manipulation
    print("\n🔧 Processing questions...")
    questions = []
    for item in test_data:
        questions.append({
            "question_id": item.get("question_id", ""),
            "category": item.get("category", ""),
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
            "prompt": format_question_as_prompt(item)
        })
    
    df = pd.DataFrame(questions)
    print(f"✅ Processed {len(df)} questions")
    
    # Category distribution
    print(f"\n📊 Category distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count}")
    
    # Random sample with seed for reproducibility
    total_size = train_size + eval_size
    print(f"\n🎲 Randomly sampling {total_size} questions (seed={seed})...")
    print(f"   Train: {train_size} questions")
    print(f"   Eval:  {eval_size} questions")
    
    if len(df) < total_size:
        raise ValueError(
            f"Dataset has only {len(df)} questions, "
            f"but {total_size} requested (train={train_size}, eval={eval_size})"
        )
    
    # Sample total_size questions with seed
    sampled_df = df.sample(n=total_size, random_state=seed).reset_index(drop=True)
    
    # Split into train and eval
    train_df = sampled_df.iloc[:train_size].reset_index(drop=True)
    eval_df = sampled_df.iloc[train_size:].reset_index(drop=True)
    
    print(f"✅ Split complete:")
    print(f"   Train: {len(train_df)} questions")
    print(f"   Eval:  {len(eval_df)} questions")
    
    # Show category distribution in splits
    print(f"\n📊 Train split category distribution:")
    for category, count in train_df['category'].value_counts().items():
        print(f"   {category}: {count}")
    
    print(f"\n📊 Eval split category distribution:")
    for category, count in eval_df['category'].value_counts().items():
        print(f"   {category}: {count}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train prompts (only 'prompt' column for Dementor)
    train_output = os.path.join(output_dir, f"mmlu_pro_prompts_train_{train_size}_seed{seed}.csv")
    train_prompts_df = pd.DataFrame({'prompt': train_df['prompt']})
    train_prompts_df.to_csv(train_output, index=False)
    print(f"\n💾 Saved train prompts to: {train_output}")
    
    # Save eval prompts (only 'prompt' column for Dementor)
    eval_output = os.path.join(output_dir, f"mmlu_pro_prompts_eval_{eval_size}_seed{seed}.csv")
    eval_prompts_df = pd.DataFrame({'prompt': eval_df['prompt']})
    eval_prompts_df.to_csv(eval_output, index=False)
    print(f"💾 Saved eval prompts to: {eval_output}")
    
    # Also save full metadata (with answers) for reference
    full_train_output = os.path.join(output_dir, f"mmlu_pro_full_train_{train_size}_seed{seed}.csv")
    train_df.to_csv(full_train_output, index=False)
    print(f"\n📋 Saved full train metadata (with answers) to: {full_train_output}")
    
    full_eval_output = os.path.join(output_dir, f"mmlu_pro_full_eval_{eval_size}_seed{seed}.csv")
    eval_df.to_csv(full_eval_output, index=False)
    print(f"📋 Saved full eval metadata (with answers) to: {full_eval_output}")
    
    # Show example prompts
    print(f"\n📝 Example prompts from train set:")
    print("="*60)
    for i in range(min(2, len(train_df))):
        print(f"\nExample {i+1}:")
        print(f"Category: {train_df.iloc[i]['category']}")
        print(f"Answer: {train_df.iloc[i]['answer']}")
        print(f"\nPrompt:")
        print(train_df.iloc[i]['prompt'][:300] + "..." if len(train_df.iloc[i]['prompt']) > 300 else train_df.iloc[i]['prompt'])
        print("-"*60)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Generate responses with generate_responses.py")
    print(f"      python scripts/generate_responses.py \\")
    print(f"        --prompts-file {train_output} \\")
    print(f"        --output-csv data/model-responses/mmlu_pro/full/MODEL_NAME.csv \\")
    print(f"        basic --model PROVIDER/MODEL_NAME")
    print(f"")
    print(f"   2. Apply disguise with disguise.py")
    print(f"   3. Score with scorer.py")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MMLU-Pro dataset for Dementor pipeline"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=300,
        help="Number of questions for training split (default: 300)"
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=200,
        help="Number of questions for evaluation split (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/mmlu_pro",
        help="Output directory for CSVs (default: data/datasets/mmlu_pro)"
    )
    
    args = parser.parse_args()
    
    download_and_prepare_mmlu_pro(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

