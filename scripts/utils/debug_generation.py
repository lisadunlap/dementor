#!/usr/bin/env python3
"""Debug the generation issue."""
import sys
import os
sys.path.append('/home/ethanliu/dementor')

from scripts.generate_responses import read_prompts, _gen_litellm

# Test reading prompts
print("Testing prompt reading...")
prompts = read_prompts('data/datasets/gsm8k/gsm8k_prompts_500.csv')
print(f"Read {len(prompts)} prompts")
print(f"First prompt: {prompts[0][:100]}...")

# Test API call
print("\nTesting API call...")
try:
    response = _gen_litellm(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompts[0]}],
        max_tokens=100,
        temperature=0.7
    )
    print(f"API response: {response[:100]}...")
except Exception as e:
    print(f"API error: {e}")

print("\nTest complete.")
