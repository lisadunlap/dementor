#!/usr/bin/env python3
"""Test if the API is working."""
import os
from litellm import completion

# Test a simple API call
try:
    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=50,
        temperature=0
    )
    print("API working!")
    print("Response:", response["choices"][0]["message"]["content"])
except Exception as e:
    print("API error:", e)
