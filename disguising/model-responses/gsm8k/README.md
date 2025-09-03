GSM8K usage (dementor)

This folder contains GSM8K prompts and model responses used by disguise.py.

Files
- gpt-5_gsm8k_test_500.csv: GPT-5 responses for 500 GSM8K test prompts
- meta-llama_Meta-Llama-3-8B-Instruct.csv: Source model responses for the same prompts
- gpt-4o_gsm8k_responses_temp.csv: Optional legacy target

Run disguise on GSM8K
Use disguise.py directly on this folder. Path resolution is flexible for GSM8K-style filenames.

Example (prompt-only sanity check):
python disguising/disguise.py \
  --method hierarchical_math_disguise \
  --model meta-llama_Meta-Llama-3-8B-Instruct \
  --disguise_as gpt-5 \
  --data_dir disguising/model-responses/gsm8k \
  --test --skip_generation

Example (small real generation):
python disguising/disguise.py \
  --method hierarchical_math_disguise \
  --model meta-llama_Meta-Llama-3-8B-Instruct \
  --disguise_as gpt-5 \
  --data_dir disguising/model-responses/gsm8k \
  --test

Notes
- --skip_generation writes disguised prompts and runs analysis without inference.
- Plots are saved as HTML if Chrome/Kaleido is not available.
- Results are saved under disguising/model-responses/disguised/.

