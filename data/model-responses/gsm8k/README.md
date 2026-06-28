GSM8K usage (dementor)

This folder contains GSM8K prompts and model responses used by the `dementor-disguise` CLI.

Files
- gpt-5_gsm8k_test_500.csv: GPT-5 responses for 500 GSM8K test prompts
- meta-llama_Meta-Llama-3-8B-Instruct.csv: Source model responses for the same prompts
- gpt-4o_gsm8k_responses_temp.csv: Optional legacy target

Run disguise on GSM8K
See `README.md` (Quick Start) and `AGENTS.md` (Apply Disguise) for canonical
`dementor-disguise` invocations. Pass the CSVs above as
`--source-responses` / `--target-responses` and supply a `--method` from the
registered choices in `dementor-disguise`. Results land under
`data/results/gsm8k/` by method/run.
