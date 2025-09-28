Examples and Provider Setup

This page shows concrete, copy‑paste examples for common providers via LiteLLM and how to run the core methods succinctly.

Environment variables
- OpenAI: `export OPENAI_API_KEY=...`
- Anthropic: `export ANTHROPIC_API_KEY=...`
- Google (Gemini): `export GEMINI_API_KEY=...`
- Together: `export TOGETHER_AI_API_KEY=...`
- xAI: `export XAI_API_KEY=...`

Model strings
- LiteLLM providers:
  - OpenAI: `openai/gpt-4o-mini`, `openai/gpt-4o`
  - Anthropic: `anthropic/claude-3-haiku-20240307`, `anthropic/claude-3-5-sonnet-20240620`
  - Google: `gcp/gemini-1.5-pro` (LiteLLM provider alias may vary)
  - Together: `together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
  - xAI: `xai/grok-2-latest`

- Local HuggingFace (Transformers): prefix with `hf:`
  - e.g., `hf:meta-llama/Llama-3.1-8B-Instruct`

- Local vLLM: prefix with `vllm:`
  - e.g., `vllm:/path/to/your/model` or `vllm:meta-llama/Llama-3.1-8B-Instruct`

Disguise commands
- Random baseline
  - `python disguise.py --model openai/gpt-4o-mini --disguise_as gpt-4o --method random_sampling --num_samples 200`

- Contrastive rules
  - `python disguise.py --model openai/gpt-4o-mini --disguise_as gpt-4o --method contrastive --num_samples 200`

- Composite (rules + AL‑selected examples)
  - `python disguise.py --model openai/gpt-4o-mini --disguise_as gpt-4o \
      --method contrastive_with_al_examples --num_samples 200 \
      --al-num-examples 5 --al-max-iterations 5 --al-batch-size 10 \
      --example-selector al`

Scoring with metrics
```bash
python -m scripts.scorer pairwise \
  --input data/results/chatbot_arena/disguised/my_run.csv \
  --output data/results/chatbot_arena/scores/my_run_scored/scored.csv
cat data/results/chatbot_arena/scores/my_run_scored/scored_metrics.csv
cat data/results/chatbot_arena/scores/my_run_scored/summary.json
```

Run summary (composite method)
- When you run `disguise.py` with the composite method, a run summary JSON is saved alongside the CSV:
  - `{method}_{model}_as_{target}_summary.json`
  - Includes method args, selection statistics (if AL selector), metrics file paths.

Notes
- If you cannot use an API provider, you can still evaluate existing results with the scorer and heuristics-only mode.
- For local judge scoring, install `vllm` and run the scorer with the default LLM judge; otherwise use `--heuristics-only`.
