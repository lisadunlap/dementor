Legacy Scripts (Deprecated)

This folder contains legacy or experimental scripts retained for reference. Prefer the streamlined scripts in `scripts/` and the canonical outputs under `disguising/`.

Current entrypoints:

- `disguise.py` for running disguise methods
- `disguising/scorer.py` (as a module) for scoring and metrics

Legacy scripts:
- `run_llm_scorer.py`, `llm_scorer.py`
- `generate_benchmark.py`, `run_math_evaluation.py`, `run_heuristic_analysis.py`
- `disguise_old.py`

Notes on paths:
- Inputs live under `data/datasets/` (prompts, GSM8K, style archetypes).
- Generated artifacts live under `disguising/` (`model-responses/`, `comparisons/`, `scores/`, `visualization/`).
- Legacy outputs previously written under `data/` are being migrated; avoid adding new results there.

These scripts are deprecated and may be removed in a future cleanup.
