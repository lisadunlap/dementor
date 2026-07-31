# call_center results — provenance manifest

Imported from Naz's run on `/home/nazcol/dementor-sync/` (paths preserved
inside each `*_summary.json`). Statuses below classify whether each artifact
is safe to cite/aggregate or whether it is an early/incomplete run.

Status legend:
- **usable** — full-size run, suitable for paper aggregation/plots.
- **exploratory** — runs whose parameters or scope were small (e.g., few
  samples); look at the summary JSON before using.
- **placeholder** — CSV exists but contains only the header row; the
  matching `_summary.json` records the *intended* run, not actual data. Do
  not aggregate or score.

## Pairwise comparison artifacts (`as_X` files)

| Pair | Method | Planned samples | Status | Notes |
| --- | --- | --- | --- | --- |
| `gpt-4.1_as_gpt-4.1` | random_sampling | 500 | usable | Self-baseline. CSV 1.7 MB, summary fully populated. |
| `gpt-4.1_as_llama-3-8b` | contrastive_with_al_examples | 500 | usable | Largest artifact (~11 MB); selection statistics and uncertainty fields populated. |
| `llama-3-8b_as_gpt-4.1` | random_sampling | 5 | placeholder | Result/score CSVs are header-only (102 B / 156 B). Re-run before using. |
| `qwen3-8b_as_gpt-4.1` | random_sampling | 2 | placeholder | Same pattern — header-only CSVs (102 B / 156 B). Model id was `vllm/Qwen/Qwen3-8B`. |

Each row above corresponds to four files in this directory:
`{pair}.csv`, `{pair}_scores.csv`, `{pair}_scores_metrics.csv`,
`{pair}_summary.json`.

## Split artifacts

| Path | Rows | Status | Notes |
| --- | --- | --- | --- |
| `eval200/openai_gpt-4.1-mini_disguised-meta-llama_Meta-Llama-3.1-8B-Instruct_clusters-500_eval_200.csv` | 773 lines | usable | Cluster-based disguise (500 clusters) over 200 eval prompts. |
| `train300/...train_300.csv` | 1115 lines | usable | Same setup over 300 train prompts. |

## Upstream inputs

Live in `data/model-responses/call_center/` (committed alongside the
results). The `call_center_prompts.csv` is the source prompt set; the
remaining files are raw responses per model:

- `call_center_prompts.csv` — 14.3 MB source prompts.
- `gpt-4.1.csv` — gpt-4.1 responses.
- `Qwen_Qwen3-8B.csv` — Qwen3-8B responses.
- `meta-llama_Meta-Llama-3-8B-Instruct-500.csv` — Llama-3-8B responses (n≈500).
- `meta-llama_Meta-Llama-3.1-8B-Instruct-500.csv` — Llama-3.1-8B responses (n≈500).

Treat all five as usable inputs; they are not derived artifacts and have
no `_summary.json` to validate against.

## What to re-run

Before paper-ready aggregation:
1. Regenerate `llama-3-8b_as_gpt-4.1` at 500 samples with a real disguise
   method (the existing summary used `random_sampling` with only 5).
2. Same for `qwen3-8b_as_gpt-4.1` (originally 2 samples).
3. Consider adding the missing legs of the 4×4 matrix
   (`gpt-4.1_as_qwen3-8b`, `llama-3-8b_as_llama-3-8b`, etc.) if the
   paper claims a full pairwise comparison.
