# End-to-end golden fixture — OLMo-3-7B baseline × AdvBench

These artifacts pin the **behavior** of the imitation erosion pipeline
(`experiments/imitation_safety/run_erosion_item.py`, stages A→D) on a tiny,
fully-deterministic-in-the-parts-that-matter slice. They exist so a refactor
(e.g. de-forking the `prompt_*` daemon twins, or splitting the runner monolith)
can be shown to preserve pipeline behavior end-to-end, not just at the unit level.

## Exact command that produced them

```bash
cd experiments/imitation_safety
CUDA_VISIBLE_DEVICES=5 \
DEMENTOR_IMITATION_ROOT=<scratch> \
HF_HOME=/data/ethantsliu/hf-cache HF_HUB_CACHE=/data/ethantsliu/hf-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
/data/ethantsliu/venvs/dementor-ethan/bin/python run_erosion_item.py baseline_olmo-3-7b \
  --benchmarks advbench --max-prompts 8 --gen-batch 8 --max-new-tokens 64 --no-heavy-graders
```

- **Item:** `baseline_olmo-3-7b` (base `allenai/OLMo-3-7B-Instruct`, no adapter — self-contained).
- **HF cache:** `/data/ethantsliu/hf-cache` holds BOTH OLMo-3-7B and the RTL judge
  `Qwen/Qwen3-8B` (the default `/data/ethantsliu/huggingface` cache has only the
  judge tokenizer, not its weights — generation there works, judging does not).
- Runtime ~0.8 min on one H100.

## Determinism finding (why the test asserts on verdicts, not raw text)

Running this slice **twice** on the same GPU:

| artifact | run-to-run |
|---|---|
| `prompt` (seed-42 subsample) | byte-identical |
| `model_response` (greedy gen) | **3 of 8 rows differ** — late-token flips from FP non-associativity in batched matmul |
| `rtl_raw` (judge raw text) | may differ (echoes the response) |
| `rtl_code` / `rtl_label` / `genuine_harm` (judge verdicts) | **byte-identical** |
| `metrics.json` (`metric`, `n`, `canonical_col`) | **byte-identical** |

Greedy decoding is **not** bit-reproducible on GPU here, so a golden test cannot
assert byte-identical generated text without false failures. The judge verdicts
and the computed metric ARE stable, and they are what the science depends on, so
the test (`tests/test_e2e_erosion_golden.py`) asserts on those and treats
`model_response` / `rtl_raw` as reference-only. A de-fork that preserves pipeline
behavior reproduces the verdicts + metric exactly; GPU text noise cannot break it.

`all_gens.csv` / `all_judged.csv` are kept in full as human-readable references
(their stable columns are what the test checks); `metrics.json` is the per-benchmark
metric and `item_metrics.json` the top-level item result.
