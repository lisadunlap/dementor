# Matrix Run Log — SFT + DPO Fine-Tuning (2026-05-26 → 05-28)

Execution record for rungs 3–4 of the intervention-ladder experiment
(`docs/experiment_implementation_plan.md`). This documents what was actually
run, where the artifacts live, and what remains.

## Summary

| Item | Result |
| --- | --- |
| Models (sources == targets) | 4: Llama-3.1-8B-Instruct, Qwen3.6-27B, Nemotron-3-Nano-30B-A3B-BF16, gpt-oss-20b |
| Datasets (train) | GSM8K, Chatbot Arena, WritingPrompts (500 train prompts each) |
| Datasets (eval, held out) | + HumanEval (164, eval-only) |
| **SFT adapters** | **108** (4 src × 3 cross-tgt × 3 train-ds × 3 seeds) |
| **DPO adapters** | **108** (same shape, trained on top of SFT) |
| **Self-baseline SFT adapters** | **12** (4 models × 3 datasets × seed1, source==target — drift control) |
| Total adapters | 228, all trained on Tinker (LoRA r=32, all-linear) |
| Errors | 0 SFT, 0 DPO |
| Storage | All adapters mirrored to HuggingFace org `dementor-research/{sft,dpo,self_sft}_*` (public; grouped into per-dataset collections) |

## Pipeline (workflows/run_matrix.py)

All phases driven by a single dispatcher with subcommands:

| Subcommand | Phase | What it does |
| --- | --- | --- |
| `make-splits` (scripts/make_matrix_splits.py) | A.1 | 7 prompt CSVs; train/eval disjointness audited |
| `generate-target-responses` | A.2 | Target model responses on train prompts (SFT completions) |
| `build-sft-data` | B | Join (train_prompt, target_response) → 36 SFT CSVs |
| `launch-sft` | C | 108 Tinker LoRA SFT jobs, parallel-4, retry + Llama-first |
| `build-dpo-data` | D.1 | (prompt, chosen=target, rejected=source) → 36 DPO CSVs |
| `launch-dpo` | D.2 | 108 Tinker DPO jobs on top of SFT adapters |
| `backfill-register-dpo` | — | Recover DPO URIs from cookbook checkpoints.jsonl |
| `push-to-hf` | — | Stream Tinker → PEFT → HuggingFace, parallel-4 |

## Key parameters

- **SFT:** LoRA rank 32, target_modules=all-linear, 3 epochs, batch 16, lr 1e-4.
- **DPO:** β=0.1, 1 epoch, lr 1e-5, initialized from the matching SFT adapter.
- **Renderers (thinking disabled for comparability):** Llama=`llama3`,
  Qwen=`qwen3_5` (+`enable_thinking=False`), Nemotron=`nemotron3`
  (+`enable_thinking=False`), gpt-oss=`gpt_oss` (+`reasoning_effort=low`,
  multi-channel `final` parsed out).

## Adapter naming convention

```
{stage}_{dataset}_{source_slug}_as_{target_slug}_seed{N}
e.g. dpo_gsm8k_llama-3.1-8b_as_qwen3.6-27b_seed1
self_sft_gsm8k_llama-3.1-8b_as_llama-3.1-8b_seed1
```

Each appears as a public HF repo `dementor-research/{alias}` and as an entry in
`data/tinker_adapters.json` (with both the sampler URI for inference and the
state URI for download).

## Loading any adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "dementor-research/dpo_gsm8k_llama-3.1-8b_as_qwen3.6-27b_seed1")
```

## Notable issues solved during the run

1. **Qwen/Nemotron/gpt-oss thinking traces** — disabled per-model via
   chat-template kwargs so outputs are comparable for fingerprint analysis.
2. **Matplotlib GUI backend** crashed in ThreadPoolExecutor workers — fixed
   with `matplotlib.use("Agg")` at import.
3. **Tinker checkpoint expiry** — older sampler checkpoints get GC'd; fixed by
   exporting immediately and saving both sampler + state URIs.
4. **Tinker PEFT-export timeout** — the SDK's high-level `weights.download`
   times out at ~80s before Tinker responds. Worked around by calling the
   archive endpoint directly with httpx + manual 302/503 polling + long timeout.
5. **Tinker DPO checkpoint URIs** — cookbook writes them to
   `checkpoints.jsonl`, not `metrics.jsonl`; added a parser + backfill command.
6. **Network resilience** — survived 2 wifi outages + 1 network switch via
   per-cell retry with backoff + skip-if-already-done resumability.

## What's done vs remaining

**Done:** rungs 3 (SFT) + 4 (DPO), all adapters trained, verified by sampling,
and stored portably on HuggingFace.

**Remaining (the actual measurement):**
- A.3 — eval baselines (4 models × 4 eval datasets × 3 seeds)
- Rungs 1–2 — prompting/example-selection grid (7 methods × 336 cells)
- Phase E — eval inference on all 228 adapters over held-out splits
- Phase F — behavioral-inertia scoring (source_persistence, probe, per-axis)
- Rung 5 — activation steering (optional)
- Headline figure — persistence vs intervention-ladder rung
