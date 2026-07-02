# Matrix Run Log — SFT + DPO Fine-Tuning (2026-05-26 → 05-28)

Execution record for rungs 3–4 of the intervention-ladder experiment. This
documents what was actually run, where the artifacts live, and what remains.

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

## Pipeline (dementor/training/matrix.py)

All phases driven by a single dispatcher with subcommands:

| Subcommand | Phase | What it does |
| --- | --- | --- |
| `make-splits` (dementor/data_utils/make_matrix_splits.py) | A.1 | 7 prompt CSVs; train/eval disjointness audited |
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

---

## Local-backend + FSDP validation (2026-06-30)

Supplementary to the Tinker matrix above. This validates the **non-Tinker local training
path** (`dementor/training/local_backend.py`) and adds **multi-GPU FSDP** — the capability the
`backend: local` roster models (`google/gemma-4-*`) and any large local source need.
**These are infrastructure + methods checks on DENSE Qwen stand-ins — NOT roster cells.**

| Item | Result |
| --- | --- |
| Pipeline | local SFT → DPO → held-out gen → persistence, scored end-to-end with no Tinker (Qwen2.5-7B/14B/32B → Qwen3-8B, chatbot_arena, 60 held-out eval) |
| FSDP + PEFT fix | the FSDP adapter save wrote a sharded/empty slice; fixed in `_save_peft_adapter` (commit `5da6016`) via an all-rank `FULL_STATE_DICT` gather. Validated on dense Qwen: single-GPU and 2-GPU FSDP produce identical 392-tensor adapters; 32B genuinely shards (~46 GiB/GPU vs the 57.7 GiB full model). |
| Methods note (stand-ins, **not** roster) | DPO erases ≥ SFT fingerprint; **significant only at 14B/32B, indistinguishable at 7B** (0.39 vs 0.38, overlapping CIs); larger source ⇒ more persistent + more self-consistent (self-baseline B 0.74 → 0.90 → 0.97). Caveats: persistence is generation-length-sensitive; bf16 LoRA is NaN-prone at lr ≥ 3e-5 (use ≤ 1e-5). |
| **Roster readiness** | **All 10 roster models are multimodal (6), MoE (7), or hybrid-Mamba (2) — none is the dense text arch validated above.** The FSDP wrap config is `Qwen2DecoderLayer`-only; each family needs its own wrap class + (multimodal ones) text-decoder extraction. **gpt-oss-20b VALIDATED trainable** (2026-06-30): loads through the unchanged backend, single-GPU and 2-GPU-FSDP LoRA both save clean 192-tensor adapters; FSDP genuinely shards (GPU2/3 both ~12.5 GiB peak, symmetric; wrap class `GptOssDecoderLayer`). A *real* gpt-oss cell still needs gpt-oss-rendered disguise data. **Unvalidated: 6 multimodal (Qwen3.5/3.6, gemma-4) + Nemotron-H (Mamba).** |

Metric caveat: the headline `persistence` is a **difference-of-means** projection
(basis-independent), not the Fisher-LDA "supervised" axis; `sep_ratio` is a geometry diagnostic,
not the trust gate (the gate is probe CV accuracy ≥ 0.70). Verified definitions in
`dementor/metric/behavioral_inertia_metrics.py:253-262` and `latent_behavior_axes.py:668-686`.

---

## First roster cell — gpt-oss-20b → Qwen3-8B (2026-06-30)

The first genuine roster persistence cell trained fully locally (no Tinker, source = a real roster
model). Single-GPU bf16 sufficed (21B MoE dequantizes to ~42 GiB, fits one H100). Trustworthy:
sep_ratio 0.937, probe_cv 0.975, all controls present.

| anchor / rung | persistence | movement | z vs baseline |
| --- | --- | --- | --- |
| identity I (Qwen3-8B) | 0.053 | — | — |
| **DPO** | **0.731** | 0.269 | 1.56 |
| baseline B (gpt-oss self) | 0.866 | — | — |
| **SFT** | **0.959** | 0.041 | 0.04 |

**Finding:** DPO disguises gpt-oss toward Qwen3-8B (z=1.56 below baseline) while **SFT does not**
(z=0.04, at baseline) — but gpt-oss **retains most of its fingerprint either way** (DPO floor 0.73 vs
the "launder"-tier Qwen stand-in's ~0.38), matching gpt-oss's roster `tier: retain` tag.
**Honest caveat:** the per-prompt paired SFT-vs-DPO test is NOT significant (Δ=0.016, p=0.68; bootstrap
CIs wide/overlapping at n=60) — the robust claims are "DPO moves vs baseline, SFT doesn't" and the
retain-tier persistence; tightening the SFT↔DPO gap needs more eval prompts. gpt-oss specifics handled:
harmony final-channel extraction (0 token leaks / 360 gens), `reasoning_effort=low`, MoE LoRA on
attention q/k/v/o (fused expert tensors skipped). Artifacts: `/data/ethantsliu/roster/gptoss_cell/`.
Note: `download_roster.py` originally dropped `chat_template.jinja` (now fixed to include `.jinja`/
templates) — gpt-oss ships no chat_template in the weights, so the harmony template was fetched separately.
