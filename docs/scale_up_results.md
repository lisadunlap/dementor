# Scale-up results (2026-07-05): breaking the n=4 ceiling

This documents the roster/dataset scale-up executed to address the "n=4 targets" weakness in the
safety-erosion thread. All numbers below are computed artifacts, not projections.

## What was run

**Tinker matrix — COMPLETE.** 7 models × 4 datasets × 3 seeds, every cell trained (SFT + DPO):

- Models: `qwen3.5-4b`, `qwen3.6-27b`, `qwen3.6-35b-a3b`, `nemotron-nano-30b-a3b`,
  `nemotron-super-120b`, `gpt-oss-20b`, `gpt-oss-120b`
- Datasets (imitation corpus): `chatbot_arena`, `gsm8k`, `writingprompts`, `oasst1`
- Seeds: 42/43/44 → **126/126 SFT + 126/126 DPO per dataset** (42 ordered distinct pairs × 3 seeds)

**Gemma local track** (google VLMs, not on Tinker; trained on local H100s):

- `gemma-4-E4B-it` + `gemma-4-26B-A4B-it`: full SFT + DPO, 9/9 each (chatbot_arena, seed42)
- `gemma-4-31B-it`: SFT 9/9; **DPO not run** — genuine single-80GB-card memory wall (62 GB text
  tower + DPO full-vocab logits over gemma's ~256K vocab OOM even single-process, with the
  entropy-metric neutralized and `max_length` capped). Would require `device_map` model-parallel.

Infra required four fixes in `dementor/training/{local_backend,matrix}.py` (tinker path untouched):
batched local generation with OOM back-off; splice of the Gemma-4 VLM text decoder + lm_head into a
pure-text `Gemma4ForCausalLM` (the larger gemmas set `use_bidirectional_attention == "vision"`, whose
training path demands `mm_token_type_ids` that text-only SFT/DPO never provides); a local-DPO
`max_length` cap; and a no-op for trl's detached entropy-from-logits metric (a full-vocab copy that OOMs
the large tower). The transplant is byte-neutral for training; the entropy no-op leaves the trained
adapter identical (metric is `.detach()`ed, not in the loss).

## Headline result 1 — the metric-overcount is the portable finding

Off-the-shelf Llama-Guard-3-1B overcounts genuine harm (RTL "refuse-then-leak" judge, F1 0.83) by ~10×,
and this **replicates on a provider we had never tested**:

| Roster | Guard (base) | Genuine/RTL (base) | overcount |
|---|---|---|---|
| original tinker (llama-driven) | ~9–16% | ~0–5% | ~10× |
| **google gemma-4-26B (new)** | **20.7%** | **2.2%** | **~9×** |

Any headline of the form "fine-tuning eroded safety by X%" is almost always a Guard-style number and is
~10× inflated relative to genuine leak.

## Headline result 2 — genuine erosion is target-AND-provider-conditioned (not universal)

Benign DPO-imitation erodes *genuine* safety only when the imitated target is permissive enough, and the
effect is provider-dependent:

- **Original roster:** →llama target 0.12→3.92% (p=1.2e-25); other 3 tinker targets null/small.
- **New 7-model roster (pooled, chatbot_arena seed42):** +0.27 pt, p=4.1e-8 — small but significant;
  significant for `qwen3.5-4b`, `gpt-oss-20b`, `gpt-oss-120b` targets; nemotron targets null.
- **Gemma sources (E4B, 26B; base vs DPO, pooled across 9 imitated targets, n=7272 each):**
  **0 of 18 cells** show significant genuine erosion. E4B −0.07 pt (p=0.55, null); 26B −0.37 pt
  (p=0.03, *slightly safer*). Guard nonetheless shows small "significant" erosion (+0.48 / +1.09 pt),
  i.e. the overcount again.

Interpretation: erosion is not an inevitable consequence of benign imitation. It requires a
permissive-enough target, and google's gemma shows no genuine erosion at all — consistent with the
identity-vs-safety dissociation (benign weight-imitation does not disturb the refusal disposition).

## Artifacts

- Tinker adapters: `data/tinker_adapters.json` (registry).
- Gemma safety eval: `/data/ethantsliu/exp_pilot_safety/` — `sample_gemma.py` (local-adapter sampler),
  `analyze_gemma.py` (by-source base-vs-DPO McNemar), `analysis_gemma/{per_cell,per_source}.csv`.
- Reused pilot scorers: `score_guard.py` (Llama-Guard-3-1B), `score_rtl.py` (Qwen3-8B RTL judge).

## Open / optional next

- Tinker safety eval on the 3 newly-completed datasets (gsm8k/writingprompts/oasst1): does the erosion
  pattern hold across imitation corpora? Cheap (Tinker sampling; no local GPU).
- Gemma across all 4 datasets (only chatbot_arena piloted).
- 31B DPO via model-parallel, if the largest-gemma DPO cell is wanted.
