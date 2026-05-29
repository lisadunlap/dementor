# Fingerprint-Persistence Measurement Plan (Phases A.3 → F)

The fine-tuning (rungs 3–4) is done — 228 adapters trained + on HF. This plan
covers the actual measurement: generating outputs at every rung, scoring them
with the behavioral-inertia pipeline, and producing the headline figure.

## What we measure

For each `(source, target, dataset)` triple and each intervention rung, we
measure **`persistence`** — how much of the source model's behavioral fingerprint
remains after it's been pushed to imitate the target. It is the k-independent
full-feature source→target projection (`1 - movement`), measured against a
supervised LDA basis and rescaled by two controls (self-baseline ≈1, identity
control ≈0) into `anchored`. The original hypothesis was that persistence
decreases monotonically across rungs but stays above zero even after DPO.

> **Status (first cells).** The pipeline is operational
> (`scripts/analysis/run_cell_pipeline.py`); the headline figure still needs the
> full matrix. First cells (llama-3.1-8b → {gpt-oss-20b, qwen3.6-27b,
> nemotron-30b}, gsm8k; + gpt-oss on writingprompts) already complicate the prior:
> **DPO drove persistence to ≈0 (full assimilation, sometimes overshoot) on every
> target and both datasets**, and the prompting ladder is not monotonic. So
> "stays above zero after DPO" does not hold for these pairs — to be confirmed
> across more sources/datasets.

Rungs:
- **Rung 0 (self-baseline):** inter-seed variance, the noise floor
- **Rungs 1–2 (prompting/examples):** 7 disguise methods on the *base* source model
- **Rung 3 (SFT):** the 108 SFT adapters
- **Rung 4 (DPO):** the 108 DPO adapters
- **Rung 5 (activation steering):** optional, needs local GPU

## Phase breakdown

### A.3 — Eval baselines (prerequisite for all scoring)
Generate each model's own responses on every eval split. These are the
`source` and `target` reference distributions, and (across seeds) the tier-0
self-baseline.
- 4 models × 4 eval datasets × 3 seeds
- Eval sizes: gsm8k 1000, arena 1000, wp 500, humaneval 164 → 2664/seed
- **~32k generations**

### Rungs 1–2 — Prompting/example-selection grid
For each `(source, target, dataset)`, run the 7 disguise methods on the base
source model over the eval prompts. Output = source-disguised-as-target.
- 7 methods × 4 src × 3 cross-tgt × 4 eval datasets
- contrastive/clustering need source+target examples (we have baselines)
- **~670k generations** (full) — the biggest bucket

### Phase E — Eval inference on adapters (rungs 3–4)
Run each adapter on the eval splits. Output = the disguised distribution for
that rung.
- 216 adapters (+12 self) × eval splits
- In-distribution (train_ds == eval_ds) minimum; + HumanEval cross-dist
- **~180k (in-dist) to ~575k (full cross-dist) generations**

### Phase F — Scoring (CPU, no API cost)
For every cell, run `scripts/analysis/run_behavioral_inertia.py`:
- Inputs: source baseline, target baseline, disguised output
- Outputs: `source_persistence`, probe (`source_residue`/`target_assimilation`),
  per-axis movement, anisotropy
- Tier-0 normalization from the self-baseline
- Aggregate via `run_intervention_ladder.py`

### Calibration (small) — LLM judge
GPT-5 on ~500 sampled (prompt, disguised, target) triples to show the
deterministic metric correlates with human-aligned style judgment.

### Rung 5 — Activation steering (optional)
Local Transformers + base model + hidden-state hooks. Heavy (needs GPU for
27–30B models). Defer unless explicitly wanted.

## Cost / scale knobs

The full measurement is ~900k–1.3M generations — dominated by rungs 1–2 and
Phase E cross-dist. Tinker inference at parallel-4 (~1–3 gen/s) ≈ several days.

Reduction levers:
- **1 seed instead of 3** for rungs 1–2 → 3× fewer there
- **In-distribution eval only** (skip cross-dataset OOD except HumanEval) → ~3× fewer in Phase E
- **Subsample eval prompts** (e.g. 300 instead of 1000) for a fast first pass
- **Subset of methods** for rungs 1–2 (e.g. just_name_it + contrastive + embedding_clustering as representative weak/medium/strong)

## Proposed execution order

1. A.3 eval baselines (unblocks everything)
2. Phase E in-distribution (rungs 3–4 — we already have the adapters)
3. Phase F scoring on rungs 3–4 → first persistence numbers for the strongest rungs
4. Rungs 1–2 grid (the big generation bucket)
5. Phase F scoring on rungs 1–2 → full ladder
6. HumanEval cross-dist eval + scoring → generalization claim
7. Headline figure
8. (optional) rung 5
