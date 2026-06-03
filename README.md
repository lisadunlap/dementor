# Dementor — measuring whether a model's behavioral fingerprint survives disguise

> **Collaborator onboarding.** This README brings a new contributor up to speed on what the
> project is, the *honest* current headline, the recommended framing, where the artifacts live,
> and how to reproduce one result. For the authoritative direction, read
> [`docs/strategy.md`](docs/strategy.md) first — it supersedes the older `docs/argument.md` and
> `docs/aaai_plan.md` on every point of conflict.

## What this is

Open-weight LLMs carry an **involuntary behavioral fingerprint** — stylistic/structural
regularities (verbosity, markdown habits, list/LaTeX use, sentence shape) that make one model's
outputs identifiable from another's. Dementor asks a robustness question: **when you push a
*source* model to imitate a specific *target* model, how much of the source's own fingerprint
survives?**

We answer it with a **disguise ladder** of escalating interventions and a **calibrated, judge-free
persistence metric**. The metric projects held-out responses onto a supervised Fisher-LDA
source→target axis (over MiniLM adjective-descriptor scores + 32 hand-coded style features),
anchored between a self-baseline (≈1, the fingerprint fully intact) and an identity control
(≈0, fully laundered to the target), with a per-cell separability gate that refuses untrustworthy
cells. It is deterministic and basis/k-independent by construction. Canonical definition:
[`docs/evaluation_framework.md`](docs/evaluation_framework.md).

### The 4×4×3 disguise ladder (36 cells)

- **4 source × 4 target models** = 12 ordered source→target pairs (diagonal = self-baseline):
  `llama-3.1-8b`, `qwen3.6-27b`, `gpt-oss-20b`, `nemotron-nano-30b-a3b`.
- **3 datasets**: `gsm8k` (math), `writingprompts` (creative), `chatbot_arena` (open chat).
- **5 rungs of increasing disguise strength** per cell: `just_name_it` (tell the model to act as
  the target) → `random_sampling` (few-shot target exemplars) → `stylistic` (style prompt) →
  **SFT** (LoRA toward target outputs) → **DPO** (preference tuning on top of SFT).

All 180 cell-rungs pass the trust gate. Persistence collapses monotonically with intervention
strength — **0.915 (naming) → 0.436 / 0.485 (prompting) → 0.369 (SFT) → 0.155 (DPO)** — i.e.
naming alone never disguises; only weight-level edits move the fingerprint, and **DPO is the
strongest eraser.** This monotone ladder is the experimental scaffold and Figure 1, not the
headline.

## The honest current headline

**DPO erasure of the behavioral fingerprint is model-dependent and source-driven — not universal.**
One epoch of LoRA-DPO drives source-vs-target separability to the floor for two of four models and
leaves a robust, seed-stable structural residue for the other two. Verified this session against
`data/results/multiseed_ci_s3.csv`:

| source model | per-source DPO persistence | tier |
|---|---|---|
| nemotron-nano-30b-a3b | **0.211** | retains |
| gpt-oss-20b | **0.190** | retains |
| qwen3.6-27b | **0.077** | launders |
| llama-3.1-8b | **0.012** | launders |

- The two tiers are separated by a gap ~3× the within-tier spread; **seed-sd median 0.018** (≈20×
  smaller than the survivor effect — the metric is highly reproducible across the 3 adapter seeds).
- **Survivor enrichment is real but small-n**: 7/36 DPO cells survive (point estimate > 0.3), *all*
  gpt-oss/nemotron-sourced (Fisher p ≈ 0.008). The exact count is definition-fragile (7 by point
  estimate, 3 by a seed-sd lower bound).
- **Which models retain is a property of the *source* model** — not the imitation target, the
  domain, or model size: the 27B model (qwen) launders while the 20B (gpt-oss) resists, so capacity
  is ruled out.
- **The residue is structural style, not personality**: a Big-Five logprob probe is null
  (mean |Cohen d| ≈ 0.14, 0/900 cell-facet pairs reach even a medium 0.5 effect); the faint
  positive signal lives in verbosity/structure style axes.

### What did NOT survive stress-testing (do not lead with this)

An earlier thesis — *"a model's fingerprint resists DPO in proportion to how distinctive its
outputs are"* — produced a striking r ≈ 0.97. **It did not survive.** Recomputing distinctiveness
in a **zero-MiniLM, 32-feature structural space** (the circularity-killer test,
[`scripts/analysis/distinctiveness_structural.py`](scripts/analysis/distinctiveness_structural.py))
**inverts the relationship: r = −0.31**, and llama — the cleanest launderer (0.012) — is the *most*
structurally distinctive model (0.577). The r=0.97 lives only in the MiniLM space the persistence
metric is ~81% built from, i.e. exactly the circularity it was meant to kill. The "n=36 cell-level"
backbone is also pseudo-replicated (the source-distinctiveness predictor has only 4 unique values;
effective n is 4). **The mechanism for the source-driven split is therefore open.** Treat
`source_fingerprint_figure.py` / `docs/argument.md` as the *superseded* distinctiveness story.

## Recommended framing

Sell this as an **auditing / provenance negative result**, not as a watermarking or
distinctiveness-law paper (see `docs/strategy.md` §1–2 for the full argument and venue logic):

> Black-box behavioral fingerprinting is proposed for model-provenance / distillation auditing.
> We stress-test it under adversarial imitation (a disguise ladder up to DPO) and show its
> reliability is **source-model-dependent and not predictable from output distinctiveness**: for
> half of four open models, one epoch of LoRA-DPO drives the signal to the floor. Provenance tools
> relying on involuntary style therefore carry a **model-dependent false-negative risk.**

The claim is *existence + unpredictability*, not a graded law — which survives the n=4 limitation
honestly. Two cheap mechanism analyses now sharpen *what* survives and gate the expensive next step:

- **Structural decomposition (D4):** the surviving residue lives in **document-formatting** features
  — math/LaTeX notation (survivor−launderer residue-gap **0.50**), numbered lists (0.45), markdown
  (0.39) — **not** raw verbosity (word-count gap 0.003) and **not** Big-Five personality (null
  probe). (`scripts/analysis/structural_decomp.py`)
- **Activation-bridge probe (e5-small-v2, CPU, 36 cells):** an independent encoder + logistic probe
  recovers the **same source ordering** as the MiniLM persistence metric (per-source residue vs
  persistence **Pearson r=0.978, ρ=1.0**) — so the two-tier split is **not a MiniLM artifact**. But
  the residual signal in the output text is weak (survivor residue 0.17, survivor−launderer gap
  0.10), so the verdict for activation **steering is NO-GO** on this evidence: a native-internals
  probe (GPU) is the next gate, not a steering run. (`scripts/analysis/bridge_decontam_driver.py` →
  `data/results/bridge_decontam/verdict.json`)

The mechanism (*why* gpt-oss/nemotron retain) remains open at the causal level; `docs/strategy.md`
recommends a Findings/workshop target now.

## Where things live

- **Branch**: `ethan` (current working branch; `main` for PRs).
- **HuggingFace org `dementor-research`** (the canonical hand-off; any `ethantsliu/...` reference is
  stale):
  - **228 LoRA adapters** (SFT + DPO + self-SFT per cell × seed): https://huggingface.co/dementor-research
  - **Dataset** `dementor-matrix-responses` (all matrix generations):
    https://huggingface.co/datasets/dementor-research/dementor-matrix-responses
  - The analysis pipeline is **HF-independent** — it reads Tinker sampler paths from
    `data/tinker_adapters.json`, not HF. The HF artifacts are for external reproduction.
- **Docs**: [`docs/strategy.md`](docs/strategy.md) (authoritative decision memo — read first),
  [`docs/argument.md`](docs/argument.md) (the *superseded* distinctiveness write-up; kept for the
  objection/rebuttal table), [`docs/aaai_plan.md`](docs/aaai_plan.md) (paper plan; its
  distinctiveness/"n=36" framing is superseded by strategy.md),
  [`docs/evaluation_framework.md`](docs/evaluation_framework.md) (canonical metric definition).
- **Key analysis scripts** (`scripts/analysis/`):
  - `run_cell_pipeline.py` — end-to-end runner for **one** cell (generate → stage → score → ladder).
  - `decontaminate.py` — re-evaluates all 36 cells on chat-template-CLEANED cached text (the D1
    gpt-oss CoT-leak de-confound; no regeneration). Supports `--adapter-seeds` for multi-seed CIs.
  - `gen_multiseed.py` — generates seed2/seed3 SFT+DPO eval outputs for the CIs.
  - `big5_directions.py` — named-direction probes (Big-Five personality vs style axes; logprob
    backend).
  - `source_fingerprint_figure.py` / `distinctiveness_structural.py` — the (superseded) MiniLM
    distinctiveness figure and the zero-MiniLM structural recomputation that **inverts** it.
  - `behavioral_cell_evaluator.py`, `latent_behavior_axes.py`, `behavioral_inertia_metrics.py` —
    the metric internals (supervised basis, anchors, style features).
  - `activation_bridge.py` + `bridge_decontam_driver.py` — the fixed-encoder mechanism probe, **run**
    over 36 cells (CPU): cross-encoder corroboration r=0.978, steering verdict NO-GO.
  - `structural_decomp.py` — D4 feature decomposition (residue = document-formatting structure).
  - `activation_steering.py` — built, **not run** (GPU capstone; gated NO-GO by the bridge probe).
- **Key data artifacts** (`data/results/`):
  - `multiseed_ci_s3.csv` — **the spine**: per-cell DPO persistence, 3-seed mean ± CI, seed-sd 0.018.
  - `decontam/` + `decontam_before_after.csv` — de-confounded re-eval text/results.
  - `big5_directions_matrix_{,_style_}logprob.csv` — personality-null / style-axis tables.
  - `structural_decomp_byfeature.csv` — D4: per-feature survivor−launderer residue gaps.
  - `bridge_decontam/verdict.json` + `per_source_summary.csv` — activation-bridge mechanism probe
    (cross-encoder corroboration r=0.978; steering NO-GO).
  - `fig1_source_fingerprint.png` — the (superseded-thesis) source figure.
- **Curated paper-handoff subset** (`results/`, committed via LFS): the small,
  paper-relevant CSVs + figures lifted out of `data/results/` — `matrix_ladder/` (per-rung
  ladder per dataset), `d2_multiseed_ci.csv`, `d1_decontam_before_after.csv`,
  `d3_encoder_swap_bge.csv`, `source_distinctiveness.csv`, `style_directions.csv` /
  `big5_personality_directions.csv`, `fig1_source_fingerprint.png`. See
  [`results/README.md`](results/README.md) for the index. (`data/results/` above is the
  working tree the scripts read/write; `results/` is the trimmed external hand-off.)
- **Workflows** (`workflows/`): `run_matrix.py` (matrix runner; holds `MODEL_SLUG`,
  `clean_response`, chat-template kwargs), `run_gsm8k_workflow.py` (canonical SFT/DPO entry point),
  `tinker.py` (Tinker LoRA backend + adapter registry).

> **Encoder-robustness (two independent checks, both pass):**
> - **Activation-bridge probe** — e5-small-v2 (a different encoder *and* probe method) recovers the
>   same source ordering as MiniLM at **r=0.978**.
> - **Encoder-swap (D3, `encoder_swap.py` → `encoder_swap_bge-small-en-v1.5.csv`)** — re-scoring the
>   full ladder under **bge-small-en-v1.5** (35/36 cells; the gsm8k qwen→gpt-oss cell errored)
>   reproduces the ladder almost exactly: rung means within 0.03 of MiniLM at every rung (DPO floor
>   **0.116 vs 0.119**; overall Spearman 0.80). The exact per-cell DPO survivor *set* is
>   encoder-sensitive (Jaccard 0.38, ~3/6 robust across encoders) — so the load-bearing claim is the
>   **tier-level split**, not exact per-cell survivor identity. (The earlier buggy
>   12-cells-triplicated mpnet CSV has been removed.)

## Reproduce

Use the project venv and set `PYTHONPATH`. `TINKER_API_KEY` (generation) and `OPENAI_API_KEY`
(logprob probes) live in `.env`. The venv is a `uv` venv (no `pip`); install with
`VIRTUAL_ENV=.venv uv pip install <pkg>`.

```bash
export PYTHONPATH=/Users/EthanLiu/Documents/Programming/dementor
PY=./.venv/bin/python   # arm64 venv with numpy/pandas/scipy/sklearn/sentence-transformers
```

**Run one cell end-to-end** (generate all 5 rungs via Tinker, stage on shared endpoints, score,
print the calibrated ladder). This is the cleanest zero-CoT-leak survivor:

```bash
$PY -m scripts.analysis.run_cell_pipeline \
  --dataset gsm8k --source nemotron-nano-30b-a3b --target gpt-oss-20b \
  --eval-size 200 --adapter-seeds 3
# idempotent: cached outputs with the right row count are skipped, so reruns only fill gaps.
```

**Re-run the headline analyses on cached text (no regeneration, no GPU):**

```bash
# De-confounded 36-cell re-eval + multi-seed CIs -> writes multiseed_ci_s3.csv
$PY -m scripts.analysis.decontaminate --adapter-seeds 3

# Named-direction probes: Big-Five personality (null) vs style axes (the faint positive)
$PY -m scripts.analysis.big5_directions --matrix --axes big5  --backend logprob --max-prompts 40 --workers 16
$PY -m scripts.analysis.big5_directions --matrix --axes style --backend logprob --max-prompts 40 --workers 16

# The (superseded) MiniLM distinctiveness figure ...
$PY scripts/analysis/source_fingerprint_figure.py
# ... and the zero-MiniLM structural recomputation that INVERTS it (prints r = -0.31)
$PY scripts/analysis/distinctiveness_structural.py
```

Tests (no pytest installed): `$PY -m unittest scripts.tests.test_behavioral_inertia`.

---

For coding conventions see `AGENTS.md`; for local HF/vLLM/provider routing see
`docs/local_generation.md`; for SFT/DPO orchestration see `workflows/README.md`. **Everything about
direction and current status is in `docs/strategy.md`.**
