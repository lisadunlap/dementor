# Dementor — does a model's behavioral fingerprint survive disguise?

Open-weight LLMs carry an **involuntary behavioral fingerprint** — verbosity, markdown/list/LaTeX
habits, sentence shape — that identifies which model wrote a text. **Dementor asks: when you push a
*source* model to imitate a *target*, how much of the source's own fingerprint survives?** — and what
that means for model-provenance auditing and for safety.

## How we measure it

A **disguise ladder** of escalating interventions, scored by a **judge-free persistence metric**:
project held-out responses onto the source→target axis (**standardized difference of class means**
over MiniLM style-descriptor scores + 32 hand-coded style features), anchored between a self-baseline
(≈1 = fingerprint intact) and an identity control (≈0 = fully laundered to the target), with a
per-cell separability *trust gate*. Deterministic, basis-independent. Definition:
[`docs/evaluation_framework.md`](docs/evaluation_framework.md).

**Ladder (weak → strong):** `just_name_it` → `random_sampling` → `stylistic` (prompt-only) → **SFT**
→ **DPO** (LoRA weight edits) → **activation steering** (inference-time; `dementor/steering/`).
Matrix: 4 sources × 4 targets × 3 datasets (`llama-3.1-8b`, `qwen3.6-27b`, `gpt-oss-20b`,
`nemotron-nano-30b-a3b`; gsm8k / writingprompts / chatbot_arena). Every committed result is on this
**executed 4-model matrix**; `config.yaml`'s larger roster is a planned, unexecuted scale-up.

## What we find

1. **Disguise scales with how hard you push; the trained rungs erase most.** Persistence falls
   monotonically down the ladder — naming ~0.92 (barely disguised) → prompting ~0.45 (~half) →
   SFT ~0.37 → **DPO ~0.16** (mostly erased). Prompt-only rungs *under*-disguise — they move the
   fingerprint about halfway, not to zero — and the largest erasure comes from weight-level edits
   (SFT/DPO), **but not exclusively**: inference-time activation steering disguises just as well
   with no weight change (finding 6). (Figure-1 scaffold, not the headline.)
2. **But erasure is source-dependent, not universal.** DPO floors 2/4 sources (llama 0.01, qwen 0.08
   — *launderers*) and leaves a seed-stable residue for 2/4 (gpt-oss 0.19, nemotron 0.21 —
   *retainers*). Which models retain is a property of the **source**, not the target, domain, or size
   (the 27B launders while the 20B resists — capacity is ruled out).
3. **Durability is unpredictable.** Every intuitive predictor fails: output distinctiveness *inverts*
   (r = −0.31 in a MiniLM-free space), model size doesn't predict, a Big-Five personality probe is
   null, and "capability drives durability" is a single-dataset mirage. → We frame this as an
   **auditing negative result**: black-box provenance tools that rely on involuntary style carry a
   **model-dependent false-negative risk.** (Honest scope: n = 4 sources; *why* gpt-oss/nemotron
   retain is mechanistically open.)
4. **"Disguise erodes safety" — real, but target-conditioned and heavily metric-dependent.** Benign
   imitation (gsm8k/writingprompts/chat — *no* harmful data) can erode refusals via a **refuse-then-leak**
   pattern (refuse, then deliver the methodology under an "educational" frame). Two caveats the raw
   numbers hide: **(a) off-the-shelf safety metrics overcount ~10×** — on 808 harmful prompts
   (AdvBench + StrongREJECT) the flagship self-similar DPO shows Guard-unsafe 9.0% → 16.5% (McNemar
   **p = 1.4e-7**, 3-seed replicated), but a validated refuse-then-leak judge puts *genuine*
   methodology-leak at only ~1–5% (Guard over-flags refusal-led text). **(b) The genuine effect is
   target-driven, not disguise-general** — across the roster, genuine erosion appears almost entirely
   when impersonating a *less-safe* target (→llama-3.1-8b: 0.12% → 3.92%, **p = 1.2e-25**); the other
   three targets are null. You partly inherit the safety profile of the model you imitate.
   Details + caveats: [`docs/experiment_review_status.md`](docs/experiment_review_status.md).
5. **Capability rides along, independently** (⊥ style, r ≈ −0.06): a disguise can read style-clean yet
   still transfer +21 MATH-500 points — a provenance **false-clean**. (Honest scope: n = 10 gap cells.)
6. **Activation steering (Rung 5).** Projection-ablation `h −= β·(h·v̂)v̂` reaches DPO-level
   persistence with *coherent* target-style text; naive additive steering only over-steers into
   gibberish. A training-free disguise lever at a ~1.5× fluency cost.
   [`docs/activation_steering_plan.md`](docs/activation_steering_plan.md).
7. **Identity is steerable; safety is not — a dissociation.** The same steering operator that erases the
   fingerprint (finding 6) does **not** erode safety: across both operators and layers 4/8/14/20 it
   leaves genuine-harm at the random-direction-control level, while DPO *weight* imitation of the same
   target does erode it (finding 4). So the fingerprint/identity direction and the safety disposition are
   **separately represented** — imitation-induced erosion is a weight-update effect, not carried by the
   steerable identity direction. Novelty is real but *contested* by concurrent persona work; status +
   closest prior art: [`docs/novelty_dissociation_lit_review.md`](docs/novelty_dissociation_lit_review.md).

Current direction + full caveats: [`docs/paper_direction.md`](docs/paper_direction.md), with honest
per-finding status in [`docs/experiment_review_status.md`](docs/experiment_review_status.md).
(`docs/strategy.md` is the earlier fingerprint-era memo — superseded as the lead.)

## Reproduce

Results/datasets/figures are in **git-LFS** — a fresh clone has only pointer stubs, so pull the real
content (~2 GB) first. Editable install with `uv` (the venv has no bare `pip`).

```bash
git lfs install && git lfs pull
uv venv && uv pip install -e .          # TINKER_API_KEY / OPENAI_API_KEY go in .env
PY=./.venv/bin/python

# One cell end-to-end (Tinker): generate the 5 rungs, score, print the calibrated ladder.
$PY -m dementor.metric.run_cell_pipeline --dataset gsm8k \
    --source nemotron-nano-30b-a3b --target gpt-oss-20b --eval-size 200 --adapter-seeds 3

# Re-run the headline analyses on cached text (no GPU): de-confounded 36-cell re-eval + multi-seed CIs.
$PY -m experiments.analysis.decontaminate --adapter-seeds 3

# Local (non-Tinker) SFT/DPO for large / backend:local roster models — single- or multi-GPU FSDP.
$PY -m dementor.training.matrix cell --source <id> --target <id> ...   # dementor/training/README.md

# Safety-constrained imitation mitigation scaffold: target imitation plus refusal replay.
# Data builders are free; launch commands support dry-run and must not be run for real without approval.
$PY -m dementor.training.matrix build-safety-sft-data --dry-run --max-cells 1
$PY -m dementor.training.matrix build-safety-dpo-data --dry-run --max-cells 1
$PY -m dementor.training.matrix launch-safety-sft --dry-run --max-cells 1
$PY -m dementor.training.matrix launch-safety-dpo --dry-run --max-cells 1

# Activation-steering rung (writes a scorable rung CSV).
$PY -m dementor.steering.steering_rung ...

pytest -q
```

## Where things live

- **`config.yaml`** — roster, datasets, seeds, hyperparameters (single source of truth).
- **`dementor/`** — `metric/` (persistence scorer + trust gate), `methods/` (prompt-disguise rungs),
  `training/` (Tinker + local SFT/DPO backends, multi-GPU FSDP), `steering/` (activation-steering rung).
- **`data/results/`** (working, git-LFS) + **`results/`** (trimmed paper hand-off) — per-cell
  persistence CSVs + figures. Spine: `data/results/multiseed_ci_s3.csv`.
- **`data/results/matrix/safety_sft_data` / `safety_dpo_data`** — constrained-imitation training
  CSVs plus `.manifest.json` audit files; corresponding launch outputs go under
  `safety_sft_runs` / `safety_dpo_runs`.
- **`docs/`** — `paper_direction.md` (current direction, read first), `evaluation_framework.md` (metric),
  `experiment_review_status.md` (honest per-experiment status), `novelty_dissociation_lit_review.md`
  (novelty vs prior art), `matrix_run_log.md` (training log). `strategy.md` is the earlier memo (superseded).
- **`experiments/analysis/`** — the de-confound / robustness re-analyses behind the caveats above.
- **HuggingFace `dementor-research`** — 228 LoRA adapters + the `dementor-matrix-responses` dataset
  (external reproduction; the analysis pipeline itself is HF-independent).

Coding conventions: [`AGENTS.md`](AGENTS.md). Branch: `ethan` (PRs → `main`).
