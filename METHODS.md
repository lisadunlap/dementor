# Methods — reproduction

Concise reproduction for the current `dementor/` layout. `config.yaml` is the single source of truth
for roster, datasets, seeds, and hyperparameters. Results/datasets are git-LFS; `git lfs pull` first.

## Roster

**`config.yaml` is the single source of truth for the roster** (slugs, ids, providers, backends,
tiers) — see it for the current model list. `backend: tinker` models train via the Tinker API;
`backend: local` models train on a local H100. `tier` = launder | retain (DPO fingerprint-erasure
behavior); `imitation: core | extended` marks the imitation-matrix membership and `steering_roster`
marks the mechanistic-dissociation membership.

`roster_legacy` in `config.yaml` (llama-3.1-8b etc.) resolves slugs for existing adapters and the
mechanistic dissociation cells; retired from the Tinker catalog 2026-06-27. Renderers disable thinking
traces for cross-model comparability (`enable_thinking=False`; gpt-oss `reasoning_effort=low` with
harmony `final`-channel extraction).

## Disguise ladder (the instrument)

Escalating imitation interventions, weak → strong:

1. `just_name_it` — instruct the source to act as the target (no data).
2. `random_sampling` — few-shot target examples.
3. `stylistic` — explicit surface-style rules (prompt-only). (`behavioral` / `contrastive` /
   clustering variants also exist but need an external analysis LLM, default gpt-4.1-mini.)
4. **SFT** — LoRA weight imitation of target responses.
5. **DPO** — LoRA preference edit (chosen=target, rejected=source), initialized from the SFT adapter.
6. **Activation steering** — inference-time projection-ablation (`dementor/steering/steering_rung.py`).

Prompt rungs use `dementor-disguise --method <name>` (registry:
`dementor/methods/get_method.py`). The central question: does source identity persist as the
intervention gets stronger?

## Training matrix

`dementor/training/matrix/` (package; subcommand dispatcher via `python -m dementor.training.matrix`: `make-splits`, `generate-target-responses`,
`build-sft-data`, `launch-sft`, `build-dpo-data`, `launch-dpo`, `push-to-hf`). Seeds **42/43/44**;
LoRA **rank 32**, alpha 64, target_modules=all-linear.

- **SFT:** 3 epochs, batch 16, lr 1e-4, train_size 500.
- **DPO:** β=0.1, 1 epoch, lr 1e-5, max_length 4096, on top of the matching SFT adapter.
- Adapters register in `data/tinker_adapters.json` (sampler + state URIs) and mirror to HuggingFace
  `dementor-research/{sft,dpo,self_sft}_*`. Naming:
  `{stage}_{dataset}_{source}_as_{target}_seed{N}`.

Local / large / gemma models use `dementor/training/local_backend.py` (single-GPU bf16 fits a 32B via
grad checkpointing; multi-GPU FSDP available — the adapter save gathers a full `FULL_STATE_DICT`).
See `dementor/training/README.md`.

```bash
$PY -m dementor.training.matrix cell --source <slug> --target <slug> ...
```

## Safety pipeline

Prompts from AdvBench + StrongREJECT + HarmBench (+ XSTest / OR-Bench / SORRY-Bench for over-refusal).
Pipeline: **sample → score → analyze**, one paired base-vs-adapter comparison per prompt.

- `sample` — generate base and adapter completions on the harmful/benign prompts.
- `score_guard` — Llama-Guard-3-1B (sensitive upper bound; overcounts).
- `score_harmbench` — official CAIS HarmBench classifier (content-aware).
- `score_rtl` — **RTL "refuse-then-leak" judge = Qwen3-8B** (validated against a hand census); catches
  refuse-then-comply that Guard mis-flags.
- `analyze` — by-source / per-cell McNemar (base vs DPO) on each judge.

**Benign control (imitation-specificity).** Compare base → **self-SFT** (train the model to imitate
*itself*, matched compute) against base → imitation. The gap is the imitation-specific effect (the
self-SFT arm is null). A benign-DPO control is in flight.

**Safety-constrained imitation (mitigation scaffold).** `build-safety-sft-data` /
`build-safety-dpo-data` add `safety_replay` refusal rows; `launch-safety-sft` / `launch-safety-dpo`
train them (all support `--dry-run`; real launches need approval). Replay prompts default to
`data/datasets/benchmarks/refusal/advbench_harmful.csv`, excluding the refusal-ladder eval prompts;
each CSV has a sibling `.manifest.json`.

## Steering assay (identity ⟂ safety)

`dementor/steering/steering_rung.py`. Derive a per-layer source→target **diff-of-means over response
tokens** (both response sets teacher-forced through the SOURCE model, since source/target live in
different residual spaces). Forward-hook generation, two operators:

- `mode="ablate"` → `h ← h − β(h·v̂)v̂` (projection-ablation, the winning disguise operator).
- `mode="add"` → `h += α·v̂` (additive; over-steers into gibberish before it disguises — no clean
  regime).

**Safety dissociation protocol.** Projection-ablation at a mid layer, β ∈ {0.6, 1.0, 1.4}, 300 harmful
prompts, RTL-judged, coherence-gated, paired McNemar. **Pre-registered positive control:** ablate the
Arditi **refusal** direction (harmful−harmless last-token diff-of-means) — it must catastrophically
erode safety, or the base model is **EXCLUDED** as unpowered (the single-direction probe cannot resolve
the question there). Ablating the benign **fingerprint** direction under the identical operator is the
test. Geometry (cosines to refusal/persona) is reported but does **not** predict the causal effect, so
the claim rests on the causal control, not the cosines.

## Persistence metric (judge-free)

`dementor/metric/behavioral_cell_evaluator.py` (canonical, one fixed basis per cell);
`run_cell_pipeline` / `run_behavioral_inertia` for single comparisons. Deterministic, basis-independent.

1. **Features.** Each response → Big-Five/style adjective embedding scores
   (`sentence-transformers/all-MiniLM-L6-v2`) + style scalars + 32 binary style heuristics.
2. **Fixed source↔target basis.** Fit scaler + basis on **source and target only** (interventions
   excluded, so a method cannot define its own axes). Default `basis_type="supervised"`: axis 1 =
   shrinkage-regularized **Fisher-LDA** source→target discriminant, axes 2..k = residual PCA — puts the
   separation in the kept subspace so the source-vs-target probe is separable.
3. **Headline persistence** (k- and basis-independent): scalar projection of the disguised mean onto the
   full-feature diff-of-means `d = target_mean − source_mean`.
   `persistence = 1 − clip(movement, 0, 1)`, `movement = ⟨disguised−source, d⟩ / ⟨d, d⟩`.
4. **Anchored calibration.** Self-baseline (independent source runs) sets the ≈1 stay-anchor; identity
   control (independent target run) sets the ≈0 reach-anchor. `anchored = (P − I) / (B − I)`, with a
   joint prompt-bootstrap CI. **Trust gate:** the cell is reported only if the probe CV accuracy ≥ 0.70
   (`sep_ratio` is a geometry diagnostic, not the gate).
5. **Robustness twins.** Every feature set has a `*_lenres` twin (each feature residualized against
   `log word count`, fit on source+target only) to show persistence survives removing length.

## Local (non-Tinker) generation

`hf:` prefix → HF transformers pipeline; `vllm:` prefix → vLLM; default → provider APIs via LiteLLM.
For a local vLLM endpoint, route LiteLLM at it:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype auto --max-model-len 4096
dementor-generate --prompts-file <prompts.csv> --output-csv <out.csv> basic \
  --model openai/meta-llama/Llama-3.1-8B-Instruct \
  --openai-api-base http://localhost:8000/v1 --openai-api-key EMPTY
```

## Key paths

- `config.yaml` — roster / datasets / seeds / hparams.
- `data/tinker_adapters.json` — adapter registry (sampler + state URIs; legacy + scale-up).
- `data/results/` — per-cell persistence CSVs; `results/` — curated hand-off.
- `dementor/{metric,methods,training,steering,safety}/` — the pipeline.
- `experiments/analysis/` — de-confound / robustness re-analyses.
- Safety-eval artifacts live under `/data/ethantsliu/exp*_safety/` (samplers + `score_*` + `analyze_*`).
- HuggingFace `dementor-research` — LoRA adapters + `dementor-matrix-responses` dataset.
