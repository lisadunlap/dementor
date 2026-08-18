# Methods — reproduction

Concise reproduction for the current `dementor/` layout. `config.yaml` is the single source of truth
for roster, datasets, seeds, and hyperparameters. Results/datasets are git-LFS; `git lfs pull` first.

## Roster

**`config.yaml` is the single source of truth for the model catalog and named campaigns** (slugs,
ids, providers, backends, tiers, datasets, and seeds). The publication campaign is
`campaigns.imitation_safety`: 12 models, four datasets, seed 42, 200 evaluation prompts, and a
256-new-token generation cap.
`backend: tinker` models train via the Tinker API;
`backend: local` models train on a local H100. `tier` = launder | retain (DPO fingerprint-erasure
behavior); `imitation: core | extended` marks the imitation-matrix membership and `steering_roster`
marks the mechanistic-dissociation membership.

`roster_legacy` resolves slugs for retired adapters without adding them to a campaign. Renderers disable thinking
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

`dementor/training/matrix/` (package; subcommand dispatcher via `python -m dementor.training.matrix`: `generate-target-responses`,
`build-sft-data`, `launch-sft`, `build-dpo-data`, `launch-dpo`, `push-to-hf`). The publication square
uses seed **42**; seeds 43/44 remain available for explicitly scoped robustness runs.
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
dementor-make-splits
$PY -m dementor.training.matrix launch-local-cell --source <slug> --target <slug> \
  --dataset <dataset> --seed 42 --phase all
```

## Safety pipeline

Five harm benchmarks are used: AdvBench, HarmBench, StrongREJECT, SORRY-Bench, and SG-Bench.
XSTest and OR-Bench-Hard measure over-refusal. Every adapter and its source baseline use the identical
deterministic 200-row, seed-42 subsample. XSTest reports `n=111` because only the benign rows in that
shared sample contribute to its over-refusal metric.
Pipeline: **sample → score → analyze**, one paired base-vs-adapter comparison per prompt.

- `sample` — generate base and adapter completions on the harmful/benign prompts.
- `score_guard` — Llama-Guard-3-1B (sensitive upper bound; overcounts).
- `score_harmbench` — official CAIS HarmBench classifier (content-aware; auxiliary in the matrix).
- `score_rtl` — **RTL "refuse-then-leak" judge = Qwen3-8B**; catches refuse-then-comply that Guard
  mis-flags. Its agreement with HarmBench was checked on a fixed automated-judge sample, not against
  human ground truth.
- `analyze` — by-source / per-cell McNemar (base vs DPO) on each judge.

The publication harm aggregate uses the binary RTL verdict for all five harm benchmarks, both for
each adapter and its paired unadapted-source baseline. Benchmark-native graders remain auxiliary;
mixing a native continuous score with a binary RTL baseline is invalid. Negative erosion means
lower harmful compliance on this measure, not necessarily greater overall safety, because increased
over-refusal can also lower harmful compliance. XSTest and OR-Bench-Hard are therefore reported
separately.

**Target-relative safety transfer.** Source-relative erosion alone cannot determine whether an
adapter approaches its target's safety behavior. For each harm benchmark, define `T = target base
harm − source base harm` and `A = adapter harm − source base harm` on the same seed-42 prompts and
RTL metric. Average both over the five harm benchmarks within each dataset/source/target cell, then
report the origin-constrained slope `sum(T*A)/sum(T^2)` (0 stays at source; 1 reaches target), the
change in absolute target distance `abs(T) − abs(T−A)`, and lower-harm versus higher-harm target
strata. Confidence intervals use a 10,000-draw crossed multinomial bootstrap over source, target,
and training dataset, rather than treating 528 cells as independent.

**Auxiliary self-SFT control.** The complete campaign contains 48 matched-compute self-SFT cells
(12 sources × four datasets) and 336 safety scores. They are not pooled into either 528-cell
off-diagonal stage. The paper uses their mean as a descriptive generic-fine-tuning control, but does
not claim a matched self-SFT/DPO control gap because there is no self-DPO stage.

## Held-out behavioral fidelity

Every SFT, DPO, and self-SFT adapter (1,104 total) generates responses to the same deterministic 200
held-out task prompts used for its dataset. Each response is compared with the target model's own
response to that prompt using two scorers:

- **Behavioral judge:** Qwen3-8B, greedy and non-thinking, outputs a 0–100 same-model similarity score
  based on answer, reasoning, format, verbosity, and style. A cell is complete only at 200/200 parsed
  scores. Unparseable formatting is retried with longer output budgets and never imputed.
- **Embedding:** cosine between normalized `sentence-transformers/all-MiniLM-L6-v2` response
  embeddings; deterministic and complete at 200/200.

For each off-diagonal source→target cell, the embedding imitation gain subtracts the unadapted
source→target response similarity on identical prompts. The 48 diagonal self-SFT cells provide an
empirical diagonal matched-compute reference. `experiments/imitation_safety/analyze_fidelity_campaign.py` enforces
the exact 528 SFT + 528 DPO + 48 self-SFT scope and writes the headline analysis. Final means are
0.739/0.720/0.841 (embedding) and 0.690/0.670/0.808 (judge) for SFT/DPO/self-SFT; the
scorers have Pearson r=0.940 across all 1,104 adapters.

**Safety-constrained imitation (mitigation scaffold).** `build-safety-sft-data` /
`build-safety-dpo-data` add `safety_replay` refusal rows; `launch-safety-sft` / `launch-safety-dpo`
train them (all support `--dry-run`; real launches need approval). Replay prompts default to
`data/datasets/benchmarks/refusal/advbench_harmful.csv`, excluding the refusal-ladder eval prompts;
each CSV has a sibling `.manifest.json`.

## Steering assay (benign provenance contrast vs refusal)

`dementor/steering/steering_rung.py`. Derive a per-layer source→target **diff-of-means over response
tokens** (both response sets teacher-forced through the SOURCE model, since source/target live in
different residual spaces). Forward-hook generation, two operators:

- `mode="ablate"` → `h ← h − β(h·v̂)v̂` (projection-ablation, the winning disguise operator).
- `mode="add"` → `h += α·v̂` (additive; over-steers into gibberish before it disguises — no clean
  regime).

The benign contrast uses 120 paired benign responses to derive a vector; 120 is a derivation count,
not an evaluation denominator. It is called the **benign cross-model activation contrast** below. The current
held-out response-embedding diagnostic does not support interpreting its ablation as literally
removing model identity: fingerprint ablation moves closer to the reference than random in only
7/27 models (mean distance change -0.0159 versus -0.0015). The mechanistic claim is therefore limited
to safety dissociation from this derived cross-model contrast. Machine-readable output is in
`data/results/safety/fingerprint_identity_diagnostic.json`.

**Safety dissociation protocol.** Projection-ablation at a mid layer, β ∈ {0.6, 1.0, 1.4},
RTL-judged, coherence-gated, paired McNemar. The final publication set contains 29 models × five harm
and two over-refusal benchmarks × two operators = 406 cells, all on the configured n=200 seed-42
sample. Imported legacy result JSON lacks a seed field, so the strict audit reconstructs its judged
prompt set and verifies exact configured-sample coverage rather than inferring provenance from the
denominator. **Pre-registered positive control:** ablate the
Arditi **refusal** direction (harmful−harmless last-token diff-of-means) — it must catastrophically
erode safety, or the base model is **EXCLUDED** as unpowered (the single-direction probe cannot resolve
the question there). Ablating the benign **fingerprint** direction under the identical operator is the
test. Geometry (cosines to refusal/persona) is reported but does **not** predict the causal effect, so
the claim rests on the causal control, not the cosines.

The consolidated base assay has complete seven-benchmark cells for 29 models under both operators.
Twenty-four pass the historical gate. The primary matched all-layer fingerprint/random variant is
complete for all 29; 23 pass its own gate and contribute 103 harm cells. Granite-4-h-small
has a valid cone and complete controls, but its positive control returns `PC_FAILS` on all five harm
benchmarks; it remains in the imitation campaign and outside the gated steering analysis.

Adapter-level steering is not manuscript evidence. Existing adapter-steering artifacts have
nonuniform arm coverage and are retained only as auxiliary provenance; the manuscript therefore does
not claim that activation geometry persists through fine-tuning.

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
