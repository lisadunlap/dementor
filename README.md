# Dementor

Dementor measures whether behavioral fingerprints persist when one language model is made to imitate
another, and whether weight-based imitation changes safety. The publication experiment separates two
questions:

1. Does benign SFT or SFT→DPO move a source model toward a target model's behavior?
2. Does the same intervention change genuine harmfulness relative to the unadapted source?

The current safety result is being rebuilt from the complete, stage-separated core-12 campaign. Older
`n=769`/13×13 headline numbers are historical and must not be cited as the final core-12 result.

## Publication campaign

[`config.yaml`](config.yaml) is the single source of truth. The named `imitation_safety` campaign is:

- 12 models: Aya, Gemma E4B/31B, Granite-4-H-Small, Llama-3.1-8B, Llama-3.3-70B, Ministral-8B,
  OLMo-3-7B, Phi-4, Qwen3.6-27B, GPT-OSS-20B, and Nemotron-Nano-30B-A3B.
- Four training datasets: Chatbot Arena, GSM8K, OASST1, and WritingPrompts.
- Seed 42.
- 12 × 11 × 4 = **528 off-diagonal cells per stage**, separately for SFT and DPO.
- Seven safety metrics on one deterministic 200-row seed-42 evaluation sample: five harm benchmarks
  plus XSTest and OR-Bench-Hard over-refusal controls.

The wider model catalog and seeds 43/44 remain resolvable for historical and robustness analyses, but
the default matrix planner dispatches only the named publication campaign.

```bash
dementor-plan
# Roster: 12 models; SFT 528, DPO 528, self-SFT 48
```

## Completion and recomputation

Training completion and safety-evaluation completion are different. A registered adapter is not a
completed safety cell. Audit the actual checkpoints before reporting coverage:

```bash
python experiments/imitation_safety/audit_erosion_coverage.py \
  --work-root data/imitation_safety/work \
  --output data/results/safety/erosion_coverage.json \
  --strict
```

The audit requires all seven benchmark metrics, a source baseline, and matched evaluation sample
sizes. It emits exact missing and incomplete cell ids. Multiple `--work-root` arguments logically
merge results from different boxes, deduplicate agreeing checkpoints, prefer harmonized n=200 over
legacy n=300 checkpoints, and fail on unresolved conflicts.

Once both stages reach 528/528:

```bash
python experiments/imitation_safety/regenerate_campaign.py
```

The command refuses partial coverage. Its aggregated CSVs contain an explicit `stage` column. DPO headline statistics never pool SFT rows;
the incremental SFT→DPO effect is paired on exact dataset/source/target/seed cells.

## Steering scope

Base-model steering uses projection ablation with three controls:

- refusal cone: positive control that must move harm;
- benign cross-model fingerprint/provenance contrast: test direction;
- norm-matched random direction: null control.

The benign contrast is derived from 120 paired benign responses. That is a direction-derivation count,
not an evaluation sample size. Its held-out diagnostic does not justify calling it a causally validated
“identity direction,” so the paper uses **benign provenance contrast** or **imitation-aligned
fingerprint direction** and limits the claim to dissociation from refusal behavior.

The consolidated base tree contains complete five-harm-benchmark evaluations for 29 models; 24 pass
the positive-control gate. The stricter matched all-layer fingerprint/random robustness variant is
complete for 22 of the 29 models (19 gated), including Llama-3.3-70B. Granite now has a valid RTX-run
cone and complete controls, but its positive control fails on all five harm benchmarks, so it remains
in the imitation matrix and outside the gated steering analysis.

Adapter steering is retained as auxiliary provenance but excluded from the paper's evidence because
its model/arm coverage is nonuniform. In particular, no adapter-geometry claim is made at 70B, so
additional 70B adapter fingerprint/random controls are not required.

## Package layout

- `dementor/` — importable library: configuration, disguise methods, metrics, training backends,
  serving, steering, and safety utilities.
- `experiments/` — analysis, figures, campaign evaluators, and reproducibility tools.
- `tests/` — CPU-focused pytest suite.
- `METHODS.md` — canonical metric and evaluation definitions.
- `docs/RESULTS.md` — current result status and claim boundaries.

## Installation

Most datasets and result artifacts use Git LFS. Materialize them before testing or analysis:

```bash
git lfs install
git lfs pull
pip install -e .
```

Optional dependencies:

- `.[train]` — PEFT/TRL/Accelerate/Datasets/Tinker training paths.
- `.[serve]` — vLLM and Gradio.
- `.[dev]` — pytest.

Remote and GPU dependencies stay lazily imported so CPU analysis remains usable without the training
stack.

## Common commands

```bash
# Generate base responses
dementor-generate \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/full/openai_gpt-4.1-mini.csv \
  basic --model openai/gpt-4.1-mini

# Apply a disguise method
dementor-disguise \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --disguise-as openai/gpt-4.1-mini \
  --method contrastive --num-samples 50

# Score a comparison
python -m dementor.scorer pairwise --input comparison.csv --output scored.csv

# Dry-run the matrix (never launches training)
dementor-plan
python -m dementor.training.matrix list-cells
```

Actual Tinker or local-GPU training is cost-gated and must not be launched without explicit approval.

## Testing

```bash
pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
```

The local-backend tests may download a tiny random model unless it is already cached.
