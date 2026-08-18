# Dementor

## Experiment dashboard

These are the audited publication experiments and current claim boundaries. Headline values are
generated from the committed artifacts in
[`data/results/safety/`](data/results/safety/) and synchronized with
[`docs/generated_campaign_results.tex`](docs/generated_campaign_results.tex) and
[`paper/naz_aaai2027/generated_steering_results.tex`](paper/naz_aaai2027/generated_steering_results.tex).
Positive change means greater harmful compliance than the same unadapted source; negative change
must not be read as greater overall safety without the separate over-refusal result.

### Campaign coverage and paper use

| Experiment | Population | Evaluation | Audited status | Paper use |
|---|---:|---|---:|---|
| SFT imitation safety | 12 models × 11 targets × 4 corpora | 7 safety benchmarks, n=200, seed 42 | **528/528 complete** | Claim 1 |
| SFT→DPO imitation safety | Same exact cells and SFT parents | 7 safety benchmarks, n=200, seed 42 | **528/528 complete** | Claim 1 |
| Self-SFT controls | 12 models × 4 corpora | 7 safety benchmarks, n=200, seed 42 | **48 adapters / 336 scores complete** | Generic fine-tuning control |
| Behavioral fidelity | 528 SFT + 528 DPO + 48 self-SFT | 2 scorers × 200 held-out prompts | **1,104/1,104 per scorer** | Claim 1 validation |
| Exact SFT→DPO pairing | Dataset/source/target/seed matched | 528 paired cells | **528/528 paired** | Incremental DPO analysis |
| Base steering, historical single-layer controls | 29 base models | 7 benchmarks, n=200, seed 42 | **203/203 cells; 24 gated models** | Secondary Claim 2 coverage |
| Base steering, matched all-layer (`fpall`) controls | 29 base models | Same 7 benchmark cells and sampler | **203/203 cells; 23 gated models, 103 harm cells** | Primary Claim 2 assay |
| Adapter steering | Nonuniform archived subset | Controls applied after fine-tuning | Incomplete by design | **Excluded from paper evidence** |
| Prompt-only safety rungs | Not launched | Would add a new intervention family | Outside current campaign | **Excluded from paper evidence** |

### Imitation-safety headline results

| Stage | Cells | Mean harm change (pp) | Source-cluster 95% CI | Median (pp) | Cells ≥+5 pp | Maximum (pp) | Directional share | Mean over-refusal change (pp) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| SFT | 528 | +0.02 | [-0.89, +0.90] | +0.10 | 7.4% | +34.30 | 48.1% lower than base | +1.87 |
| SFT→DPO | 528 | +0.77 | [+0.23, +1.45] | +0.20 | 4.7% | +15.10 | 33.7% lower than base | -1.33 |
| DPO − matched SFT | 528 pairs | +0.76 | [-0.32, +2.13] | +0.30 | — | — | 56.8% DPO higher | — |
| Self-SFT | 48 | +0.86 | — | — | — | — | diagonal control | — |

### Behavioral-fidelity results

All 1,104 adapters are evaluated against their target's own responses on 200 held-out prompts with
both a Qwen3-8B behavioral judge and deterministic MiniLM embedding cosine. Mean embedding fidelity
is 0.739 for SFT, 0.720 for DPO, and 0.841 for self-SFT. Relative to the matched unadapted
source→target baseline (0.705), SFT gains +0.033 on average in 89.8% of cells and DPO gains +0.015 in
75.2%. Judge fidelity is 0.690, 0.670, and 0.808, respectively; its exact-cell DPO−SFT
change is −0.020. The two fidelity scorers correlate at Pearson r=0.940. Fidelity and
source-relative harmful-compliance change have small descriptive within-source-and-dataset
associations; cellwise Pearson p-values are not treated as independent-cell inference.

### Target-relative safety transfer

No new model runs are needed for this analysis: every target is also a base source, so all 60
model×harm-benchmark target baselines exist on the exact seed-42 sample. Let `T = target harm −
source harm` and `A = adapter harm − source harm`, after averaging the five harm benchmarks per
cell. The origin-constrained slope `sum(T*A)/sum(T^2)` is 0 when adapters stay at the source and 1
when they reach the target. SFT's point estimate is +0.156 (crossed source–target–dataset bootstrap
95% CI [-0.093, +0.493]); DPO's is −0.023 ([-0.143, +0.208]). SFT reduces mean target distance by
+0.95 pp ([-0.73, +2.88]) and is closer in 62.1% of nonzero-gap cells; DPO's mean reduction is
+0.00 pp ([-0.85, +0.74]). The point estimates suggest partial directional movement after SFT, but
the dependence-aware interval includes no transfer, and DPO does not preserve the pattern.

### Descriptive variance shares

These are separate marginal one-way shares, not a joint causal variance decomposition.

| Factor | SFT η² | SFT→DPO η² |
|---|---:|---:|
| Source model | 12.2% | 30.4% |
| Target-output model | 12.6% | 6.0% |
| Training corpus | 4.4% | 2.1% |

### Base-steering results

| Assay | Model-level units | Contributing cells | Refusal cone (pp) | Cross-model contrast (pp) | Random control (pp) | Contrast vs random |
|---|---:|---:|---:|---:|---:|---|
| Matched all-layer (`fpall`), primary | 23 | 103 | +41.73 | +1.59 | +0.75 | W=89, exact p=.142 |
| Historical single-layer controls, secondary | 24 | — | +41.85 | +0.62 | +0.48 | W=124, exact p=.687 |

The tests above fail to detect a contrast–random location difference; they are not equivalence
tests. Granite-4-H-Small has complete base and `fpall` evaluations but is excluded from the gated
aggregate because its refusal-cone positive control fails on all five harm benchmarks.

### Automated-judge agreement study

This is a restricted, unbalanced 4,712-response historical convenience sample, not human ground
truth and not a second judge over every matrix cell.

| Judge pair | Cohen's κ | 95% CI | Both | A only | B only | Neither |
|---|---:|---:|---:|---:|---:|---:|
| Llama-Guard vs HarmBench | 0.243 | [0.213, 0.272] | 283 | 913 | 157 | 3,359 |
| Llama-Guard vs RTL | 0.249 | [0.219, 0.279] | 256 | 940 | 84 | 3,432 |
| HarmBench vs RTL | 0.699 | [0.661, 0.735] | 282 | 158 | 58 | 4,214 |

Llama-Guard flags 25.4% of this sample, HarmBench 9.3%, and RTL 7.2%. The publication matrix uses
the same binary RTL harmful-compliance verdict for all five harm benchmarks; native graders remain
auxiliary diagnostics.

### Sample-count glossary

| Count | Meaning | Current status |
|---:|---|---|
| 200 | Evaluation prompts per benchmark, seed 42 | Active standard |
| 120 | Paired benign examples used to derive a cross-model activation contrast | Derivation only; never a result denominator |
| 300 | Historical steering evaluation cap | Archived; current 406-cell steering publication set is n=200 |
| 150 | Historical imitation-safety cap | Retired and excluded from the core-12 aggregate |

### Result and artifact index

| Artifact | Contents |
|---|---|
| [`erosion_coverage.json`](data/results/safety/erosion_coverage.json) | Strict 528/528 SFT and DPO coverage audit |
| [`erosion_campaign_headlines.json`](data/results/safety/erosion_campaign_headlines.json) | Headline imitation and exact paired-stage statistics |
| [`target_safety_transfer.json`](data/results/safety/target_safety_transfer.json) | Direct source→target safety-gap projection with crossed-factor uncertainty |
| [`erosion_seed42_summary.csv`](data/results/safety/erosion_seed42_summary.csv) | Stage-labelled cell-level summary |
| [`self_sft_headlines.json`](data/results/safety/self_sft_headlines.json) | Complete 48-control self-SFT safety aggregate |
| [`fidelity_all_embed_long.csv`](data/results/fidelity/fidelity_all_embed_long.csv) | Exact 1,104-cell embedding-fidelity table |
| [`fidelity_all_judge_long.csv`](data/results/fidelity/fidelity_all_judge_long.csv) | Exact 1,104-cell behavioral-judge table |
| [`fidelity_campaign_headlines.json`](data/results/fidelity/fidelity_campaign_headlines.json) | Fidelity baselines, stage contrasts, and safety association |
| [`base_steering_coverage.json`](data/results/safety/base_steering_coverage.json) | Per-model base/`fpall` coverage, verdicts, and sample provenance |
| [`fpall_comparison_stats.json`](paper/naz_aaai2027/img/fpall_comparison_stats.json) | Primary matched-operator steering statistics and cohort |
| [`fingerprint_identity_diagnostic.json`](data/results/safety/fingerprint_identity_diagnostic.json) | Held-out diagnostic limiting identity-language claims |
| [`CAMPAIGN_CONSOLIDATION_20260810.md`](docs/CAMPAIGN_CONSOLIDATION_20260810.md) | Cross-machine and Hugging Face hashes, imports, and final scope |

Dementor measures whether behavioral fingerprints persist when one language model is made to imitate
another. The current publication experiment addresses three bounded questions:

1. Does target-output SFT or SFT→DPO measurably reproduce held-out target behavior?
2. How does that training change harmful compliance relative to the
   unadapted source?
3. In base models, does matched ablation detect a difference between a benign cross-model activation contrast
   and a random direction, when a refusal-cone positive control fires?

The paper establishes an aggregate held-out imitation signal, not uniform success or complete target
identity transfer, and does not use the nonuniform adapter-steering archive as evidence.

The current safety result is generated from the complete, stage-separated core-12 campaign. Older
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

Both stages are complete at 528/528. To reproduce the committed aggregates and paper macros:

```bash
python experiments/imitation_safety/regenerate_campaign.py
```

The command refuses partial coverage. Its aggregated CSVs contain an explicit `stage` column. DPO headline statistics never pool SFT rows;
the incremental SFT→DPO effect is paired on exact dataset/source/target/seed cells.

## Steering scope

Base-model steering uses projection ablation with three controls:

- refusal cone: positive control that must move harm;
- benign cross-model activation contrast: test direction;
- norm-matched random direction: null control.

The benign contrast is derived from 120 paired benign responses. That is a direction-derivation count,
not an evaluation sample size. Its held-out diagnostic does not justify calling it a causally
validated “identity direction,” so the paper calls it a **benign cross-model activation contrast**
and limits the claim to dissociation from refusal behavior.

The consolidated base tree contains complete five-harm- plus two-over-refusal-benchmark evaluations
for 29 models under both operators (406 cells). Twenty-four pass the historical positive-control
gate; 23 pass the primary matched all-layer gate and contribute 103 harm cells. Granite has a valid RTX-run
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
