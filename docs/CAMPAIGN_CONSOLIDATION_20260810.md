# Campaign consolidation — 2026-08-10

> **Completion postscript (2026-08-17):** the later full campaign supersedes the coverage snapshot
> below. Current totals are 528 SFT + 528 DPO + 48 self-SFT adapters; two 200-prompt fidelity scorers
> cover all 1,104; self-SFT has 336 safety scores; and steering has 29 models × seven benchmarks ×
> two operators = 406 evaluations. Both steering operators now have all 145 harm cells at n=200;
> 23 models and 103 harm cells enter the primary fpall aggregate. Historical counts in this dated
> handoff are retained only as provenance for what had been available on August 10.

This is the handoff record for the core-12 imitation-safety campaign and base-steering assay. No new
GPU campaign is required under the manuscript scope below.

## Publication scope and completion

- Claim 1: benign target-output fine-tuning changes source-relative harmful compliance little on
  average, with material cellwise tails. The exact configured matrix is complete: 528/528 SFT cells
  and 528/528 matching DPO cells, four training datasets, seven safety benchmarks, evaluation cap
  200, seed 42. The paper does not claim uniform held-out target-behavior transfer.
- Claim 2: in base models, the matched assay detects a large refusal-cone effect but no statistically
  detectable benign-cross-model-contrast--random difference. The canonical tree has 29 models complete on the
  five harm benchmarks; 24 pass the positive-control gate. Failure to detect a difference is not an
  equivalence result.
- Matched all-layer fingerprint/random controls are a robustness subset: 22/29 evaluated models and
  19/24 gated models have complete five-benchmark coverage. Llama-3.3-70B is complete in both base
  variants.
- Adapter steering is excluded from manuscript evidence because its model and arm coverage is
  nonuniform. The paper does not claim that activation geometry persists after fine-tuning, so
  Llama-3.3-70B adapter fingerprint/random controls are not required.
- Prompt-only safety rungs and a second fingerprint reference would broaden the claim rather than
  complete either claim above; they are outside the current campaign.

The generated sources of truth are:

- `data/results/safety/erosion_coverage.json`
- `data/results/safety/erosion_campaign_headlines.json`
- `data/results/safety/base_steering_coverage.json`

## Code integration

The `ethan` history contains both divergent campaign lines: Box A commit `a5574f9` and Box B commit
`864e937`, descended from common base `e7e2893`, were reconciled at `90de930`. The temporary Box A
and campaign-integration refs were deleted after consolidation; `ethan` is the continuing campaign
branch and `main` remains unchanged.

The consolidated history also contains the configuration-defined core-12 cohort, strict coverage gate, stage-aware
aggregation, exact-cell SFT→DPO comparison, data-derived eroder classification, documentation cleanup,
and the steering harmonization/loader corrections from this consolidation. The dirty `ethan`
worktree's apparent tracked `data/` deletions were never staged or committed.

## Hugging Face artifact record

Steering directions are in
[`dementor-research/dementor-steering-directions`](https://huggingface.co/datasets/dementor-research/dementor-steering-directions)
at revision `d8d34f3f72d2135afd73b25aef68dcc63d5193a4`:

- `granite-4-h-small/selected_cone.pt`: SHA-256
  `5bc06d5dd6da2682203d4bebecedf6a5b9d07f6ef83bd39cc60d5ed982ef812d`
- `granite-4-h-small/vectors_ml.pt`: SHA-256
  `41b706f5f00292c302b3cb8ccf20deee8bffa56223781320a8440c86ad22de79`
- `granite-4-h-small/selection.json`

Worker backups are in
[`dementor-research/dementor-adapter-registry`](https://huggingface.co/datasets/dementor-research/dementor-adapter-registry)
at revision `74ab9c4de8d328e91ec316d68de61b9916141cb0`:

- RTX Granite archive: `worker_backups/rtx_20260810/granite_base_steering.tgz`, SHA-256
  `3a377426f616cf0e561b7763a6081d0f35afd9be5d862ca8926ef6c0ee21eff2`
- RTX manifest: `worker_backups/rtx_20260810/granite_base_steering_manifest.json`
- H100 metrics: `h100_metrics_20260809.tgz`, SHA-256
  `7f4fbed7ddc52f567a1049df5e3617234b6ef4fa31f2358f03225bd656d6fd75`
- H100 cone-pass judged: `h100_conepass_judged_20260809.tgz`, SHA-256
  `91203e64eb9416021de15b7777a98254726a604f56160dde24ff158244be9bb3`
- H100 all judged: `h100_all_judged_20260809.tgz`, SHA-256
  `9eef4e085d7d2e6a71e0c1cdb9c22aecb384ca53fb27c2248e06ae3c381de710`

All 192 files in the RTX manifest were verified by size and SHA-256 before import. The previous failed
Granite directory is preserved at
`/data/ethantsliu/exp_steer_safety/repl80_rdo_archives/granite-4-h-small.pre_rtx_20260810`.
The verified package is installed at
`/data/ethantsliu/exp_steer_safety/repl80_rdo/granite-4-h-small` and returns `PC_FAILS` on all five
harm benchmarks. Five previously absent Gemma-4-31B fpall directories were imported from the H100
backup without overwriting existing base cells.

## Sampling caveat fixed during consolidation

The core-12 imitation matrix uses the exact seed-42 200-row samples throughout. New steering cells
also use that standard, but some legacy steering generations used 300 prompts (or 100 for SG-Bench)
and do not contain all prompts in the newer selection. The old harmonizer incorrectly wrote their
partial intersections as `metrics_n200.json`.

The corrected loader accepts a harmonized file only when `n_prompts == 200`; otherwise it falls back
to the native `metrics.json`. The harmonizer now writes only when the complete selected prompt set is
present. Its audit finds 553 complete 200-row rescores, 202 incomplete legacy overlaps, 30 cells without a
standard subsample, and eight unreadable/empty cells across the full base-plus-adapter archive. This
requires a denominator disclosure for legacy steering, not new GPU work.

The base-only coverage manifest now derives native denominators from baseline generation rows. Its
145 canonical harm cells comprise 58 complete harmonized n=200 rescores, 74 native n=300 cells, 12 native
n=100 cells, and one native n=200 cell without recoverable seed metadata. The 113 fpall cells comprise
55 complete harmonized n=200, 51 native n=300, and seven native n=100 cells. Older harmonized JSON
does not encode a seed or prompt hash, so the manifest does not infer either from the denominator.
Per-cell provenance is stored
under `models.<slug>.{base,fpall}_cells` in `base_steering_coverage.json`.

## Rebuild commands

```bash
python experiments/imitation_safety/audit_erosion_coverage.py --strict
python experiments/imitation_safety/regenerate_campaign.py
python experiments/steering/harmonize_to_200.py --dry-run
python experiments/steering/audit_base_steering.py
python experiments/figures/rebuild_steering_figures.py --outdir paper/naz_aaai2027/img
python experiments/figures/compare_fpall.py --min-coverage 5 \
  --json-out paper/naz_aaai2027/img/fpall_comparison_stats.json \
  --tex-out paper/naz_aaai2027/generated_steering_results.tex
python experiments/steering/test_fingerprint_is_identity.py \
  --json-out data/results/safety/fingerprint_identity_diagnostic.json
pytest -q
```
