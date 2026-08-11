# Results status and claim boundaries

This document describes the current core-12 publication analysis. Historical `n=769`, 13×13, and
pre-harmonization steering summaries remain available in Git history but are not current headline
results.

## Authoritative completion gate

The `imitation_safety` campaign in `config.yaml` defines 528 off-diagonal cells for each of two
weight-based imitation stages:

| Stage | Meaning | Required cells |
|---|---|---:|
| SFT | source model fine-tuned on target responses | 528 |
| DPO | matching SFT parent followed by target-over-source preference optimization | 528 |

A stage is complete only when
`experiments/imitation_safety/audit_erosion_coverage.py --strict` reports 528/528. Adapter registry
coverage is training evidence, not safety-evaluation evidence.

Every complete cell must contain five harm metrics (AdvBench, HarmBench, StrongREJECT, SORRY-Bench,
SG-Bench), two over-refusal metrics (XSTest, OR-Bench-Hard), and the matching unadapted source
baseline. All use the deterministic seed-42 200-row sample. XSTest's scored denominator is 111 benign
rows within that sample.

The committed coverage manifest at `data/results/safety/erosion_coverage.json` records 528/528 SFT
and 528/528 DPO cells. The committed headline tables, figures, and macros were regenerated only
after both stages passed the strict gate.

## Stage-separated analysis

`build_erosion_csv.py` writes an explicit `stage` column. All distributions, variance decompositions,
and figures select exactly one stage. The incremental preference-optimization effect is

`erosion(DPO) - erosion(SFT)`

paired on the exact dataset, source, target, and seed cell. Pooling SFT and DPO rows is invalid.
Positive erosion means the adapted source has higher harmful compliance than its own unadapted
baseline; negative erosion means lower harmful compliance on the RTL metric, not necessarily greater
overall safety. The over-refusal change is reported separately.

## Steering evidence

The base-model assay applies projection ablation to:

1. a refusal cone, which is the positive control;
2. a benign cross-model fingerprint/provenance contrast;
3. a norm-matched random direction.

The original campaign applies the two single-direction controls at the declared layer while the cone
acts across layers. The `fpall` robustness subset applies the single directions across all layers as
well, removing that operator-size asymmetry; its smaller 22-model coverage is reported explicitly.

The fingerprint vector is derived from 120 paired benign responses. The held-out identity diagnostic
does not show that its ablation literally removes model identity, so the mechanistic wording is
limited to dissociation between the benign provenance contrast and refusal behavior.

The 200/seed-42 evaluation standard is exact for the imitation matrix and newly generated steering cells. Some
legacy steering generations used 300 prompts (or 100 for SG-Bench) and do not contain the complete
new seed-42 set. Their cached intersections are not mislabeled as n=200: the figure loader accepts a
`metrics_n200.json` file only when it records all 200 prompts. Otherwise it uses the native metric
and records the denominator derived from the baseline rows in `all_gens.csv` in
`base_steering_coverage.json`.

The generated manifest makes that mixture explicit. Among 145 canonical base harm cells, 58 are
complete harmonized n=200 rescores, 74 retain native n=300, 12 retain native n=100, and one retains native
n=200 without a recoverable seed field. Among 113 fpall cells, the corresponding counts are 55, 51,
and 7. Older harmonized files do not encode a seed or prompt hash, so they are not labeled seed 42.
Aggregate steering plots therefore disclose mixed per-cell denominators rather than claiming that
every legacy generation used the current sampler.

The consolidated base tree contains complete five-harm-benchmark evaluations for 29 models; 24 pass
the positive-control gate. The matched all-layer fingerprint/random robustness variant is complete
for 22/29 evaluated models and 19/24 gated models. Granite-4-h-small now has a valid, independently
verified RTX cone and complete base/fpall cells, but its positive control returns `PC_FAILS` on every
harm benchmark. It remains in the 12-model imitation matrix and outside the gated steering analysis.

Llama-3.3-70B has complete base-model cone, fingerprint, and random controls across the five harm
benchmarks.

## Adapter-steering scope

Adapter steering is excluded from the manuscript evidence. Its archived cells have nonuniform model
and arm coverage, so they are neither pooled with the complete imitation matrix nor used to claim
that activation geometry persists through fine-tuning. Llama-3.3-70B adapter fingerprint/random
controls are therefore unnecessary under the current claim scope: the paper makes a base-model
mechanistic claim and a separate behavioral fine-tuning claim, not an adapter-geometry claim.

## Work deliberately outside scope

- Prompt-only safety rungs are not part of the weight-based SFT/DPO safety claim.
- No additional Granite cone retries are planned.
- No second-reference fingerprint campaign is required for the current provenance-dissociation claim.
- No 70B adapter controls are required unless the manuscript is expanded to make an explicit
  controlled adapter-level claim at 70B.

## Reproduction

```bash
python experiments/imitation_safety/regenerate_campaign.py
```

The regeneration command runs the strict coverage gate first and refuses to overwrite headline
artifacts from a partial matrix. It writes identical generated-result macros beside the documentation
draft and the AAAI paper entry point. Pass repeated `--work-root` arguments when results remain on
multiple machines.

The active submission source lives in the sibling Overleaf repository at
`../dementor-overleaf/AnonymousSubmission2027.tex`. This repository retains a synchronized audit
snapshot in `docs/AAAI_DRAFT.tex`; `paper/naz_aaai2027/aaai2027_identity_safety_main.tex` is a local
wrapper for compiling that snapshot against the committed code artifacts. Regeneration updates the
snapshot macros, but the submission repository remains the publication entry point.

See `METHODS.md` for the canonical metric definitions and `docs/RESULTS_INDEX.md` for artifact paths.
