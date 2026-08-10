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

The generated coverage manifest at `data/results/safety/erosion_coverage.json` is the source of
truth while evaluation is in progress. Headline numbers are regenerated only after both stages pass
the strict gate.

## Stage-separated analysis

`build_erosion_csv.py` writes an explicit `stage` column. All distributions, variance decompositions,
and figures select exactly one stage. The incremental preference-optimization effect is

`erosion(DPO) - erosion(SFT)`

paired on the exact dataset, source, target, and seed cell. Pooling SFT and DPO rows is invalid.
Positive erosion means the adapted source is more harmful than its own unadapted baseline; negative
erosion means it became safer.

## Steering evidence

The base-model assay applies the same projection-ablation operator to:

1. a refusal cone, which is the positive control;
2. a benign cross-model fingerprint/provenance contrast;
3. a norm-matched random direction.

The fingerprint vector is derived from 120 paired benign responses. The held-out identity diagnostic
does not show that its ablation literally removes model identity, so the mechanistic wording is
limited to dissociation between the benign provenance contrast and refusal behavior.

Granite-4-h-small has no valid cone: the final target filters retained zero examples and optimization
became NaN. Granite remains in the 12-model imitation matrix but is excluded from steering. The
matched cross-method overlap is therefore 11 models; imitation membership is not required to imply a
successful steering assay.

Llama-3.3-70B has complete base-model cone, fingerprint, and random controls across the five harm
benchmarks.

## Adapter-steering scope

The controlled adapter analysis consists only of cells with cone, fingerprint, and random arms. The
70B adapter extension contains cone and baseline arms, so it supports this limited statement:

> The refusal-cone positive control remains effective after imitation fine-tuning at 70B.

It does not support a controlled adapter-level fingerprint-null claim at 70B. The manuscript must not
use the cone-only 70B cells to extend the controlled adapter dissociation range.

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

The canonical submission entry point is
`paper/naz_aaai2027/aaai2027_identity_safety_main.tex`; it imports the single manuscript body from
`docs/AAAI_DRAFT.tex`, preventing an older paper copy from retaining stale roster sizes or headline
numbers. Compile it from `paper/naz_aaai2027/` after regeneration.

See `METHODS.md` for the canonical metric definitions and `docs/RESULTS_INDEX.md` for artifact paths.
