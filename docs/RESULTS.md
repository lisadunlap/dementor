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

The diagonal control contains 48 self-SFT adapters and 336 safety scores. Behavioral fidelity is
separately complete for all 1,104 adapters (528 SFT, 528 DPO, and 48 self-SFT), with exactly 200
parsed prompt comparisons under both the behavioral judge and embedding scorer.

## Stage-separated analysis

`build_erosion_csv.py` writes an explicit `stage` column. All distributions, variance decompositions,
and figures select exactly one stage. The incremental preference-optimization effect is

`erosion(DPO) - erosion(SFT)`

paired on the exact dataset, source, target, and seed cell. Pooling SFT and DPO rows is invalid.
Positive erosion means the adapted source has higher harmful compliance than its own unadapted
baseline; negative erosion means lower harmful compliance on the RTL metric, not necessarily greater
overall safety. The over-refusal change is reported separately.

## Behavioral-fidelity analysis

Each adapter is compared with its target model's own responses on 200 held-out prompts from the
training corpus. The Qwen3-8B scorer judges same-model behavioral similarity; MiniLM embedding cosine
is a deterministic secondary measure. SFT embedding fidelity averages 0.739, DPO 0.720, and self-SFT
0.841. Against a matched unadapted source→target baseline of 0.705, the mean gains are +0.033 for SFT
and +0.015 for DPO. Judge fidelity is 0.690/0.670/0.808 for SFT/DPO/self-SFT. Exact-cell
DPO−SFT changes are −0.018 for embedding and −0.020 for the judge, and the two scorers correlate
at Pearson r=0.940. The campaign therefore demonstrates measurable imitation without a positive
within-source-and-dataset coupling between fidelity and harmful-compliance change.

## Steering evidence

The base-model assay applies projection ablation to:

1. a refusal cone, which is the positive control;
2. a benign cross-model activation contrast;
3. a norm-matched random direction.

The original campaign applies the two single-direction controls at the declared layer while the cone
acts across layers. The `fpall` robustness subset applies the single directions across all layers as
well, removing that operator-size asymmetry; it is complete for all 29 models.

The fingerprint vector is derived from 120 paired benign responses. The held-out identity diagnostic
does not show that its ablation literally removes model identity, so the mechanistic wording is
limited to dissociation between the benign provenance contrast and refusal behavior.

The publication steering design is 29 models × seven benchmarks × two operators = 406 evaluations,
all with the configured n=200 seed-42 prompt set. Imported legacy result JSON does not encode
a seed, so the strict audit reconstructs judged prompts and verifies exact sample coverage instead
of inferring provenance from the denominator. Among the
145 harm cells per operator, 58 base and 60 fpall cells are explicit harmonized rescores; the
remainder are native n=200 files.

The consolidated base tree contains complete seven-benchmark evaluations for 29 models under both
operators. Twenty-four pass the historical positive-control gate; 23 pass the fpall gate and
contribute 103 harm cells to the primary aggregate. Granite-4-h-small now has a valid, independently
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
