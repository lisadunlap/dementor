# Experiment and artifact inventory

This index distinguishes the current core-12 publication campaign from historical experiments. It
does not report in-progress headline values; those are valid only after the strict coverage gate.

## Current publication campaign

`config.yaml` defines the named `imitation_safety` cohort:

- 12 models and four imitation datasets;
- seed 42;
- 528 off-diagonal SFT cells and 528 matching DPO cells;
- seven safety benchmarks, using deterministic 200-row evaluation samples.

Training artifacts exist for every SFT and DPO cell. Safety-evaluation coverage is established only
from per-cell `metrics.json` checkpoints by `audit_erosion_coverage.py`; registry rows are not counted
as evaluated cells.

| Artifact | Meaning | Validity rule |
|---|---|---|
| `data/results/safety/erosion_coverage.json` | Exact checkpoint coverage and missing IDs | Authoritative during evaluation |
| `data/results/safety/erosion_seed42_long.csv` | One row per adapter × benchmark, with `stage` | Rebuild only after strict coverage |
| `data/results/safety/erosion_seed42_summary.csv` | One row per adapter, with `stage` | Rebuild only after strict coverage |
| `data/results/safety/erosion_variance_stats_sft.json` | SFT-only headline analysis | Requires 528/528 SFT |
| `data/results/safety/erosion_variance_stats_dpo.json` | DPO-only headline analysis | Requires 528/528 DPO |
| `data/results/safety/erosion_campaign_headlines.json` | Both stages and exact-cell SFT→DPO change | Requires both strict gates |
| `data/results/safety/figures_paper_{sft,dpo}/` | Stage-separated paper figures | Requires both strict gates |
| `docs/generated_campaign_results.tex` | Generated numeric paper macros | Requires both strict gates |

Use `experiments/imitation_safety/regenerate_campaign.py` for the guarded rebuild.

## Steering artifacts and scope

- `/data/ethantsliu/exp_steer_safety/repl80_rdo/`: base-model refusal-cone, benign provenance,
  and random-control evaluations.
- `/data/ethantsliu/exp_steer_adapter/`: controlled adapter subset plus cone-only 70B extension.

Llama-3.3-70B has complete controls at the base-model level. Its adapter extension has cone and
baseline arms, so it is not counted as evidence for a controlled adapter-level fingerprint null.
Granite remains in imitation but has no valid steering cone.

## Historical artifacts

The top-level `results/` tree contains the earlier four-model persistence ladder, refusal-rate safety
analysis, multi-seed pilots, durability analyses, and robustness checks. Historical 13×13/`n=769`
erosion tables and the wider steering roster remain useful provenance, but they are not current
core-12 headline artifacts.

The prompt-only ladder has fidelity measurements but no current RTL seven-benchmark safety campaign.
That is intentionally outside the present weight-based SFT/DPO claim scope, not a completion blocker.
