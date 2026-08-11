# Experiment and artifact inventory

This index distinguishes the completed core-12 publication campaign from historical experiments.
All current headline artifacts below were regenerated after the strict coverage gate passed.

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
| `data/results/safety/erosion_coverage.json` | Exact checkpoint coverage and missing IDs | Complete: 528/528 each stage |
| `data/results/safety/erosion_seed42_long.csv` | One row per adapter × benchmark, with `stage` | Current guarded rebuild |
| `data/results/safety/erosion_seed42_summary.csv` | One row per adapter, with `stage` | Current guarded rebuild |
| `data/results/safety/erosion_variance_stats_sft.json` | SFT-only headline analysis | Requires 528/528 SFT |
| `data/results/safety/erosion_variance_stats_dpo.json` | DPO-only headline analysis | Requires 528/528 DPO |
| `data/results/safety/erosion_campaign_headlines.json` | Both stages and exact-cell SFT→DPO change | Requires both strict gates |
| `data/results/safety/figures_paper_{sft,dpo}/` | Stage-separated paper figures | Requires both strict gates |
| `docs/generated_campaign_results.tex` | Generated numeric paper macros | Requires both strict gates |
| `data/results/safety/base_steering_coverage.json` | Exact base/fpall steering coverage and verdicts | Rebuild after artifact import |

Use `experiments/imitation_safety/regenerate_campaign.py` for the guarded rebuild.

## Steering artifacts and scope

- `/data/ethantsliu/exp_steer_safety/repl80_rdo/`: base-model refusal-cone, benign provenance,
  and random-control evaluations.
- `/data/ethantsliu/exp_steer_adapter/`: nonuniform adapter-steering archive, excluded from manuscript evidence.

The consolidated base tree has 29/29 evaluated models complete on five harm benchmarks. Twenty-four
pass the positive-control gate. Matched all-layer (`fpall`) controls are complete for 22/29 evaluated
models and 19/24 gated models. Llama-3.3-70B is complete in both base variants. Granite's verified RTX
package is also complete in both variants; all five base harm verdicts are `PC_FAILS`, so it does not
enter the gated analysis.

`metrics_n200.json` is preferred only when it records 200 prompts. Several legacy steering
generations overlap only part of the selected set; those intersection files are ignored and the
native `metrics.json` denominator remains authoritative. Older harmonized files do not encode a
seed or prompt hash, so the generated coverage manifest records them as `harmonized_n200` without
inventing seed provenance. It records
the prompt count and metric source for every base/fpall benchmark cell, plus aggregate sampling
counts; it does not apply a false global n=200 label.

## Historical artifacts

The top-level `results/` tree contains the earlier four-model persistence ladder, refusal-rate safety
analysis, multi-seed pilots, durability analyses, and robustness checks. Historical 13×13/`n=769`
erosion tables and the wider steering roster remain useful provenance, but they are not current
core-12 headline artifacts.

The prompt-only ladder has fidelity measurements but no current RTL seven-benchmark safety campaign.
That is intentionally outside the present weight-based SFT/DPO claim scope, not a completion blocker.
