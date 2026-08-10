# Results index

| Location | Experiment | Status |
|---|---|---|
| `data/results/safety/coverage_core12*.json` | Exact SFT/DPO checkpoint coverage | **Authoritative during evaluation** |
| `data/results/safety/erosion_seed42_{summary,long}.csv` | Stage-labelled core-12 erosion tables | Regenerate after strict 528/528 gates |
| `data/results/safety/erosion_variance_stats{,_sft}.json` | DPO and SFT headline statistics | Regenerate after strict gates |
| `data/results/safety/figures*/` | Stage-separated erosion figures | Regenerate after strict gates |
| `results/` | Earlier persistence/disguise ladder | Legacy instrument context |
| `/data/ethantsliu/exp_steer_safety/repl80_rdo/` | Base steering artifacts | Current, off-repo |
| `/data/ethantsliu/exp_steer_adapter/` | Adapter-steering artifacts | Current, off-repo |

The canonical framing and claim limits are in [`RESULTS.md`](RESULTS.md). Historical `n=769` and
13×13 artifacts are not final core-12 results. Never infer coverage from registry rows alone; use the
strict coverage audit over actual `metrics.json` checkpoints.
