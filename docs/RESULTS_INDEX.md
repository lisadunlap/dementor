# Results index

| Location | Experiment | Status |
|---|---|---|
| `data/results/safety/erosion_coverage.json` | Exact SFT/DPO checkpoint coverage | **Complete: 528/528 each** |
| `data/results/safety/erosion_seed42_{summary,long}.csv` | Stage-labelled core-12 erosion tables | Current guarded rebuild |
| `data/results/safety/erosion_variance_stats_{sft,dpo}.json` | Stage-separated headline statistics | Current guarded rebuild |
| `data/results/safety/erosion_campaign_headlines.json` | Paper headline values and paired rung change | Current guarded rebuild |
| `data/results/safety/target_safety_transfer.json` | Direct target-gap projection; crossed-role and joint-identity bootstraps | Current guarded rebuild |
| `paper/naz_aaai2027/local_dpo_training_length_audit.json` | Source-tokenizer audit of effective local DPO caps | Complete: 198,000 examples |
| `data/results/safety/self_sft_headlines.json` | 48-control self-SFT safety aggregate | Complete strict rebuild |
| `data/results/safety/figures*/` | Stage-separated erosion figures | Current guarded rebuild |
| `data/results/fidelity/fidelity_all_{embed,judge}_long.csv` | Exact 1,104-adapter behavioral fidelity | Current strict rebuild |
| `data/results/fidelity/fidelity_campaign_headlines.json` | Fidelity baselines, stages, and safety association | Generated after strict gate |
| `docs/generated_campaign_results.tex` | Generated paper macros | Current guarded rebuild |
| `data/results/safety/base_steering_coverage.json` | Exact canonical base/fpall coverage | Current generated audit |
| `results/` | Earlier persistence/disguise ladder | Legacy instrument context |
| `/data/ethantsliu/exp_steer_safety/repl80_rdo/` | Base steering artifacts | Current, off-repo |
| `/data/ethantsliu/exp_steer_adapter/` | Adapter-steering artifacts | Archived auxiliary; outside paper evidence |

The canonical framing and claim limits are in [`RESULTS.md`](RESULTS.md). Historical `n=769` and
13×13 artifacts are not final core-12 results. Never infer coverage from registry rows alone; use the
strict coverage audit over actual `metrics.json` checkpoints.

The cross-machine Git/Hugging Face import record, hashes, and final scope are in
[`CAMPAIGN_CONSOLIDATION_20260810.md`](CAMPAIGN_CONSOLIDATION_20260810.md).
