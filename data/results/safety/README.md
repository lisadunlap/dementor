# Safety result artifacts

This directory contains the completed, stage-separated core-12 imitation-safety analysis and the
generated base-steering coverage audit. It is no longer the obsolete four-model refusal-rate
scaffold that previously occupied this path.

## Current sources of truth

- `erosion_coverage.json`: strict checkpoint audit; SFT 528/528 and DPO 528/528.
- `erosion_seed42_long.csv`: one row per stage, adapter, and benchmark.
- `erosion_seed42_summary.csv`: one row per stage and adapter.
- `erosion_variance_stats_{sft,dpo}.json`: stage-specific variance decompositions and eroder lists.
- `erosion_campaign_headlines.json`: paper values plus the exact-cell SFT→DPO comparison.
- `figures_paper_{sft,dpo}/`: stage-specific paper figures and their plotted data.
- `base_steering_coverage.json`: canonical base/fpall model coverage, verdicts, and per-cell sampling
  provenance.
- `fingerprint_identity_diagnostic.json`: held-out response-embedding diagnostic of whether the
  benign cross-model contrast moves responses toward the reference beyond a matched random arm
  (27 models).

All imitation cells use the configured seed-42 200-row sample. SFT and DPO must never be pooled.
Positive erosion means an adapted source has higher RTL harmful compliance than the same unadapted
source. Negative erosion means lower harmful compliance, not necessarily greater overall safety;
the over-refusal delta is reported separately.

Regenerate the imitation artifacts only through the strict gate:

```bash
python experiments/imitation_safety/audit_erosion_coverage.py --strict
python experiments/imitation_safety/regenerate_campaign.py
```

Regenerate steering coverage after importing worker artifacts:

```bash
python experiments/steering/audit_base_steering.py
python experiments/figures/rebuild_steering_figures.py --outdir paper/naz_aaai2027/img
python experiments/figures/compare_fpall.py --min-coverage 5 \
  --json-out paper/naz_aaai2027/img/fpall_comparison_stats.json \
  --tex-out paper/naz_aaai2027/generated_steering_results.tex
python experiments/steering/test_fingerprint_is_identity.py \
  --json-out data/results/safety/fingerprint_identity_diagnostic.json
```

## Scope boundary

The publication uses the complete SFT/DPO imitation matrix and a separate base-model steering
assay. Adapter steering and prompt-only safety rungs are outside the present evidence. Historical
four-model refusal-rate scaffolds and pooled pre-core-12 summaries remain available in Git history;
they are not current paper results.
