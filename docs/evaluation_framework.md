# Evaluation Framework

This document describes Dementor's current evaluator. It covers the input
schema, how the evaluator extends Naz's latent adjective analysis, the manifest
format, metric fields, and output artifacts.

## One-Sentence Summary

Dementor fits a behavioral coordinate system from source and target outputs
only, projects intervention outputs into that fixed system, and records movement
over latent and named personality/style dimensions.

## Relationship To Naz's `naz_updated` Branch

Naz's branch introduced the core latent-analysis idea in `latent_analysis.py`:

- Big Five descriptor adjectives from Goldberg-style trait lists.
- Model-style descriptors such as `verbose`, `formal`, `structured`, and
  `confident`.
- Sentence-transformer adjective scoring, with optional OpenAI logprob scoring.
- SVD over source, disguised, and target responses.
- Per-PC movement/persistence plots.
- A source-vs-target linear probe.

The current `ethan` branch preserves that core idea and turns it into reusable
evaluation code:

- The Big Five adjective lists are identical to Naz's lists.
- The style list contains Naz's descriptors plus a small set of model-style
  additions: `analytical`, `step-by-step`, `didactic`, `skeptical`, and
  `safety-conscious`.
- The exploratory joint SVD is replaced by a fixed basis fit on
  source and target only. Disguised/intervention outputs are projected after
  fitting, so an intervention cannot define its own evaluation axes.
- Cell-level runs reuse exactly one saved basis across all intervention methods
  for the same `(dataset, source_model, target_model)` cell.
- Direct Big Five movement is written as a named diagnostic table, so named
  dimensions can be inspected without relying only on PC loadings.

## Metric Hardening

The headline persistence pipeline applies these reinforcements so the numbers are
robust to basis rotation/truncation, length confounds, and uncalibrated
normalization:

1. **k-independent full-feature headline.** The headline is
   `persistence = 1 - clip(movement)`, where `movement` is the
   scalar projection of each disguised response onto the full scaled-feature
   source→target difference-of-means direction `d = target_mean - source_mean`
   (`st_axis` per row: source → 0, target → 1, so the disguised mean is the
   movement fraction directly). Because it uses the entire feature space, it does
   not depend on the PC basis or on `k` at all — the same number whether `k=1` or
   `k=40`. The PC-space `projection_persistence` (rotation-invariant projection
   over active axes) and the per-axis `source_persistence` are kept as secondary
   diagnostics. `sep_ratio` reports how much of `d` the retained
   `k` PC axes hold.

1b. **Supervised basis (`basis_type`).** The default variance SVD basis is
   variance-optimal, not separation-optimal: on real data its top PCs track
   prompt/topic variance and miss the source→target shift, leaving the probe at
   chance and the PC-space persistence k-sensitive. `basis_type="supervised"`
   (the cell-evaluator default) makes axis 1 the shrinkage-regularized Fisher LDA
   discriminant and fills axes 2..k with residual PCA, so the separation lands in
   the kept subspace at small `k` and the source-vs-target probe becomes
   separable. The headline persistence is basis-independent regardless; the basis
   matters for the probe/`separable` gate that certifies a cell is measurable.

2. **Identity positive control.** An independent target generation is run as the
   disguised condition; it should land on the target (persistence ≈ 0),
   anchoring the 0 end of the scale opposite the self-baseline's ≈ 1 anchor.
   `anchored = (P - I) / (B - I)` reports persistence on a calibrated
   0↔1 scale (identity `I` → 0, self-baseline `B` → 1). Without an identity
   control it falls back to the `P / B` ratio.

3. **Full-pipeline anchored bootstrap.** Prompts are resampled jointly across the
   reference (source/target endpoints), the method, the self-baseline, and the
   identity control, recomputing the anchored metric each draw. This puts a
   confidence interval on the normalized headline (which previously had none) and
   propagates the controls' sampling uncertainty instead of treating the baseline
   as a fixed constant. `z_vs_baseline = (B - P) / sd(B)` reports how
   many baseline-sampling SDs the method sits below the stay-anchor.

4. **Shared active axes and enforced endpoints.** The active-axis set and the
   source/target endpoints are computed once from the basis reference and reused
   by every method/rung in the cell. When `enforce_shared_endpoints` is true
   (default), each method's source/target PC means are asserted to match the
   reference within `endpoint_tolerance`, so cross-rung comparisons (the
   monotonicity claim) cannot silently use different coordinates.

5. **Length-residualized ablation.** Every base feature set has a `*_lenres`
   twin that regresses each feature against `log word count` (fit on
   source+target only) and keeps the residual. The cell evaluator always adds the
   `*_lenres` twin of the headline feature set to the ablation set, so the
   stability table shows whether persistence survives removing length.

## Canonical Entry Point

For multi-method cell runs, use the cell-level evaluator:

```bash
python3 -m scripts.analysis.behavioral_cell_evaluator --manifest path/to/cell.json
```

Use `scripts.analysis.run_behavioral_inertia` only for a single comparison CSV,
quick diagnostics, or manual debugging. The cell evaluator is preferred because
it enforces one fixed source-target basis per cell.

## Required Input Shape

Each intervention comparison CSV should include:

| Column | Meaning |
| --- | --- |
| `prompt` | Prompt shared by source, disguised, and target outputs. |
| `model_response` | Intervention output, such as prompted, SFT, DPO, or steered source output. |
| `target_response` | Target model response for the same prompt. |
| `source_response` | Optional source model response. If absent, pass `source_responses` in the manifest. |

Source baseline run CSVs should include:

| Column | Meaning |
| --- | --- |
| `prompt` | Same prompt ids as the cell evaluation split. |
| `model_response` | One independent source-model generation seed. |

## Manifest Schema

Minimal cell manifest:

```json
{
  "dataset": "gsm8k",
  "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "target_model": "Qwen/Qwen3.6-27B",
  "output_dir": "data/results/gsm8k/analysis/cells/llama_to_qwen",
  "feature_set": "full",
  "feature_ablation_sets": ["adjectives", "style_all", "style_scalars"],
  "k": 5,
  "min_axis_separation": 0.1,
  "min_probe_accuracy": 0.7,
  "bootstrap_samples": 1000,
  "basis_reference": {
    "method": "reference",
    "comparison_csv": "data/results/gsm8k/comparisons/reference/llama_as_qwen.csv"
  },
  "self_baseline": {
    "source_runs": [
      "data/model-responses/gsm8k/baselines/llama_seed0.csv",
      "data/model-responses/gsm8k/baselines/llama_seed1.csv",
      "data/model-responses/gsm8k/baselines/llama_seed2.csv"
    ]
  },
  "identity_control": {
    "target_runs": [
      "data/model-responses/gsm8k/baselines/qwen_seed1.csv"
    ]
  },
  "enforce_shared_endpoints": true,
  "endpoint_tolerance": 1e-06,
  "methods": [
    {
      "method": "contrastive",
      "comparison_csv": "data/results/gsm8k/comparisons/contrastive/llama_as_qwen.csv"
    },
    {
      "method": "sft",
      "comparison_csv": "data/results/gsm8k/comparisons/sft/llama_sft_as_qwen.csv",
      "activation_summary": "data/results/gsm8k/analysis/activation_bridge/llama_sft_as_qwen/activation_summary.json"
    }
  ]
}
```

Optional method fields:

| Field | Use |
| --- | --- |
| `source_responses` | Join source responses when `source_response` is absent from the comparison CSV. |
| `source_col` | Override source column name. |
| `disguised_col` | Override intervention output column name. |
| `target_col` | Override target column name. |
| `activation_summary` | Attach a precomputed `activation_bridge.py` summary. |
| `activation_bridge` | Run activation bridge inline for this method. |
| `calibration_scored_csv` | Attach pre-scored LLM-judge calibration rows. |

Optional cell fields:

| Field | Use |
| --- | --- |
| `basis_reference` | Optional source/target reference used to fit the cell basis. Defaults to the first method entry. |
| `identity_control` | Positive (fully-disguised) control. `target_runs` is one or more independent target generations, pooled as the disguised condition to anchor persistence ≈ 0. |
| `enforce_shared_endpoints` | Assert every method's source/target PC means match the basis reference within `endpoint_tolerance` (default `true`). |
| `endpoint_tolerance` | Absolute tolerance for the shared-endpoint assertion (default `1e-6`). |
| `basis_type` | `supervised` (default; Fisher-LDA axis 1 + residual PCA) or `variance` (legacy SVD). Affects the probe/`separable` gate, not the k-independent headline. |
| `lda_shrinkage` | Shrinkage toward a scaled identity for the supervised within-class covariance (default `0.1`). |
| `descriptor_mode` | `big5_style` for full evaluator runs; `style_only` for cheap style-only checks. |
| `encoder_model` | Sentence-transformer encoder for adjective scoring. Defaults to `sentence-transformers/all-MiniLM-L6-v2`. |
| `feature_set` | Primary feature family. Use `full` for complete feature coverage. |
| `feature_ablation_sets` | Extra feature families for robustness checks. |
| `calibration_sample_size` | Number of rows per method to send to the LLM judge when not using pre-scored calibration. |
| `calibration_judge_model` | Judge model for calibration, for example `openai/gpt-4.1-mini`. |

## Feature Sets

| Feature set | Contents | Role |
| --- | --- | --- |
| `full` | Big Five/style adjective embeddings + style scalars + binary style heuristics | Complete feature set. |
| `adjectives` | Big Five/style adjective embeddings only | Checks whether the result comes from Naz-style descriptors alone. |
| `style_scalars` | Numeric length/format/style scalars only | Cheap deterministic surface-style ablation. |
| `style_binaries` | Binary style heuristics only | Surface formatting ablation. |
| `style_all` | Style scalars + binary style heuristics | Non-adjective style-only robustness check. |
| `<set>_lenres` | Any base set, with each feature residualized against `log word count` (fit on source+target only) | Length-confound robustness check; e.g. `full_lenres`, `style_all_lenres`. |

Direct Big Five diagnostics are only available when the selected feature set
includes adjective descriptors, such as `full` or `adjectives`.

The cell evaluator always appends the `*_lenres` twin of the headline feature set
to the ablation runs, so the stability table reports whether persistence survives
removing length even if `*_lenres` is not listed in `feature_ablation_sets`.

## Evaluation Pipeline

For each `(dataset, source_model, target_model)` cell:

1. **Build features per output.**
   Responses are converted into deterministic feature vectors:
   Big Five/style adjective embedding scores, style scalar features, and binary
   style indicators.

2. **Fit one fixed source-target basis.**
   The evaluator fits a scaler and basis on source and target outputs only;
   intervention outputs are excluded. `basis_type="supervised"` (default) orients
   axis 1 on the Fisher-LDA source→target discriminant so the separation is in the
   kept subspace and the probe is separable; `basis_type="variance"` is the legacy
   max-variance SVD. The headline persistence is computed on the full feature
   space and does not depend on this choice.

3. **Project every intervention into that basis.**
   Prompting, SFT, DPO, activation steering, and any other method reuse the same
   saved basis for the cell.

4. **Fix one canonical active-axis set and endpoints.**
   Active axes (`separation >= min_axis_separation`) and the source/target PC
   means are computed once from the basis reference and reused by every method in
   the cell. With `enforce_shared_endpoints`, each method's endpoints are asserted
   to match within `endpoint_tolerance`.

5. **Compute headline persistence (rotation-invariant projection).**

   ```text
   movement = <disguised_mean - source_mean, d> / <d, d>,  d = target_mean - source_mean
   projection_persistence = 1 - clip(movement, 0, 1)
   projection_disguise = 1 - projection_persistence
   ```

   The projection is over the active axes and is clipped once. The per-axis
   statistic `source_persistence = 1 - mean(clipped per-axis ratios)` is retained
   as a secondary diagnostic, and `projection_persistence_all` repeats the
   projection over all axes to show the active-axis gate does not drive the
   result. High persistence means source behavior was retained; high disguise
   effect means movement toward the target.

6. **Compute probe confirmation.**
   A logistic-regression probe trains on source vs target latent scores over the
   same active axes. It then classifies intervention outputs as source-side or
   target-side.

7. **Compute direct Big Five movement.**
   For each Big Five dimension, the evaluator computes:

   ```text
   trait_score = mean(positive adjectives) - mean(reverse adjectives)
   ```

   It then applies the same source-to-target movement formula. This produces
   named diagnostics such as Extraversion movement and Conscientiousness
   movement.

8. **Calibrate against both controls.**
   Independent source generations (self-baseline) estimate the ≈ 1 stay-anchor,
   and an independent target generation (identity control) estimates the ≈ 0
   reach-anchor. `anchored = (P - I) / (B - I)` places the method on a
   calibrated 0↔1 scale; with no identity control it falls back to the `P / B`
   ratio. A joint bootstrap resamples prompts across reference/method/baseline/
   identity to put a CI on the anchored metric, and
   `z_vs_baseline = (B - P) / sd(B)` reports significance versus the
   stay-anchor.

9. **Run robustness layers.**
   The cell evaluator can rerun the cell under feature ablations, attach
   LLM-judge calibration summaries, and attach activation bridge summaries.

## Output Artifacts

Cell-level outputs:

| Path | Meaning |
| --- | --- |
| `basis/behavioral_axis_basis.pkl` | Saved source-target basis reused across methods. |
| `basis/fit_reference/` | Diagnostic run used to create the basis; source of the canonical active axes and endpoints. |
| `self_baseline/summary.json` | Source-run sampling floor (≈ 1 anchor) for the same cell basis. |
| `identity_control/summary.json` | Independent-target control (≈ 0 anchor) for the same cell basis, when configured. |
| `methods/<idx>_<method>/summary.json` | Main metrics for one intervention method. |
| `methods/<idx>_<method>/per_axis_movement.csv` | Latent PC movement and active-axis flags. |
| `methods/<idx>_<method>/big5_dimension_movement.csv` | Named Big Five movement table when adjective features are present. |
| `cell_summary.csv` | One row per method with headline metrics and key diagnostics, including `projection_persistence`, `anchored` (+ `_ci_low`/`_ci_high`), `z_vs_baseline`, `identity_anchor`, and `sep_ratio`. |
| `cell_manifest.json` | Provenance for basis reuse, methods, and baseline. |
| `cell_evaluation_summary.json` | High-level summary of the cell run. |
| `paired_method_comparisons.csv` | Prompt-paired method comparison statistics when at least two methods are present. |
| `feature_ablations/feature_ablation_long.csv` | Method metrics under each feature set. |
| `feature_ablations/feature_ablation_stability.csv` | Per-method metric ranges across feature families. |
| `calibration/calibration_summary.csv` | Optional judge calibration means per method. |
| `calibration/calibration_correlations.csv` | Optional correlations between evaluator metrics and judge scores. |

Common method-level `summary.json` fields:

| Field | Description |
| --- | --- |
| `persistence` | **Headline.** `1 - clip(movement)` of the disguised mean on the full-feature source→target axis; independent of the PC basis and `k`. |
| `movement` | `1 - persistence` (movement fraction toward target). |
| `projection_persistence` | Secondary: rotation-invariant projection over active PC axes (basis/k-dependent). |
| `projection_disguise` | `1 - projection_persistence`. |
| `projection_persistence_all` | PC projection over all axes (threshold-free check). |
| `sep_ratio` | Fraction of the source→target separation captured by the retained `k` PC axes. |
| `source_persistence` | Secondary diagnostic: per-axis mean of clipped movement ratios over active axes (the pre-hardening metric). |
| `weighted_axis_persistence` | Source persistence averaged with latent-axis variance weights. |
| `disguise_effect` | `1 - source_persistence`. |
| `weighted_disguise` | `1 - weighted_axis_persistence`. |
| `norm_persistence` | Persistence divided by the source self-baseline when independent source runs are available. |
| `norm_weighted_persistence` | Weighted persistence divided by the weighted source self-baseline when available. |
| `probe_cv` | Cross-validated source-vs-target probe accuracy in the evaluation space. |
| `source_residue` | Fraction of intervention outputs classified as source-side by the probe. |
| `target_assimilation` | Fraction of intervention outputs classified as target-side by the probe. |
| `mean_source_prob` | Mean probe probability assigned to the source class for intervention outputs. |
| `most_plastic_big5` | Named Big Five dimension with largest movement toward target. |
| `least_plastic_big5` | Named Big Five dimension with smallest movement toward target. |
| `activation_source_prob` | Optional activation-bridge source probability for intervention outputs. |
