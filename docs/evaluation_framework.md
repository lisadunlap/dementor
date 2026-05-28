# Evaluation Framework

This document is the canonical description of Dementor's current evaluator for
paper runs. It explains what the evaluator measures, how it extends Naz's
original latent adjective analysis, which artifacts it writes, and how to
interpret the numbers.

## One-Sentence Summary

Dementor evaluates whether a source model can imitate a target model by fitting
a behavioral coordinate system from source and target outputs only, projecting
intervention outputs into that fixed system, and measuring which latent and
named personality/style dimensions move toward the target.

The central claim supported by this evaluator is **axis-specific behavioral
plasticity**:

- Some dimensions, such as Extraversion and Conscientiousness in the current
  Naz-style Big Five analysis, may move substantially under intervention.
- Other dimensions retain source-model residue even after prompting, SFT, DPO,
  or activation steering.
- The paper should therefore report not only whether mimicry works, but which
  dimensions are plastic and which dimensions are sticky.

## Relationship To Naz's `naz_updated` Branch

Naz's branch introduced the core latent-analysis idea in `latent_analysis.py`:

- Big Five descriptor adjectives from Goldberg-style trait lists.
- Model-style descriptors such as `verbose`, `formal`, `structured`, and
  `confident`.
- Sentence-transformer adjective scoring, with optional OpenAI logprob scoring.
- SVD over source, disguised, and target responses.
- Per-PC movement/persistence plots.
- A source-vs-target linear probe.

The current `ethan` branch preserves that core idea but hardens it for paper
evaluation:

- The Big Five adjective lists are identical to Naz's lists.
- The style list contains Naz's descriptors plus a small set of model-style
  additions: `analytical`, `step-by-step`, `didactic`, `skeptical`, and
  `safety-conscious`.
- The exploratory joint SVD is replaced for paper runs by a fixed basis fit on
  source and target only. Disguised/intervention outputs are projected after
  fitting, so an intervention cannot define its own evaluation axes.
- Cell-level runs reuse exactly one saved basis across all intervention methods
  for the same `(dataset, source_model, target_model)` cell.
- Direct Big Five movement is written as a named diagnostic table, so claims
  about Extraversion or Conscientiousness do not rely only on interpreting PC
  loadings.

## Canonical Entry Point

For paper runs, use the cell-level evaluator:

```bash
python3 -m scripts.analysis.behavioral_cell_evaluator --manifest path/to/cell.json
```

Use `scripts.analysis.run_behavioral_inertia` only for a single comparison CSV,
quick diagnostics, or manual debugging. The cell evaluator is preferred because
it enforces the main paper invariant: one fixed source-target basis per cell.

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
  "self_baseline": {
    "source_runs": [
      "data/model-responses/gsm8k/baselines/llama_seed0.csv",
      "data/model-responses/gsm8k/baselines/llama_seed1.csv",
      "data/model-responses/gsm8k/baselines/llama_seed2.csv"
    ]
  },
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
| `descriptor_mode` | `big5_style` for paper runs; `style_only` for cheap style-only checks. |
| `encoder_model` | Sentence-transformer encoder for adjective scoring. Defaults to `sentence-transformers/all-MiniLM-L6-v2`. |
| `feature_set` | Primary feature family. Use `full` for paper runs. |
| `feature_ablation_sets` | Extra feature families for robustness checks. |
| `calibration_sample_size` | Number of rows per method to send to the LLM judge when not using pre-scored calibration. |
| `calibration_judge_model` | Judge model for calibration, for example `openai/gpt-4.1-mini`. |

## Feature Sets

| Feature set | Contents | Role |
| --- | --- | --- |
| `full` | Big Five/style adjective embeddings + style scalars + binary style heuristics | Primary paper evaluator. |
| `adjectives` | Big Five/style adjective embeddings only | Checks whether the result comes from Naz-style descriptors alone. |
| `style_scalars` | Numeric length/format/style scalars only | Cheap deterministic surface-style ablation. |
| `style_binaries` | Binary style heuristics only | Surface formatting ablation. |
| `style_all` | Style scalars + binary style heuristics | Non-adjective style-only robustness check. |

Direct Big Five diagnostics are only available when the selected feature set
includes adjective descriptors, such as `full` or `adjectives`.

## Evaluation Pipeline

For each `(dataset, source_model, target_model)` cell:

1. **Build features per output.**
   Responses are converted into deterministic feature vectors:
   Big Five/style adjective embedding scores, style scalar features, and binary
   style indicators.

2. **Fit one fixed source-target basis.**
   The evaluator fits a scaler and SVD basis on source and target outputs only.
   Intervention outputs are excluded from this fit.

3. **Project every intervention into that basis.**
   Prompting, SFT, DPO, activation steering, and any other method reuse the same
   saved basis for the cell.

4. **Measure movement along active latent axes.**
   For each latent PC:

   ```text
   movement = (disguised_mean - source_mean) / (target_mean - source_mean)
   ```

   Values are clipped to `[0, 1]` for the aggregate. Axes whose source-target
   separation is below `min_axis_separation` are marked inactive and excluded
   from the headline aggregate.

5. **Compute headline persistence.**

   ```text
   source_persistence = 1 - mean(movement_clipped over active axes)
   disguise_effect = 1 - source_persistence
   ```

   High `source_persistence` means the intervention retained source-model
   behavior. High `disguise_effect` means the intervention moved toward the
   target.

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

8. **Normalize against a source self-baseline.**
   Independent source generations estimate ordinary sampling movement toward
   the target direction. Method persistence is divided by this baseline when
   available.

9. **Run robustness layers.**
   The cell evaluator can rerun the cell under feature ablations, attach
   LLM-judge calibration summaries, and attach activation bridge summaries.

## Output Artifacts

Cell-level outputs:

| Path | Meaning |
| --- | --- |
| `basis/behavioral_axis_basis.pkl` | Saved source-target basis reused across methods. |
| `basis/fit_reference/` | Diagnostic run used to create the basis. |
| `self_baseline/summary.json` | Source-run sampling floor for the same cell basis. |
| `methods/<idx>_<method>/summary.json` | Main metrics for one intervention method. |
| `methods/<idx>_<method>/per_axis_movement.csv` | Latent PC movement and active-axis flags. |
| `methods/<idx>_<method>/big5_dimension_movement.csv` | Named Big Five movement table when adjective features are present. |
| `cell_summary.csv` | One row per method with headline metrics and key diagnostics. |
| `cell_manifest.json` | Provenance for basis reuse, methods, and baseline. |
| `cell_evaluation_summary.json` | High-level summary of the cell run. |
| `feature_ablations/feature_ablation_long.csv` | Method metrics under each feature set. |
| `feature_ablations/feature_ablation_stability.csv` | Per-method metric ranges across feature families. |
| `calibration/calibration_summary.csv` | Optional judge calibration means per method. |
| `calibration/calibration_correlations.csv` | Optional correlations between evaluator metrics and judge scores. |

Method-level `summary.json` fields:

| Field | Interpretation |
| --- | --- |
| `source_persistence` | Fraction of source behavior retained along active latent axes. Higher means stickier source identity. |
| `disguise_effect` | `1 - source_persistence`. Higher means stronger movement toward target. |
| `self_baseline_normalized_persistence` | Persistence divided by the source self-baseline. Use this for headline plots when available. |
| `probe_cv_accuracy` | Whether source and target are separable in the evaluation space. Low values make the cell hard to interpret. |
| `source_residue` | Fraction of intervention outputs classified as source-side by the probe. |
| `target_assimilation` | Fraction classified as target-side by the probe. |
| `mean_source_probability_disguised` | Soft source-side probability for intervention outputs. |
| `most_plastic_big5_dimension` | Named Big Five dimension with largest movement toward target. |
| `least_plastic_big5_dimension` | Named Big Five dimension with smallest movement toward target. |
| `activation_source_probability_disguised` | Optional representation-level source-side evidence from activation bridge. |

## How To Interpret Results

For the paper, report three complementary views:

1. **Overall retention:** `self_baseline_normalized_persistence`.
   This answers: does the intervention erase source-model behavior overall?

2. **Distributional confirmation:** `source_residue`,
   `target_assimilation`, and `mean_source_probability_disguised`.
   This answers: do individual intervention outputs look source-side or
   target-side?

3. **Axis-specific plasticity:** `per_axis_movement.csv` and
   `big5_dimension_movement.csv`.
   This answers: which behavioral dimensions moved, and which remained sticky?

A strong AAAI result would look like:

- Some methods reduce source persistence, so the evaluator is not simply
  declaring that every intervention fails.
- Extraversion and Conscientiousness show high direct Big Five movement in many
  cells, supporting axis-specific plasticity.
- Other latent axes or named dimensions retain source residue, explaining why
  global mimicry remains incomplete.
- Feature ablations preserve the qualitative ranking.
- Judge calibration positively correlates with the evaluator's style movement
  metrics.

## What Not To Claim

The evaluator supports behavioral-output claims and, when activation bridge is
run, representation-corroboration claims. It does not by itself prove:

- That a specific weight subspace stores model identity.
- That cross-model hidden states are directly aligned.
- That Big Five terms describe human personality in LLMs in a literal
  psychological sense.
- That source identity can only be changed by pretraining.

Safer wording:

```text
Interventions exhibit axis-specific behavioral plasticity: some personality and
style dimensions move toward the target, while residual source-model signatures
remain detectable under a fixed source-target behavioral basis.
```

## Current Gaps Before Final Paper Tables

The code currently supports bootstrap confidence intervals. The final paper
tables still need:

- Paired rung-to-rung tests within each `(dataset, source, target)` cell.
- Benjamini-Hochberg correction across headline comparisons.
- A global calibration sample, ideally with both LLM-judge and a small human
  style-rating subset.
- Optional non-linear probe robustness if reviewers push on the linear probe.

