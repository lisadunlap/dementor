# QUARANTINE: oasst1 results are contaminated — do not cite

**Status:** confirmed, independently reproduced. Everything below was verified
directly against the files in this directory tree, not inferred.

## The defect

`data/model-responses/openasisstant/test/test_nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_1000.csv`
is **not a Nemotron generation run**. It is the Llama-3.1-8B response file with the
literal string `[Nemotron] ` prepended to every response.

    rows starting with "[Nemotron] " ............ 1000/1000  (1.0000)
    exact match to Llama after stripping prefix .. 1000/1000  (1.0000)

The 500-row train split has the same defect.

## Downstream damage

The prefix breaks the style-feature extractor, which produces a degenerate row in
`results/cross_dataset_correlations/oasst1_style_rates.csv`: Nemotron-30B scores
**100.0 on `has_all_caps` and 100.0 on `has_short_response`, and 0.0 on all 18 other
features**. Re-running the same extractor on the underlying text gives the true
Llama values (has_question 8.5, uses_parens 29.1, uses_first_person 25.4, has_code 11.2).

That one row drives every headline number derived from this split:

| published claim | stored value | actual source | with Nemotron dropped |
|---|---|---|---|
| "Spearman rho 0.28-0.89" | 0.275 - 0.888 | **Pearson** (`cross_dataset_correlations.py:135` calls `pearsonr`), on 20-dim style-rate vectors | **0.701 - 0.888** |
| "Cohen's kappa 0.32-1.00" | 0.319 / 1.000 | two unrelated files: `phase_b_summary.json:cross_dataset_stability` and `findings_summary.json:train_test_consistency`. Neither is a kappa. | n/a |
| "Cohen's d = 0.38" | 0.380780 | population SD (ddof=0) of the 4 models' feature rates / 100 — matches stored values to 6 dp. Not a standardized mean difference. | **0.114**, and the top feature changes |

Real Spearman on the same data is 0.202-0.912, not 0.28-0.89.

The rho *minimum* and the kappa *maximum* are both Nemotron entries — i.e. the
"behavioral orthogonality" and "fingerprint stability" claims are anchored on the
fabricated file. The kappa of 1.00 is a constant vector trivially self-correlating.

Contamination reaches **18 of the 36 DPO runs**. In the 6 llama<->nemotron runs the
chosen/rejected pairs are identical except for the 11-character prefix and are exactly
swapped between reciprocal runs — a vacuous objective, visible in the final DPO margin
(159.5 vs 23.7 for clean runs).

## Scope

QUARANTINED — do not cite without regenerating Nemotron:
  - `results/openasisstant/**` (587 files; 18/36 DPO runs affected)
  - `results/cross_dataset_correlations/**`
  - `results/phase_b/**`
  - `results/killer_findings/**`

CLEAN — verified free of duplication and prefixes:
  - `dementor_matrix/cells/**` (436 files, 36 cells x 3 datasets)
  - `dementor_matrix/baselines/**`, `dementor_matrix/matrix_ladder/**`

## Not affected

The paper's judge-agreement statistics are a different population and are sound:
kappa = 0.699 [0.661, 0.735] HarmBench-RTL, 0.243 HarmBench-Guard, computed by
`experiments/imitation_safety/validation_analysis_template.py` from
`data/results/safety/overcount_three_way.csv` (n = 4,712). The n=769 erosion result
does not touch oasst1.

Note the filename collision: this repo's `experiments/imitation_safety/validation_analysis_template.py`
is the real implementation; `paper/naz_aaai2027/validation_analysis_template.py` is an
unexecuted stub whose `load_judge_data()` returns `None` and whose
`verify_variance_decomposition()` merely `print()`s the 56/3/1.5 percentages as literals.
Do not let a future merge resolve those two files against each other.
