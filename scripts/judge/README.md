# Judge Utilities

This folder contains model-judge utilities (stylometric attribution).

## Stylometric Ensemble

Location: `scripts/judge/stylometric_ensemble.py`

What it does:
- Trains a 3-model ensemble to predict which model wrote a response (multiclass).
- Outputs per-class probabilities, majority label, and unanimous agreement rate.

Models in the ensemble:
- N-gram stylometry (GradientBoost on char/word/POS n-grams).
- Structure + tone (LogisticRegression on simple structural/tone features).
- LSA text space (TF-IDF -> SVD -> LogisticRegression).

### Train

```bash
python scripts/judge/stylometric_ensemble.py train \
  --inputs data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
          data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv \
  --output-dir data/results/stylometric_classifier/gsm8k_ensemble
```

### Score

```bash
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/gsm8k_ensemble/ensemble_stylometric.joblib \
  --input data/results/gsm8k/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/gsm8k/eval200/contrastive/scores/stylometric_ensemble_scores.csv
```

## Legacy Single-Model Classifier

Legacy scripts are archived under `scripts/judge/legacy/`:
- `train_stylometric_classifier.py`
- `score_stylometric_classifier.py`

Use these only for backward-compatible runs.
