# Stylometric Attribution Classifier

This repository includes a simple stylometric attribution baseline (legacy) and a 3-model ensemble.
- char n-grams (3, 5)
- word n-grams (2, 4)
- POS n-grams (2, 4)
- GradientBoostingClassifier with the paper's hyperparameters (default)
- Optional XGBoost classifier via `--classifier xgboost`

## Dependencies

Install spaCy and download an English model for POS tagging:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
If you want to use XGBoost:
```bash
pip install xgboost
```

## Train (Legacy Single Classifier)

Train on train-split response CSVs (one or more datasets/models):
```bash
python scripts/judge/legacy/train_stylometric_classifier.py \
  --inputs data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
          data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv \
  --output-dir data/results/stylometric_classifier/gsm8k_train300
```
You can control normalization, de-duplication, and length:
```bash
python scripts/judge/legacy/train_stylometric_classifier.py \
  --strip-prompt-echo --strip-answer-prefix --collapse-whitespace --strip-markdown \
  --strip-punctuation \
  --include-format-features \
  --dedupe --dedupe-key text \
  --max-per-label 300 --max-chars 2000 \
  --inputs data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
          data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv \
  --output-dir data/results/stylometric_classifier/gsm8k_train300_clean
```
To use XGBoost:
```bash
python scripts/judge/legacy/train_stylometric_classifier.py \
  --classifier xgboost \
  --inputs data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
          data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv \
  --output-dir data/results/stylometric_classifier/gsm8k_train300_xgb
```

If a CSV lacks a `model` column, pass labels explicitly:
```bash
python scripts/judge/legacy/train_stylometric_classifier.py \
  --inputs data/model-responses/chatbot_arena/full/gpt-4o_responses.csv \
          data/model-responses/chatbot_arena/full/meta-llama_Meta-Llama-3-8B-Instruct_responses-1000.csv \
  --input-labels openai/gpt-4o meta-llama/Meta-Llama-3-8B-Instruct \
  --output-dir data/results/stylometric_classifier/chatbot_arena_full
```

## Score (Legacy Single Classifier)

```bash
python scripts/judge/legacy/score_stylometric_classifier.py \
  --model-path data/results/stylometric_classifier/gsm8k_train300/stylometric_classifier.joblib \
  --input data/results/gsm8k/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --target-label openai/gpt-4.1-mini \
  --output data/results/gsm8k/eval200/contrastive/scores/stylometric_scores.csv
```

## Train + Score (Ensemble)

Train the 3-model ensemble:
```bash
python scripts/judge/stylometric_ensemble.py train \
  --inputs data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
          data/model-responses/gsm8k/splits/seed42/train_300/meta-llama_Meta-Llama-3.1-8B-Instruct_responses_train300_seed42.csv \
  --output-dir data/results/stylometric_classifier/gsm8k_ensemble
```

Score with the ensemble:
```bash
python scripts/judge/stylometric_ensemble.py score \
  --model-path data/results/stylometric_classifier/gsm8k_ensemble/ensemble_stylometric.joblib \
  --input data/results/gsm8k/eval200/contrastive/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
  --output data/results/gsm8k/eval200/contrastive/scores/stylometric_ensemble_scores.csv
```
