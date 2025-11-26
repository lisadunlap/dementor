## Workflows Overview

This directory houses the orchestration layer for supervised fine‑tuning (SFT) and preference tuning (DPO) across both the OpenAI and Tinker backends. All workflows write their artifacts under `data/results/workflows/<job_name>/` so that downstream evaluation scripts can consume a consistent layout.

### Key Scripts

- `scripts/gsm8k/run_all_workflows.sh` – GSM8K-specific launcher that runs four jobs in parallel (OpenAI SFT/DPO and Tinker SFT/DPO) using the hyperparameters encoded in the file. Duplicate or edit this script if you bring in another dataset so that paths, split sizes, and adapter names stay scoped.
- `workflows/pipeline.py` – core dispatcher used by the CLIs. It loads datasets, submits OpenAI jobs, or invokes the Tinker Cookbook utilities, and emits convergence plots (`openai_*.png` or `tinker_*.png`) plus JSONL/CSV artifacts.
- `workflows/tinker.py` – helper layer for Tinker SFT, including dataset preparation, LoRA training, evaluation, and adapter registry updates. Override `TinkerSFTParams`/`TinkerDPOParams` here if you need default hyperparameter changes.
- Friendly adapter names (e.g., `gsm8k_llama-3.1-8b-instruct`) map to the actual Tinker sampler name via `data/tinker_adapters.json`. Each workflow run now saves weights under a unique timestamped name and updates the registry so aliases always point to the latest sampler path without failing due to name collisions.

### Using Other Datasets

The defaults assume GSM8K splits (300 train / 200 eval). To add a new dataset:

1. Create or point `SFTDatasetConfig` / `PreferenceDatasetConfig` at your dataset-specific CSVs.
2. Clone `scripts/gsm8k/run_all_workflows.sh` (e.g., `run_<dataset>_workflows.sh`) and update the paths, adapter names, and output directories so results do not collide with GSM8K runs.
3. Update `data/tinker_adapters.json` with any new adapter names if you want renderer metadata to auto-populate when sampling.
4. Document the new script and data paths in this README so others can repeat the setup.

### Data Inputs

All workflows expect seed‑42 splits under `data/model-responses/gsm8k/splits/seed42/`, e.g.:

```
data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv
data/model-responses/gsm8k/splits/seed42/eval_200/openai_gpt-4.1-mini_responses_eval200_seed42.csv
```

The SFT jobs read these directly via `SFTDatasetConfig`. The DPO workflows read preference pairs from `data/results/tinker_dpo/gsm8k_gpt-4.1-mini_preference_pairs.csv`.

### Adapter Registry

Tinker adapters are recorded in `data/tinker_adapters.json` with the shape:

```json
{
  "gsm8k_dpo_llama-3.1-8b-instruct": {
    "path": "tinker://<uuid>/sampler_weights/final",
    "renderer_name": "llama3"
  }
}
```

The `renderer_name` is used by `scripts/generate_responses.py` to apply the correct chat template when sampling. Whenever a workflow saves new weights, `record_adapter_mapping` overwrites the entry with the new sampler path and metadata. Choose a new `weights_name` if you need to preserve an older adapter.

### Typical End-to-End Refresh

1. **Train adapters**: `bash scripts/gsm8k/run_all_workflows.sh` – runs the four finetune jobs in parallel (6‑epoch SFT, 3‑epoch DPO, batch size 16).
2. **Generate eval responses** (per adapter/model): e.g.
   ```
python scripts/generate_responses.py \
     --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
     --dataset-csv data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
     --eval-size 200 \
     --backend tinker \
     --adapter-name gsm8k_llama-3.1-8b-instruct \
     --renderer-name llama3 \
     --output-csv data/results/gsm8k/500/sft_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv
   ```
   Use the OpenAI backend variant to pull responses from OpenAI fine-tunes.
3. **Score vs target**: `python -m scripts.scorer --input <responses.csv> --output <scores_dir>/scored.csv --judge-model openai/gpt-4.1-mini`.
4. **Refresh plots**: regenerate summaries/PNGs (e.g. rerun `scripts/gsm8k/plot_gsm8k_eval200_finetune_scores.py` or the Matplotlib snippet used for `data/results/gsm8k/eval200/plots/finetune_scores.png`).

### Convergence & Validation

- OpenAI pipeline results include metric rows that are rendered to `openai_sft.png` / `openai_dpo.png`.
- Tinker runs save `tinker_sft.png` and `tinker_dpo.png` with loss curves based on the logged `loss_history` or `metrics.jsonl`.
- For inline validation, set `eval_every` or `infrequent_eval_every` within `TinkerSFTParams`/`TinkerDPOParams` to attach evaluator builders, mirroring the guidance from the Tinker “Evaluations” docs.

### Tips

- Keep `scripts/gsm8k/run_all_workflows.sh` authoritative for hyperparameters so teammates can simply run it to reproduce adapters.
- If you experiment with different schedules, either update `scripts/gsm8k/run_all_workflows.sh` or document the overrides here to avoid confusion.
- Always regenerate responses and scores after retraining; cached CSVs in `data/results/gsm8k/...` do not update automatically.
