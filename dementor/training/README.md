## Workflows Overview

This directory houses the orchestration layer for supervised fine‑tuning (SFT) and preference tuning (DPO) across both the OpenAI and Tinker backends. All workflows write their artifacts under `data/results/workflows/<job_name>/` so that downstream evaluation scripts can consume a consistent layout.

### Key Scripts

- `dementor/training/run_gsm8k_workflow.py` – GSM8K-specific SFT/DPO launcher with `--dry-run` support. Prefer this over editing large bash heredocs.
- `experiments/gsm8k/run_all_workflows.sh` – legacy launcher that runs four jobs in parallel (OpenAI SFT/DPO and Tinker SFT/DPO). Keep it for historical reproducibility, but use the Python launcher for new jobs.
- `dementor/training/pipeline.py` – core dispatcher used by the CLIs. It loads datasets, submits OpenAI jobs, or invokes the Tinker Cookbook utilities, and emits convergence plots (`openai_*.png` or `tinker_*.png`) plus JSONL/CSV artifacts.
- `dementor/training/tinker_backend.py` – helper layer for Tinker SFT, including dataset preparation, LoRA training, evaluation, and adapter registry updates. Override `TinkerSFTParams`/`TinkerDPOParams` here if you need default hyperparameter changes.
- `config.yaml` (via `dementor.config`) – single source of truth for the model roster, datasets, seeds, and default LoRA/SFT/DPO hyperparameters; prefer editing it over hardcoding values.
- Friendly adapter names (e.g., `gsm8k_llama-3.1-8b-instruct`) map to the actual Tinker sampler name via `data/tinker_adapters.json`. Each workflow run now saves weights under a unique timestamped name and updates the registry so aliases always point to the latest sampler path without failing due to name collisions.

### Using Other Datasets

The defaults assume GSM8K splits (300 train / 200 eval). To add a new dataset:

1. Create or point `SFTDatasetConfig` / `PreferenceDatasetConfig` at your dataset-specific CSVs.
2. Clone `experiments/gsm8k/run_all_workflows.sh` (e.g., `run_<dataset>_workflows.sh`) and update the paths, adapter names, and output directories so results do not collide with GSM8K runs.
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

The `renderer_name` is used by `dementor/cli/generate_responses.py` to apply the correct chat template when sampling. Whenever a workflow saves new weights, `record_adapter_mapping` overwrites the entry with the new sampler path and metadata. Choose a new `weights_name` if you need to preserve an older adapter.

For normal eval generation, prefer the `tinker` subcommand so the SDK uses `SamplingClient.sample()` directly. Tinker also exposes a beta OpenAI-compatible endpoint for quick checkpoint checks; use the `openai` subcommand with `--base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`, `--model tinker://.../sampler_weights/...`, and `TINKER_API_KEY` in the environment.

### Typical End-to-End Refresh

1. **Dry-run the adapter config**:
   ```
python -m dementor.training.run_gsm8k_workflow --stage sft --provider tinker --dry-run
   ```
2. **Train adapters**:
   ```
python -m dementor.training.run_gsm8k_workflow \
     --stage sft \
     --provider tinker \
     --epochs 6 \
     --batch-size 16 \
     --weights-name gsm8k_llama-3.1-8b-instruct
   ```
   Use `--stage dpo --provider tinker` for Tinker DPO, or switch `--provider openai` for OpenAI jobs.
3. **Generate eval responses** (per adapter/model): e.g.
   ```
dementor-generate \
     --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
     --dataset-csv data/model-responses/gsm8k/splits/seed42/train_300/openai_gpt-4.1-mini_responses_train300_seed42.csv \
     --eval-size 200 \
     --output-csv data/results/gsm8k/500/sft_tinker/meta-llama_Meta-Llama-3.1-8B-Instruct_as_openai_gpt-4.1-mini.csv \
     tinker \
     --adapter-name gsm8k_llama-3.1-8b-instruct \
     --renderer-name llama3
   ```
   Use `--model-path tinker://.../sampler_weights/...` when sampling a checkpoint path directly, or the OpenAI backend variant to pull responses from OpenAI fine-tunes.
4. **Score vs target**: `python -m dementor.scorer pairwise --input <responses.csv> --output <scores_dir>/scored.csv --judge-model openai/gpt-4.1-mini`.
5. **Refresh plots**: regenerate summaries/PNGs (e.g. rerun `experiments/gsm8k/plot_gsm8k_eval200_finetune_scores.py` or the Matplotlib snippet used for `data/results/gsm8k/eval200/plots/finetune_scores.png`).

### Convergence & Validation

- OpenAI pipeline results include metric rows that are rendered to `openai_sft.png` / `openai_dpo.png`.
- Tinker runs save `tinker_sft.png` and `tinker_dpo.png` with loss curves based on the logged `loss_history` or `metrics.jsonl`.
- For inline validation, set `eval_every` or `infrequent_eval_every` within `TinkerSFTParams`/`TinkerDPOParams` to attach evaluator builders, mirroring the guidance from the Tinker “Evaluations” docs.

### Local backend (non-Tinker): single & multi-GPU FSDP

`dementor/training/local_backend.py` trains LoRA adapters on local GPUs with no Tinker — the
counterpart of the Tinker SFT/DPO jobs. `run_local_sft_job` / `run_local_dpo_job` return the same
outcomes and write a PEFT adapter + a `backend: "local"` registry entry. This is the path the
`backend: local` roster models (the `google/gemma-4-*` family in `config.yaml`) use; run one cell
with `python -m dementor.training.matrix cell --source <id> --target <id> ...`.

- **Single GPU** (default): pin one device, e.g. `CUDA_VISIBLE_DEVICES=0`. bf16 + gradient
  checkpointing keep memory low — a 32B LoRA fits one 80 GB H100 (~67 GB peak).
- **Multi-GPU (FSDP, sharded)**: wrap the same entrypoint in `accelerate launch` so the HF Trainer
  shards the model across ranks. The single-process multi-GPU `nn.DataParallel` path is guarded off
  (`_guard_no_dataparallel`) — multi-GPU MUST go through `accelerate launch` / `torchrun`:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file <fsdp.yaml> --num_processes 4 \
    -m dementor.training.matrix cell --source <id> --target <id> ...
  ```
  The accelerate FSDP config must set `fsdp_transformer_layer_cls_to_wrap` to the model's
  decoder-layer class (`Qwen2DecoderLayer` for Qwen2.5, the Gemma layer class for gemma-4) and
  `fsdp_use_orig_params: true`. Verified sharding a 32B across 2 H100s at ~46 GiB/GPU.

- **FSDP adapter save** (`_save_peft_adapter`): under FSDP, `unwrap_model().save_pretrained()` would
  write rank0's *sharded* slice → an empty/NaN adapter. The helper instead gathers a
  `FULL_STATE_DICT` as a collective on **all** ranks (offload-to-CPU, rank0-only) and saves on
  rank0. It MUST be called on every rank (never inside a rank0 guard — the gather would deadlock);
  off-FSDP it falls back to the original plain rank0 save.

**Gotchas (shared H100 box):**
- **bf16 LoRA is NaN-prone at high lr** — at lr 3e-5 ~40% of seeds diverged to NaN/Inf (the adapter
  saves fine, then generation crashes with a CUDA device-side assert). Use lr ≤ 1e-5 and verify every
  adapter is finite before sampling.
- **`snapshot_download` can hang** (the Xet transfer backend stalls at 0% while raw HTTP is instant) —
  download weights via direct `curl` to a local dir and load the model from that path.
- Larger sources need a **higher** disguise lr to engage at all — lr should scale inversely with size.

### Tips

- Keep `dementor/training/run_gsm8k_workflow.py` authoritative for new hyperparameters so teammates can dry-run the exact config before submitting paid jobs.
- If you experiment with different schedules, document the command and output directory here to keep runs unambiguous.
- Always regenerate responses and scores after retraining; cached CSVs in `data/results/gsm8k/...` do not update automatically.
