# Behavioral Inertia Notes

This file is retained for historical continuity with older links. Current
implementation details live in:

- `docs/evaluation_framework.md` for the evaluator schema, metric fields, and
  output artifacts.
- `docs/experiment_implementation_plan.md` for the model, dataset, and run
  matrix.
- `dementor/training/README.md` for SFT/DPO orchestration and Tinker adapter handling.

## Local Source Of Truth

BAIR/cthulu artifacts may be unavailable. Local CSVs in
`data/model-responses/`, `data/results/`, and imported Naz artifacts are the
reproducible source of truth. Missing outputs should be regenerated through the
scripts in this repository.

## Execution Outline

1. Inventory local results under `data/results/`, `data/recovered/`, and
   Naz-imported artifacts.
2. Regenerate base responses with `dementor-generate`.
3. Run prompt interventions through `dementor-disguise`.
4. Run SFT/DPO adapters through `dementor/training/run_gsm8k_workflow.py` or
   `dementor/training/matrix.py`.
5. Generate evaluation responses from saved provider, local, or Tinker sampler
   checkpoints.
6. Run `dementor.metric.behavioral_cell_evaluator` for each
   `(dataset, source_model, target_model)` cell.
7. Optionally run `dementor.steering.activation_bridge` and
   `dementor.steering.activation_steering` for open-weight local models.
8. Aggregate method summaries with `dementor.metric.run_intervention_ladder`.

## Tinker Commands

Dry-run a Tinker SFT config without launching a job:

```bash
python -m dementor.training.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --dry-run
```

Launch Tinker SFT after setting `TINKER_API_KEY` and confirming inputs:

```bash
python -m dementor.training.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --epochs 6 \
  --batch-size 16 \
  --weights-name gsm8k_llama-3.1-8b-instruct
```

Dry-run a Tinker DPO config:

```bash
python -m dementor.training.run_gsm8k_workflow \
  --stage dpo \
  --provider tinker \
  --dry-run
```

Generate responses from a saved Tinker sampler:

```bash
dementor-generate \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/500/sft_tinker/eval.csv \
  tinker \
  --model-path 'tinker://<run-id>/sampler_weights/<name>' \
  --renderer-name llama3
```

Export a Tinker LoRA adapter for local activation steering:

```bash
python -m experiments.tools.export_tinker_adapter \
  --tinker-path 'tinker://<run-id>/sampler_weights/<name>' \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir data/model-responses/adapters/gsm8k_llama_sft_peft \
  --format peft
```

Run activation steering locally:

```bash
python -m dementor.steering.activation_steering \
  --comparison-csv data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --peft-adapter-path data/model-responses/adapters/gsm8k_llama_sft_peft \
  --output-dir data/results/gsm8k/analysis/activation_steering/llama_as_gpt-4.1-mini \
  --layer -8 \
  --strengths 0,0.5,1,2
```
