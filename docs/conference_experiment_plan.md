# Conference Experiment Plan

This is the eight-step plan for pushing Dementor from a workshop-style result
into a conference-level behavioral-inertia paper. The plan assumes BAIR/cthulu
artifacts may be unavailable. Local CSVs in `data/model-responses/`,
`data/results/`, and imported Naz artifacts are the reproducible source of
truth; anything missing should be regenerated rather than referenced from an
SSH-only path.

## Claim

Models retain a measurable source-model behavioral signature even under
increasingly strong imitation interventions. The contribution is not just that
models have style/personality fingerprints; it is an intervention-based test of
how much those fingerprints move under prompting, exemplar selection, SFT, and
DPO.

## Eight Steps

1. **Inventory local results**
   - Use existing CSVs under `data/results/` and `data/recovered/`.
   - Do not depend on cthulu-only paths in the paper pipeline.
   - If an old result is missing locally, mark it as missing and regenerate it.

2. **Regenerate base responses**
   - Use `scripts/generate_responses.py basic` for provider/HF/vLLM models.
   - Use `scripts/generate_responses.py tinker` for Tinker sampler checkpoints.
   - Store base outputs under `data/model-responses/<dataset>/...`.

3. **Run the prompt intervention ladder**
   - Baselines: `just_name_it`, `random_sampling`, `stylistic`.
   - Stronger prompting: `behavioral_based`, `contrastive`, clustering variants.
   - Store comparisons under `data/results/<dataset>/comparisons/...`.

4. **Fine-tune with Tinker/OpenAI**
   - Tinker SFT is supported through `workflows/run_gsm8k_workflow.py`.
   - Tinker DPO is routed through `tinker_cookbook` from `workflows/pipeline.py`.
   - Save reusable sampler paths in `data/tinker_adapters.json`.
   - Tinker LoRA is sufficient for the post-training rung, but not for direct
     activation hooks unless the weights are exported locally.

5. **Generate post-training eval responses**
   - For Tinker, prefer `scripts/generate_responses.py ... tinker --model-path <tinker://...>`.
   - Use the OpenAI-compatible Tinker endpoint only for quick low-throughput checks.
   - Score all generated responses against the target.

6. **Run behavioral-inertia analysis**
   - Use `python -m scripts.analysis.run_behavioral_inertia` on each comparison CSV.
   - Report source persistence, target assimilation, anisotropy, and probe accuracy.

7. **Run activation bridge**
   - Fixed encoder mode enables cross-family comparisons without pretending raw hidden states align.
   - Native-probe mode gives source-model representation evidence when the source model is open.
   - For steering, export Tinker sampler weights to PEFT or a merged HF model,
     then run `scripts.analysis.activation_steering` with Transformers hooks.

8. **Aggregate into the intervention ladder**
   - Use `python -m scripts.analysis.run_intervention_ladder`.
   - Main figure: source persistence decreases slowly, unevenly, or not enough as interventions strengthen.
   - This is the conference framing: behavioral weights are not merely prompt-level labels; they leave residual signatures under stronger interventions.

## Tinker Commands

Dry-run a Tinker SFT config without launching a job:

```bash
python -m workflows.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --dry-run
```

Launch Tinker SFT after setting `TINKER_API_KEY` and confirming inputs:

```bash
python -m workflows.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --epochs 6 \
  --batch-size 16 \
  --weights-name gsm8k_llama-3.1-8b-instruct
```

Dry-run a Tinker DPO config:

```bash
python -m workflows.run_gsm8k_workflow \
  --stage dpo \
  --provider tinker \
  --dry-run
```

Generate responses from a saved Tinker sampler:

```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/results/gsm8k/500/sft_tinker/eval.csv \
  tinker \
  --model-path 'tinker://<run-id>/sampler_weights/<name>' \
  --renderer-name llama3
```

Export a Tinker LoRA adapter for local activation steering:

```bash
python scripts/tools/export_tinker_adapter.py \
  --tinker-path 'tinker://<run-id>/sampler_weights/<name>' \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir data/model-responses/adapters/gsm8k_llama_sft_peft \
  --format peft
```

Run activation steering locally:

```bash
python -m scripts.analysis.activation_steering \
  --comparison-csv data/results/gsm8k/comparisons/disguised_vs_target/contrastive/llama_as_gpt-4.1-mini.csv \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --peft-adapter-path data/model-responses/adapters/gsm8k_llama_sft_peft \
  --output-dir data/results/gsm8k/analysis/activation_steering/llama_as_gpt-4.1-mini \
  --layer -8 \
  --strengths 0,0.5,1,2
```

## Missing Old Results

If old cthulu/BAIR results are unavailable:

1. Prefer imported local files from `data/recovered/` and Naz's call-center results.
2. If the local artifact exists but has unclear provenance, use it only as exploratory evidence.
3. For paper tables, regenerate from scripts with recorded commands.
4. Keep regenerated outputs under `data/results/<dataset>/analysis/...` so the analysis is repeatable.
