# Safety-Constrained Imitation Scaffold

This is the mitigation arm for the safety-drift result: train the same source-to-target imitation
cells, but add a small refusal replay set so the adapter is explicitly optimized to retain refusal
behavior on harmful requests.

## What It Builds

- `safety_sft`: target-imitation SFT rows plus `safety_replay` rows.
- `safety_dpo`: target-vs-source imitation preference pairs plus `safety_replay` preference pairs.
- Each CSV has a sibling `.manifest.json` recording row counts, prompt files, excluded eval prompts,
  and whether replay completions came from cached native refusal snippets or the generic refusal
  fallback.

By default, replay prompts come from `data/datasets/benchmarks/refusal/advbench_harmful.csv`, and
prompts in `data/datasets/benchmarks/refusal/eval_120h_80o_seed42.csv` are excluded before sampling.
This avoids training on the current refusal-ladder eval prompts.

## Commands

Free data-builder dry runs:

```bash
python -m dementor.training.matrix build-safety-sft-data --dry-run --max-cells 1
python -m dementor.training.matrix build-safety-dpo-data --dry-run --max-cells 1
```

Free launch dry runs:

```bash
python -m dementor.training.matrix launch-safety-sft --dry-run --max-cells 1
python -m dementor.training.matrix launch-safety-dpo --dry-run --max-cells 1
```

Real launches use the normal matrix training backends and must not be started without explicit
approval. `safety_dpo` expects the corresponding `safety_sft_*` adapter to exist in
`data/tinker_adapters.json`.

## Caveats

The current native safety ladder intentionally discards full raw completions. When a replay prompt
matches a cached native refusal verdict, the builder can use `snippet_redacted`; otherwise it uses
`I can't help with that request.` as the safe completion and records that fallback in the manifest.
The DPO safety replay rejected side is a generic non-refusal stub, not harmful procedural content.
