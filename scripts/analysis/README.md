# Behavioral Inertia Analysis

This package measures how much source-model behavior remains after a disguise
intervention tries to make the source imitate a target.

The latent analysis uses Naz's adjective-matching setup as the canonical
feature representation: Big-Five + model-style descriptors are embedded with
`sentence-transformers/all-MiniLM-L6-v2`, each response is scored by descriptor
cosine similarity, and the resulting descriptor matrix is combined with
deterministic style scalars/binary heuristics. The source and target outputs
fit the scaler and SVD basis; disguised outputs are projected into that fixed
basis. The older TF-IDF approximation is intentionally not used for paper runs.

## End-to-end latent analysis

```bash
python3 -m scripts.analysis.run_behavioral_inertia \
  --comparison-csv data/results/<dataset>/.../<source>_as_<target>.csv \
  --source-responses data/model-responses/<dataset>/full/<source>.csv \
  --source-model <source> \
  --target-model <target> \
  --method <method> \
  --dataset <dataset> \
  --output-dir data/results/<dataset>/analysis/<method>/<source>_as_<target>
```

The comparison CSV should contain `prompt`, `model_response`, and
`target_response`. If `source_response` is absent, pass `--source-responses` so
the source outputs can be joined by prompt.

Outputs include `latent_scores.csv`, `axis_loadings.csv`,
`per_axis_movement.csv`, `bootstrap_summary.csv`, `summary.json`, and
paper-style figures. `per_axis_movement.csv` marks low-separation inactive axes
so headline persistence is computed only on axes where source and target are
meaningfully separated.

Save a reusable source/target basis for a model pair:

```bash
python3 -m scripts.analysis.run_behavioral_inertia \
  --comparison-csv data/results/<dataset>/.../<source>_as_<target>.csv \
  --source-responses data/model-responses/<dataset>/full/<source>.csv \
  --output-dir data/results/<dataset>/analysis/<method>/<source>_as_<target> \
  --save-basis data/results/<dataset>/analysis/bases/<source>_to_<target>.pkl
```

Use the same basis for another intervention on that source/target pair:

```bash
python3 -m scripts.analysis.run_behavioral_inertia \
  --comparison-csv data/results/<dataset>/.../<source>_as_<target>_sft.csv \
  --source-responses data/model-responses/<dataset>/full/<source>.csv \
  --output-dir data/results/<dataset>/analysis/sft/<source>_as_<target> \
  --load-basis data/results/<dataset>/analysis/bases/<source>_to_<target>.pkl
```

Run feature ablations with `--feature-set`:

```bash
--feature-set full            # adjective descriptors + style scalars + binary style heuristics
--feature-set adjectives      # adjective descriptors only
--feature-set style_scalars   # numeric length/format/style scalars only
--feature-set style_binaries  # binary surface-style heuristics only
--feature-set style_all       # style scalars + binary style heuristics
```

## Activation bridge

Use fixed-encoder mode for cross-model-family comparisons:

```bash
python3 -m scripts.analysis.activation_bridge \
  --comparison-csv <comparison.csv> \
  --source-responses <source.csv> \
  --mode fixed-encoder \
  --encoder-model intfloat/e5-small-v2 \
  --output-dir <analysis_dir>/activations
```

Use native-probe mode when the source model is open and you want scalar probe
evidence inside the source model's representation space:

```bash
python3 -m scripts.analysis.activation_bridge \
  --comparison-csv <comparison.csv> \
  --source-responses <source.csv> \
  --mode native-probe \
  --source-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output-dir <analysis_dir>/native_probe
```

Raw hidden states from different model families are not compared directly. The
bridge reports comparable scalar probe metrics such as source probability for
disguised outputs.

## Activation steering

Activation steering needs local hidden-state hooks during generation. Tinker
remote `SamplingClient` inference does not expose those hooks directly, so use
Tinker for LoRA training and sampler export, then run steering locally with
Transformers.

Export a Tinker sampler checkpoint to a local PEFT adapter:

```bash
python scripts/tools/export_tinker_adapter.py \
  --tinker-path 'tinker://<run-id>/sampler_weights/<name>' \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir data/model-responses/adapters/gsm8k_llama_sft_peft \
  --format peft
```

Run steering on the base model or exported adapter:

```bash
python3 -m scripts.analysis.activation_steering \
  --comparison-csv data/results/<dataset>/.../<source>_as_<target>.csv \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --peft-adapter-path data/model-responses/adapters/gsm8k_llama_sft_peft \
  --output-dir data/results/<dataset>/analysis/activation_steering/<source>_as_<target> \
  --layer -8 \
  --strengths 0,0.5,1,2 \
  --max-rows 200
```

The runner computes a native source-to-target vector from paired
`source_response` and `target_response` activations, injects it during
generation, and writes one comparison CSV per steering strength. Feed those CSVs
back into `run_behavioral_inertia.py` so steering becomes another rung on the
same intervention ladder.

## Intervention ladder

Aggregate multiple `summary.json` files:

```bash
python3 -m scripts.analysis.run_intervention_ladder \
  --summaries path/to/run1/summary.json path/to/run2/summary.json \
  --output-dir data/results/<dataset>/analysis/intervention_ladder
```
