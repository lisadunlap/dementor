# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

Dementor measures **behavioral-fingerprint persistence** in LLMs under disguise: techniques that
make one model imitate another (naming → few-shot/style prompting → SFT → DPO → activation
steering), plus analyses of which behavioral axes survive each intervention. The canonical metric
definitions live in `METHODS.md` — treat it as the source of truth for the
persistence metric, manifest shape, and evaluator usage.

## Repository layout

Installable package (`pip install -e .`). Three top-level trees:

- **`dementor/`** — the importable library:
  - `config.py` — loads `config.yaml` (the single source of truth; see below).
  - `methods/` — disguise methods (registered in `get_method.py`).
  - `metric/` — persistence metric + cell evaluator (`behavioral_cell_evaluator`, `run_cell_pipeline`,
    `run_matrix_cells`, `latent_behavior_axes`, `behavioral_inertia_metrics`, `run_intervention_ladder`, …).
  - `steering/` — activation steering + fixed-encoder probes (`activation_steering`, `activation_bridge`).
  - `training/` — Tinker + OpenAI + **local-GPU** backends, the experiment-matrix driver (`matrix.py`),
    and the dry-run planner (`plan.py`).
  - `serve/`, `safety/`, `data_utils/` — serving utils, refusal evaluation, dataset/prompt builders.
  - `cli/` plus top-level `scorer.py` / `cache_llm.py` / `visualize_scores.py` — entry points.
- **`experiments/`** — one-off analysis / figure / tool scripts (`analysis/`, `figures/`, `tools/`,
  `gsm8k/`, `safety/`). NOT imported by the library; safe to run, edit, or delete in isolation.
- **`tests/`** — pytest suite (CPU-only). CI: `.github/workflows/ci.yml`.

## Configuration — `config.yaml` is the single source of truth

`dementor.config` reads `config.yaml`; **nothing else hardcodes** the roster, seeds, datasets, or
hyperparameters. `matrix.py` and `plan.py` derive the entire experiment from it.

- **Roster:** the catalog contains active, extended, and legacy models. The named
  `campaigns.imitation_safety` selector chooses the publication core: 12 models, four datasets,
  seed 42, SFT/DPO/self-SFT stages, 200 evaluation prompts, and a 256-new-token cap. Sources split
  across Tinker and local-GPU backends.
  `roster_legacy` holds retired models so existing adapters still resolve slug↔id.
- **Accessors:** `config.roster()`, `config.campaign_roster()`, `config.campaign_dataset_names()`,
  `config.campaign_seeds()`, `config.campaign_stages()`, `config.campaign_evaluation()`,
  `config.model(slug_or_id)`, `config.backend_for(id)`,
  `config.dataset(name)`, `config.seeds()`, `config.sft()/dpo()/lora()`, `config.registry_path()`,
  `config.project_root()`. Paths anchor on the repo root, never the cwd.
- To change the experiment, edit `config.yaml` — do not reintroduce hardcoded model/seed lists.

## Disguise methods (`dementor/methods/`)

Registered in `get_method.py`; all inherit `MethodBase` and expose `forward(prompt: str)`:
`just_name_it`, `random_sampling`, `behavioral`, `stylistic`, `contrastive`,
`stylistic_clustering`, `stylistic_clustering_resample`, `embedding_clustering`, `behavioral_clustering`.

## Evaluation pipeline

Headline metric is `persistence` (k-independent full-feature source→target projection); `anchored`
rescales it by a source self-baseline and an identity control. Flow:

1. base responses → `data/model-responses/`
2. disguise → `data/results/<dataset>/comparisons/…`
3. score → `dementor.scorer`
4. multi-method cell → `dementor.metric.behavioral_cell_evaluator` (one shared basis per source→target cell)
5. aggregate → `dementor.metric.run_intervention_ladder`

`dementor.metric.run_cell_pipeline` runs 1–4 for one `(dataset, source, target)` cell;
`run_matrix_cells` batches it over a matrix. The "full"/"adjectives" feature path needs
`sentence-transformers` (MiniLM scoring before the supervised source/target basis is fit — disguised
outputs are *projected* into that basis, never used to fit it).

## Training matrix & backends (`dementor/training/`)

The active matrix is config-driven: `config.campaign_roster()` × campaign datasets × campaign seeds
× {SFT, DPO} + self-SFT controls. The publication design has 528 cross-model cells per SFT/DPO
stage. Sources route to the Tinker or local backend by `config.backend_for`.

- **Dry-run the full plan (no spend):** `dementor-plan` (a.k.a. `python -m dementor.training.plan`) —
  enumerates every job with its backend and prints counts; launches nothing.
- **Run (gated — costs money / needs a GPU):** `dementor-matrix {generate-target-responses,
  build-sft-data, launch-sft, build-dpo-data, launch-dpo, …}`; all support `--dry-run`.
- Adapters are recorded in `data/tinker_adapters.json` (Tinker `tinker://` URIs or local PEFT dirs,
  tagged with `backend`). Registry-touching ops resolve the source model from the alias, so they
  cover both current and legacy adapters.

Local SFT/DPO are exercised on CPU in `tests/test_local_backend.py`; actual training is run later on
a GPU. Do **not** launch Tinker/GPU jobs without explicit confirmation.

## Install & dependencies

**First, materialize the LFS data** (`*.csv/json/png/pdf/svg/html` are git-LFS; a fresh clone has only
~130-byte pointer stubs and the test suite + every analysis will fail on them):

```bash
git lfs install && git lfs pull
pip install -e .            # or: uv pip install -e .   (registers the dementor-* console scripts)
```

No `PYTHONPATH` prefix is needed. Core install covers analysis + the metric. Optional extras:

- `.[train]` — `peft, trl, accelerate, datasets, tinker` (local + Tinker training).
- `.[serve]` — `vllm, gradio` (local serving + the viewer UI).
- `.[dev]`   — `pytest`.

`tinker` / `tinker_cookbook` are not on PyPI (install from Thinking Machines); a `TINKER_API_KEY`
(in `.env`) is required for Tinker runs. Every remote/training import is lazy, so analysis works
without them.

Console scripts: `dementor-matrix`, `dementor-plan`, `dementor-generate`, `dementor-disguise`,
`dementor-viewer`, `dementor-score`, `dementor-cache`, `dementor-visualize`, `dementor-make-prompts`,
`dementor-make-splits`.

## Common commands

```bash
# Generate base responses
dementor-generate --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --output-csv data/model-responses/gsm8k/full/openai_gpt-4.1-mini.csv basic --model openai/gpt-4.1-mini

# Apply a disguise method
dementor-disguise --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv \
  --model meta-llama/Llama-3.1-8B-Instruct --disguise-as openai/gpt-4.1-mini --method contrastive --num-samples 50

# Score disguised-vs-target
python -m dementor.scorer pairwise --input <comparison.csv> --output <scored.csv>

# Plan / dry-run the training matrix (no spend)
dementor-plan
python -m dementor.training.matrix list-cells

# Activation steering (export the Tinker LoRA to a local PEFT adapter first)
python -m experiments.tools.export_tinker_adapter --tinker-path 'tinker://<run>/sampler_weights/<name>' \
  --base-model meta-llama/Llama-3.1-8B-Instruct --output-dir <peft_dir> --format peft
python -m dementor.steering.activation_steering --comparison-csv <comparison.csv> \
  --model-name meta-llama/Llama-3.1-8B-Instruct --peft-adapter-path <peft_dir> \
  --output-dir <out> --layer -8 --strengths 0,0.5,1,2
```

## Testing

```bash
pytest -q                              # CPU-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q   # skip the cached-MiniLM network check
```

The local-backend SFT/DPO tests download a tiny random model (need network). CI runs `pytest` on push.

## Result organization

- `data/model-responses/<dataset>/` — base-model outputs.
- `data/results/<dataset>/comparisons/` — disguise comparisons.
- `data/results/<dataset>/analysis/` — behavioral-inertia / bridge / steering / ladder outputs (gitignored).
- `data/results/safety/` — audited campaign coverage, stage-separated aggregates, steering manifests,
  and publication figures.
- `data/tinker_adapters.json` — adapter registry (Tinker URIs or local PEFT dirs).

## Agent guidelines

- Run commands directly; don't ask the user to run them.
- `config.yaml` is the single source of truth — extend the roster/datasets/seeds/hparams there, not in code.
- Keep remote/training imports (`tinker`, `tinker_cookbook`, `peft`, `trl`, `vllm`, `gradio`, `lmdb`)
  lazy, inside the paths that need them, so analysis stays importable without every backend.
- Never launch Tinker/GPU training (cost) without explicit confirmation; prefer `--dry-run` / `dementor-plan`.
- Keep generated results under `data/results/` and base responses under `data/model-responses/`.
- Use focused CPU tests for metric/math/CLI behavior; avoid tests that need remote APIs or large downloads.
- Update this file, `README.md`, and `docs/` when command args, the data layout, or the config schema change.
