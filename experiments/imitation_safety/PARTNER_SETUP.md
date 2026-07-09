# Partner setup — Dementor imitation pipeline on a fresh 4×H100 box

This package runs the **imitation** side of Dementor end-to-end on a machine that does **not** share
our `/data` layout. Two evaluations, both keyed off the same disguise-adapter registry:

- **Erosion** (`tinker_erosion.py`, `erosion_daemon.py`, `run_erosion_item.py`, `build_erosion_csv.py`):
  for each disguise adapter *A_as_B* (source model A trained to imitate target B), how much does A's
  safety erode vs A's own unadapted baseline, across 7 safety benchmarks.
- **Fidelity** (`fidelity_daemon.py`, `fidelity_eval.py`): does *A_as_B* actually behave like B on
  held-out dataset prompts (behavioral similarity, embed or LLM-judge scorer).

There are **two generation tracks**:

| Track | Adapters | Weights | GPU for generation? |
|------|---------|---------|---------------------|
| **Tinker** (what you run) | **572** disguise adapters across 4 datasets (chatbot_arena 107, gsm8k 161, oasst1 143, writingprompts 161) + 8 base baselines / 28 target refs | `tinker://` sampler URIs — sampled **remotely** | No (remote). GPU only for the local **judge/grade** step. |
| **Local** (stays on our box) | ~92 chatbot_arena seed42 PEFT adapters + 11 local refs | local PEFT weights on our `/data` | Yes. Skipped automatically on your box (no weights) unless you pull them from HF — see §10. |

The registries are committed in-package, so `build_worklist` / `tinker_worklist` resolve the full
572-adapter Tinker worklist out of the box; the ~92 local cells auto-drop on your box because their
PEFT dirs don't exist there.

---

## 1. Prerequisites

- Linux, 4× H100 (80 GB), Python **3.11+**, `git`, `rsync`.
- ~200–400 GB free disk for the HF model cache (judge/grader models + base tokenizers).
- A **Tinker API key** with access to our adapters (see §4 — this is account-scoped, not solved by
  your own credits) and an **HF token** (for gated models like Llama-Guard-3).

## 2. Clone + branch

```bash
git clone git@github.com:lisadunlap/dementor.git
cd dementor
git checkout ethan
git pull origin ethan
```

The package lives at `experiments/imitation_safety/`.

## 3. Python environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"       # base deps + train extra: tinker, peft, accelerate, datasets, trl
```

The imitation pipeline uses: `transformers`, `torch`, `peft`, `tinker`, `sentence-transformers`,
`pandas`, `numpy` — all pulled by the command above.

## 4. Credentials — read this carefully

### `TINKER_API_KEY` (required for the Tinker track — **account-scoped**)

The `tinker://` adapters we trained are **scoped to our Tinker account/organization**. Your own
Tinker credits do **not** grant access to our adapter URIs (you'll get `PermissionDeniedError`). To
sample the 572 Tinker-backed adapters you need **one** of:

1. **Use our `TINKER_API_KEY`** (recommended, simplest). You sample remotely with our key; it's cheap
   inference on our credits. *(Confirmed you already have our keys.)*
2. **Be added to our Tinker organization/project**, so your own key/credits work on the shared
   adapters — depends on whether Naz can manage org membership.
3. **Download + merge each adapter to local HF weights** (via
   `tinker_cookbook/scripts/merge_tinker_adapter_to_hf_model.py`) for full Tinker-independence — but
   this *still* needs our key to read the adapters, plus the base models locally (incl. gpt-oss-120b /
   nemotron-super-120b, 120 B, heavy). Only if you want to cut Tinker out entirely.

Default: **option 1.** Set it in your shell or in the repo `.env`:

```bash
export TINKER_API_KEY=...      # ours
export HF_TOKEN=...            # yours or ours; needed for gated judge/base models
```

`tinker_erosion.py` / `fidelity_common.py` also read `TINKER_API_KEY` + `HF_TOKEN` from the repo
`.env` if present (`_load_env`), so a repo-root `.env` works too.

### `HF_TOKEN`

Needed for gated HF repos — notably **`meta-llama/Llama-Guard-3-8B`** (SG-Bench grader) and some base
tokenizers/weights.

## 5. Environment variables (path portability)

Every root defaults to an in-repo / in-package location. On your box you only need to set a few. All
are optional except where noted.

| Env var | Default | Set on your box? |
|---------|---------|------------------|
| `DEMENTOR_GPUS` | `5,6,7` (our box) | **Yes → `0,1,2,3`** |
| `HF_HOME` | `~/.cache/huggingface` | Recommended → a big-disk path |
| `DEMENTOR_DATA_ROOT` | `<repo>/data` | Recommended → big-disk path (holds `work/`, `subsamples/`, `results/`) |
| `DEMENTOR_IMITATION_ROOT` | `<DATA_ROOT>/imitation_safety` | Optional (scratch/work root; overrides `DATA_ROOT` for work dirs + GPU-lease dir) |
| `DEMENTOR_RESULTS_SAFETY` | `<DATA_ROOT>/results/safety` | Optional |
| `DEMENTOR_RESULTS_FIDELITY` | `<DATA_ROOT>/results/fidelity` | Optional |
| `DEMENTOR_STEER_ROOT` | `/data/ethantsliu/exp_steer_safety` | **Yes if judging** → rsync target (see §6) |
| `DEMENTOR_PORT_DIR` | `<STEER_ROOT>/repl80_rdo/port` | Yes if judging (or inherits STEER_ROOT) |
| `DEMENTOR_JUDGE_ALL` | `<STEER_ROOT>/judge_all.py` | Yes if judging (or inherits STEER_ROOT) |
| `DEMENTOR_RTL_JUDGE_DIR` | `/data/ethantsliu/exp3_safety/leak_fix` | **Yes if judging** → rsync target |
| `DEMENTOR_REGISTRY` | in-package `registry/tinker_adapters.json` | No (committed) |
| `DEMENTOR_BACKUP_REGISTRY` | in-package `registry/tinker_adapters.backup_*.json` | No (committed) |
| `DEMENTOR_BENCH_DIR` | in-package `benchmarks/` | No (committed) |
| `DEMENTOR_FIDELITY_SUBSAMPLES` | in-package `fidelity_subsamples/` | No (committed) |
| `DEMENTOR_PY` | `sys.executable` | No (inherits your venv) |
| `DEMENTOR_GPU_LEASE_ROOT` | `<IMITATION_ROOT>/gpu_leases` | No |

Suggested `env.sh` to source before launching:

```bash
export DEMENTOR_GPUS=0,1,2,3
export HF_HOME=/big/disk/hf
export DEMENTOR_DATA_ROOT=/big/disk/dementor_imitation
export DEMENTOR_STEER_ROOT=/big/disk/steer_helpers          # where you rsync'd §6
export DEMENTOR_RTL_JUDGE_DIR=/big/disk/steer_helpers/rtl_judge
export TINKER_API_KEY=...   # ours
export HF_TOKEN=...
```

## 6. Steering judge/grader helpers — NOT in git (required for the JUDGE/GRADE phase)

The imitation pipeline **reuses** four steering-side helper modules for judging/grading. They are
**not** part of this package (they live in the steering experiment tree and were deliberately left
untouched). **The remote SAMPLE / fidelity-embed steps do NOT need them; the JUDGE/GRADE step and the
fidelity `judge` scorer DO.** They must be rsync'd from our box:

| Module | On our box | Points via env |
|--------|-----------|----------------|
| `judge_all.py` (RTL harm judge, Qwen3-8B) | `/data/ethantsliu/exp_steer_safety/judge_all.py` | `DEMENTOR_JUDGE_ALL` / `DEMENTOR_STEER_ROOT` |
| `cone_eval.py` (HarmBench + StrongREJECT graders) | `/data/ethantsliu/exp_steer_safety/repl80_rdo/port/cone_eval.py` | `DEMENTOR_PORT_DIR` / `DEMENTOR_STEER_ROOT` |
| `canonical_graders.py` (SORRY-Bench / SG-Bench / OR-Bench / XSTest graders) | `/data/ethantsliu/exp_steer_safety/repl80_rdo/port/canonical_graders.py` | `DEMENTOR_PORT_DIR` / `DEMENTOR_STEER_ROOT` |
| `rtl_judge.py` (judge model loader, used by fidelity `judge` scorer) | `/data/ethantsliu/exp3_safety/leak_fix/rtl_judge.py` | `DEMENTOR_RTL_JUDGE_DIR` |

Example rsync (adjust hosts/paths):

```bash
rsync -av OURBOX:/data/ethantsliu/exp_steer_safety/    /big/disk/steer_helpers/
rsync -av OURBOX:/data/ethantsliu/exp3_safety/leak_fix/ /big/disk/steer_helpers/rtl_judge/
# then set DEMENTOR_STEER_ROOT=/big/disk/steer_helpers  DEMENTOR_RTL_JUDGE_DIR=/big/disk/steer_helpers/rtl_judge
```

> **OPEN ITEM:** these steering helpers themselves may pull further steering-tree imports at judge time.
> If a judge run raises `ModuleNotFoundError`, rsync the missing steering module and add its dir to
> `PYTHONPATH`. If you only want to *sample* remotely and ship the raw generations back to us for
> judging on our box, you can skip §6 entirely (run only the `sample` / `gen-tinker` phases).

## 7. HF models to pre-download

Set `HF_HUB_ENABLE_HF_TRANSFER=1` and pre-fetch with `huggingface-cli download` (or let them fetch on
first use, but the daemons default to `HF_HUB_OFFLINE=1` for cached-model runs — export
`HF_HUB_OFFLINE=0` for the first fetch).

**Tinker SAMPLING — tokenizer only (weights sampled remotely).** These 8 bases are both the disguise
sources and the target references:

```
Qwen/Qwen3-8B
Qwen/Qwen3.5-4B
Qwen/Qwen3.6-27B
Qwen/Qwen3.6-35B-A3B
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
openai/gpt-oss-20b
openai/gpt-oss-120b
```

Only the tokenizer is loaded locally (`AutoTokenizer.from_pretrained`) — you do **not** need the full
weights of these to sample. (For gpt-oss the full `tokenizer.json` lives under `HF_HOME/hub`; that's
why the sample phase points at `HF_HOME/hub`.)

**JUDGE / GRADE — full weights (local GPU).** These download on first use of the judge phase:

```
Qwen/Qwen3-8B                       # RTL harm judge + OR-Bench/XSTest 3-way classifiers + SORRY-Bench RTL fallback
cais/HarmBench-Llama-2-13b-cls      # HarmBench grader (13B)
meta-llama/Llama-Guard-3-8B         # SG-Bench grader (GATED → needs HF_TOKEN)
<StrongREJECT ft gemma-2b>          # StrongREJECT grader (id defined inside cone_eval; ungated mirror)
```

**FIDELITY embed scorer (CPU, default):** `sentence-transformers/all-MiniLM-L6-v2` (tiny). The
fidelity `judge` scorer reuses the Qwen3-8B judge above.

## 8. What's committed vs. what you download

**Committed in this package (no download):**
- 7 benchmark CSVs → `benchmarks/` (advbench, harmbench, strongreject, xstest, sorrybench, orbench_hard, sgbench)
- Adapter registries → `registry/tinker_adapters.json` (live snapshot: chatbot_arena cells) +
  `registry/tinker_adapters.backup_20260707T015844.json` (gsm8k/oasst1/writingprompts cells)
- Fidelity held-out subsamples (n=200, seed=42) + manifests → `fidelity_subsamples/`
- All 9 pipeline scripts.

**Not committed — you obtain separately:**
- HF models (§7).
- Steering judge/grader helpers (§6) — rsync from our box.
- Model weights: the Tinker sources sample **remotely** (no weights). The ~92 local chatbot_arena PEFT
  adapters are **not** in git and are large — pull from HF only if you want the local track (§10).

## 9. Launch commands

`cd` to the repo root, `source` your `env.sh` (§5) first.

### Erosion — Tinker track (your primary deliverable)

```bash
# A) SAMPLE remotely (no GPU) — writes work/<id>/<bench>/all_gens.csv
python experiments/imitation_safety/tinker_erosion.py sample --seed all --sample-workers 64

# B) JUDGE + grade on your GPUs (batched; needs §6 steering helpers)
python experiments/imitation_safety/tinker_erosion.py judge --seed all --gpus 0,1,2,3

# ...or run both back-to-back:
python experiments/imitation_safety/tinker_erosion.py all --seed all --gpus 0,1,2,3

# progress / worklist snapshot:
python experiments/imitation_safety/tinker_erosion.py status

# C) aggregate to the erosion CSVs:
python experiments/imitation_safety/build_erosion_csv.py --seed all
```

### Fidelity (subordinate; safe to run alongside)

```bash
python experiments/imitation_safety/fidelity_eval.py prep          # verify held-out subsamples/manifests
# remote sampling + CPU embed scoring only (never touches a GPU — safest):
python experiments/imitation_safety/fidelity_daemon.py --no-local --scorer embed
python experiments/imitation_safety/fidelity_eval.py build-csv --scorer embed --seed all
```

The `--scorer judge` variant needs a GPU + `rtl_judge.py` (§6). Run `fidelity_eval.py status` for
counts.

### Local track (only if you pulled the 92 PEFT adapters — §10)

```bash
python experiments/imitation_safety/erosion_daemon.py --dry-run    # will show 0 local items until §10 is done
python experiments/imitation_safety/erosion_daemon.py --seed seed42
```

> **GPU scheduling note.** The daemons were built for our *shared* box: they only claim a card that's
> been genuinely idle for N polls, arbitrated by an atomic GPU lease. On your *dedicated* box that's
> harmless but conservative — lower `--sustained-polls` (e.g. `--sustained-polls 1`) and `--interval`
> to grab cards faster. `tinker_erosion.py judge --gpu N` pins a single card and skips the idle-wait.

## 10. Optional — the ~92 local chatbot_arena seed42 adapters from HF

These are being uploaded to HF org **`dementor-research`**, named
`{sft,dpo}_chatbot_arena_<src>_as_<tgt>_seed42`.

- **Pull ONLY the repos listed in the seed42 manifest** (a precise ~92-repo-id manifest is produced
  separately — ask for `chatbot_arena_seed42` manifest). The `dementor-research` org **also contains
  ~398 STALE repos** (older seed1/2/3 + oasst runs) that are **not** the current set — do **not** pull
  everything under the org.
- To actually *run* them on the local track, the merged registry's local entries (`path`/
  `checkpoint_path`) must point at where you materialized the downloaded adapters (or use the registry
  variant shipped with the manifest). Until then `erosion_daemon.py` correctly reports 0 local items.
- If in doubt, **skip the local track** — the 572 Tinker cells across all 4 datasets are the headline
  result and run without any local weights.

## 11. Results — what to merge back

Per-item checkpoints live under `DEMENTOR_IMITATION_ROOT` (`work/<id>/metrics.json`,
`work_fidelity/adapters/<id>/fidelity_*.json`); the aggregated tables are what we merge:

| File | Path (default) |
|------|----------------|
| Erosion long (one row per adapter×benchmark) | `<DATA_ROOT>/results/safety/erosion_all_long.csv` |
| Erosion summary (one row per adapter) | `<DATA_ROOT>/results/safety/erosion_all_summary.csv` |
| Fidelity long | `<DATA_ROOT>/results/fidelity/fidelity_all_embed_long.csv` |
| Fidelity summary | `<DATA_ROOT>/results/fidelity/fidelity_all_embed_summary.csv` |

The result schema is **identical** to our local track (Tinker rows carry `backend=tinker`), so we
concatenate your CSVs with ours directly — `build_erosion_csv.py` / `fidelity_eval.py build-csv` merge
any per-item `metrics.json` present. Rsync back either the aggregated CSVs, or the whole
`work/` + `work_fidelity/` trees if we want to re-aggregate on our side.
