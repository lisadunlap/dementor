# Partner setup — Dementor imitation pipeline on a fresh 4×H100 box

This package runs the **imitation** side of Dementor end-to-end on a machine that does **not** share
our `/data` layout. Three evaluations, all keyed off the same disguise-adapter registry:

- **Erosion** (`tinker_erosion.py`, `erosion_daemon.py`, `run_erosion_item.py`, `build_erosion_csv.py`):
  for each disguise adapter *A_as_B* (source model A trained to imitate target B), how much does A's
  safety erode vs A's own unadapted baseline, across 7 safety benchmarks.
- **Fidelity** (`fidelity_daemon.py`, `fidelity_eval.py`): does *A_as_B* actually behave like B on
  held-out dataset prompts (behavioral similarity, embed or LLM-judge scorer).
- **Prompt rung** — disguise-ladder (`prompt_tinker_erosion.py`, `prompt_erosion_daemon.py`,
  `run_prompt_erosion_item.py`, `prompt_erosion_common.py`, `build_prompt_erosion_csv.py`): the
  **prompt-only** analog of erosion. Instead of *training* A to imitate B, A is **prompted** to imitate
  B via one of **5 disguise methods** (`just_name_it`, `random_sampling`, `stylistic`, `behavioral`,
  `contrastive`), measuring the same safety erosion on the same 7 benchmarks. **Its baselines are
  shared byte-for-byte with the erosion (SFT/DPO) rung** (`work/baseline_<slug>/metrics.json`, reused
  never recomputed) so the two rungs are directly comparable. `behavioral`/`contrastive` build their
  disguise with an OpenAI analyzer and need `OPENAI_API_KEY` (§4); the other three are fully local.

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

- Linux, 4× H100 (80 GB), Python **3.11+**, `git`, **`git-lfs`** (the committed benchmark CSVs +
  adapter registries + fidelity subsamples are LFS-tracked — without git-lfs you'll get pointer
  stubs), `rsync`.
- ~200–400 GB free disk for the HF model cache (judge/grader models + base tokenizers).
- A **Tinker API key** with access to our adapters (see §4 — this is account-scoped, not solved by
  your own credits) and an **HF token** (for gated models like Llama-Guard-3).

## 2. Clone + branch

```bash
git lfs install                       # once per machine — REQUIRED before clone/pull
git clone git@github.com:lisadunlap/dementor.git
cd dementor
git checkout ethan
git pull origin ethan
git lfs pull                          # if any data file is still a pointer stub
```

The package lives at `experiments/imitation_safety/`. Verify the LFS data materialized:
`head -1 experiments/imitation_safety/registry/tinker_adapters.json` should be JSON, not
`version https://git-lfs...`.

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

### `OPENAI_API_KEY` (prompt rung only — `behavioral` + `contrastive` disguise methods)

The prompt rung's two **analyzer** disguise methods build their system prompt by calling
**`gpt-4.1-mini`** to distill target B's communication essence (`behavioral`) or contrast B-vs-A
distinctive features (`contrastive`). Those two methods **require `OPENAI_API_KEY`**; the other three
(`just_name_it`, `random_sampling`, `stylistic`) are fully local and need no OpenAI key. Analyzer rules
are **cached** per (method, source, target) under `work_prompt_erosion/_rules_cache/`, so a resume
never re-calls the API and each disguise is reproducible.

```bash
export OPENAI_API_KEY=...    # only for prompt-rung behavioral/contrastive
```

`prompt_erosion_common.py` also reads `OPENAI_API_KEY` (+ `TINKER_API_KEY` / `HF_TOKEN`) from the repo
`.env` via `_load_env`, so a repo-root `.env` works too. (Not needed at all for the erosion / fidelity
evaluations — those are 100% local graders.)

## 5. Environment variables (path portability)

Every root defaults to an in-repo / in-package location. On your box you only need to set a few. All
are optional except where noted.

| Env var | Default | Set on your box? |
|---------|---------|------------------|
| `DEMENTOR_GPUS` | `5,6,7` (our box) | **Yes → `0,1,2,3`** |
| `DEMENTOR_HF_HOME` | `~/.cache/huggingface` | Recommended → a big-disk path on workers (exported as `HF_HOME`) |
| `DEMENTOR_DATA` | `<repo>/data` | Recommended → big-disk path (holds `work/`, `subsamples/`, `results/`) |
| `OPENAI_API_KEY` | — | **Prompt rung only** → set for `behavioral`/`contrastive` disguise methods (gpt-4.1-mini analyzer; §4) |
| `DEMENTOR_DISGUISE_DATA` | `<DEMENTOR_DATA>/model-responses/matrix_baselines/chatbot_arena` | No (prompt rung; follows `DEMENTOR_DATA`) |
| `DEMENTOR_IMITATION_ROOT` | `<DEMENTOR_DATA>/imitation_safety` | Optional (scratch/work root; overrides `DEMENTOR_DATA` for work dirs + GPU-lease dir) |
| `DEMENTOR_RESULTS_SAFETY` | `<DEMENTOR_DATA>/results/safety` | Optional |
| `DEMENTOR_RESULTS_FIDELITY` | `<DEMENTOR_DATA>/results/fidelity` | Optional |
| `DEMENTOR_PORT_DIR` | in-repo `experiments/steering/port` (auto; else our-box fallback) | No — judge modules are committed (§6) |
| `DEMENTOR_JUDGE_ALL` | in-repo `experiments/steering/port/judge_all.py` (auto) | No (committed) |
| `DEMENTOR_RTL_JUDGE_DIR` | in-repo `experiments/steering/port` (auto) | No (committed) |
| `DEMENTOR_STEER_ROOT` | `/data/ethantsliu/exp_steer_safety` (our-box fallback only) | No |
| `DEMENTOR_REGISTRY` | in-package `registry/tinker_adapters.json` | No (committed) |
| `DEMENTOR_BACKUP_REGISTRY` | in-package `registry/tinker_adapters.backup_*.json` | No (committed) |
| `DEMENTOR_BENCH_DIR` | in-package `benchmarks/` | No (committed) |
| `DEMENTOR_FIDELITY_SUBSAMPLES` | in-package `fidelity_subsamples/` | No (committed) |
| `DEMENTOR_PY` | `sys.executable` | No (inherits your venv) |
| `DEMENTOR_GPU_LEASE_ROOT` | `<IMITATION_ROOT>/gpu_leases` | No |

Suggested `env.sh` to source before launching:

```bash
export DEMENTOR_GPUS=0,1,2,3
export DEMENTOR_HF_HOME=/big/disk/hf   # exported as HF_HOME by run_partner.sh
export DEMENTOR_DATA=/big/disk/dementor_imitation
export TINKER_API_KEY=...   # ours
export HF_TOKEN=...
# No steering-helper vars needed — the judge modules are committed in-repo (§6).
```

## 6. Steering judge/grader helpers — COMMITTED IN-REPO (no rsync)

The imitation pipeline **reuses** four steering-side helper modules for judging/grading. These are now
**committed in the repo** at `experiments/steering/port/`, so a fresh clone resolves them
**automatically with NO rsync**. `erosion_common.py` prefers the in-repo copy when present (and falls
back to our-box `/data` paths only on our machine), so you don't set any env var for them.

| Module (in `experiments/steering/port/`) | Used by |
|------------------------------------------|---------|
| `judge_all.py` | RTL harm judge (Qwen3-8B) subprocess |
| `cone_eval.py` | HarmBench + StrongREJECT graders |
| `canonical_graders.py` | SORRY-Bench / SG-Bench / OR-Bench / XSTest graders |
| `rtl_judge.py` | judge model loader (fidelity `judge` scorer) |

**The remote SAMPLE / fidelity-embed steps don't touch these at all; the JUDGE/GRADE step and the
fidelity `judge` scorer import them** — and after a clone they're already on disk. All four live in the
single `port/` dir, which `erosion_common` puts on `sys.path` for you.

> **OPEN ITEM:** if a judge run raises `ModuleNotFoundError` for some deeper steering import, the
> committed `experiments/steering/` tree should already carry it (it ships the full `port/` set); if
> not, add the offending module's dir to `PYTHONPATH`. If you'd rather only *sample* remotely and ship
> the raw generations back to us for judging on our box, you can ignore the judge phase entirely (run
> only the `sample` / `gen-tinker` phases).

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
- All 14 pipeline scripts (9 erosion/fidelity + 5 prompt-rung disguise-ladder).

Also committed in-repo (via the steering-consolidation work): the 4 judge/grader modules at
`experiments/steering/port/` (§6) — resolved automatically, no rsync.

**Not committed — you obtain separately:**
- HF models (§7).
- Model weights: the Tinker sources sample **remotely** (no weights). The ~92 local chatbot_arena PEFT
  adapters are **not** in git and are large — pull from HF only if you want the local track (§10).

## 9. Launch commands

`cd` to the repo root, `source` your `env.sh` (§5) first.

### Erosion — Tinker track (your primary deliverable)

```bash
# A) SAMPLE remotely (no GPU) — writes work/<id>/<bench>/all_gens.csv
python experiments/imitation_safety/tinker_erosion.py sample --seed all --sample-workers 64

# B) JUDGE + grade on your GPUs (batched; uses the in-repo §6 judge modules — already present)
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

### Prompt rung — disguise-ladder (5 disguise methods)

Prompt-only analog of erosion: A is **prompted** to imitate B (no training) via one of 5 methods —
`just_name_it`, `random_sampling`, `stylistic`, `behavioral`, `contrastive`. Same 7 safety benchmarks,
same cached subsamples, same judge/graders as erosion, and — critically — **the same baselines**,
reused from `work/baseline_<slug>/metrics.json` and **never recomputed here**. `behavioral` /
`contrastive` need `OPENAI_API_KEY` (§4); the other three are fully local. It mirrors the erosion rung's
two tracks (remote-sample **Tinker** sources + a polite local-GPU daemon for local-weight sources):

```bash
# Tinker track (gpt-oss / nemotron / qwen sources) — remote SAMPLE, then local batched JUDGE:
python experiments/imitation_safety/prompt_tinker_erosion.py sample --sample-workers 64
python experiments/imitation_safety/prompt_tinker_erosion.py judge  --gpus 0,1,2,3
python experiments/imitation_safety/prompt_tinker_erosion.py status              # progress snapshot
#   restrict methods with e.g.  --methods just_name_it,stylistic

# Local track (local-weight sources; idle-GPU-polite daemon; NO baselines run here — shared):
python experiments/imitation_safety/prompt_erosion_daemon.py --dry-run           # worklist + GPU snapshot
python experiments/imitation_safety/prompt_erosion_daemon.py

# Aggregate to the prompt-rung CSVs (prompt_erosion_all_{long,summary}.csv):
python experiments/imitation_safety/build_prompt_erosion_csv.py
```

> **Baselines are shared with the erosion (SFT/DPO) rung.** The prompt rung reuses
> `work/baseline_<slug>/metrics.json` (model A unadapted) rather than recomputing it, so those
> baselines must exist first (run the erosion track, or its `erosion_daemon` baselines). Until they do,
> `build_prompt_erosion_csv.py` reports the affected sources as `baseline_available=False` and their
> erosion stays NaN — the samples/judgments are still valid and fill in once the baseline lands.

### Local track (only if you pulled the 92 PEFT adapters — §10)

```bash
python experiments/imitation_safety/erosion_daemon.py --dry-run    # will show 0 local items until §10 is done
python experiments/imitation_safety/erosion_daemon.py --seed seed42
# Stage-restricted DPO repair, generating safety + held-out fidelity in one model load:
python experiments/imitation_safety/erosion_daemon.py --seed seed42 --stage dpo \
  --generate-only --also-fidelity
```

Local DPO registry rows must retain `sft_parent`. Inference reconstructs the effective
`base + SFT LoRA + DPO LoRA` weights (the DPO directory alone is not a complete model).

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
