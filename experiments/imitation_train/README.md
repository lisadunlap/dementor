# Imitation-square training sequencer

A detached, restartable daemon that trains the **12×12 imitation "square"**: for every ordered pair
of the 12 core models (`config.yaml` roster, `imitation: core`), it trains a disguise adapter
`{sft,dpo}_chatbot_arena_<source>_as_<target>_seed42` — source model A fine-tuned to imitate target
model B. It fills only the cells still missing from the adapter registry, so it is fully resumable.

The safety/fidelity **evaluation** of the resulting adapters lives in the sibling package
`experiments/imitation_safety/` (see its `PARTNER_SETUP.md`). This package only *produces* the
adapters.

## What it does

Two workers run concurrently, both reading/writing cell status through one locked shared-state dict
(persisted to `queue_state.json`, so a restart resumes cleanly):

- **Tinker-source cells** are submitted remotely via `dementor.training.matrix` (`launch_sft` /
  `launch_dpo`) with bounded concurrency — no local GPU.
- **Local-source cells** run one at a time per free GPU via `dementor-matrix launch-local-cell` in a
  `CUDA_VISIBLE_DEVICES`-pinned subprocess. The local-model baselines each cell depends on are
  generated first, opportunistically.

GPU scheduling is deliberately polite on our shared box: GPU4 is prohibited, GPUs 0–3 are treated as
off-limits while any foreign user is present (the "js_park block"), and the daemon shares an atomic
GPU lease (`../imitation_safety/gpu_lease.py`) with the erosion daemons and yields transient frees to
the steering roster. On a dedicated box these politeness gates are disabled via env (see below).

## Modules

| File | Responsibility |
|------|----------------|
| `sequencer.py` | Entrypoint: wires the modules together, runs both worker threads, top-level restart loop. |
| `runtime.py` | Portable, env-driven config + logging (imported first by everyone). |
| `worklist.py` | Registry-driven worklist + the single locked shared-state dict. |
| `gpus.py` | GPU usability polling + js_park block policy + shared-lease arbitration. |
| `workers.py` | The tinker + local worker loops and their subprocess launchers. |
| `watch.py` | Low-noise progress watcher: prints one line per meaningful, not-yet-reported transition. |
| `download_weights.py` | One-shot helper to pre-download the small local student weights into the HF cache. |

## Running it

```bash
# Detached (outlives the launching session); logs to sequencer.log:
cd experiments/imitation_train
setsid nohup "$DEMENTOR_PY" sequencer.py >> "$DEMENTOR_TRAIN_STATE/sequencer.log" 2>&1 &

# Watch progress (silent unless a meaningful event fires):
"$DEMENTOR_PY" watch.py
```

## Environment variables

All roots default to our-box values and are overridable; the canonical `DEMENTOR_*` names match the
eval/steering halves. Because a fresh clone with defaults gets its own state dir, it never clobbers a
running campaign.

| Env var | Default | Purpose |
|---------|---------|---------|
| `DEMENTOR_REPO` | repo root (two levels up) | Repo checkout to run from. |
| `DEMENTOR_PY` | `sys.executable` | Interpreter for training subprocesses. |
| `DEMENTOR_DATA` | `<repo>/data` | Big-disk data/outputs root (holds the registry + baselines). |
| `DEMENTOR_HF_HOME` | `~/.cache/huggingface` | HF cache; exported as `HF_HOME`. |
| `DEMENTOR_TRAIN_STATE` | `<DEMENTOR_DATA>/imitation_train` | State dir (`queue_state.json`, `local_logs/`, `sequencer.log`). Point at an existing campaign dir to **resume** it. |
| `DEMENTOR_GPUS` | `5,6,7` | Cards this daemon may lease-arbitrate (partner dedicated box: `0,1,2,3`). |
| `DEMENTOR_FORBIDDEN_GPUS` | `4` | Compute-prohibited cards; set empty on a box with none. |
| `DEMENTOR_BLOCK_GPUS` | `0,1,2,3` | js_park's off-limits block; set empty on a dedicated box. |
| `SEED` | `42` | Square seed; the wrapper sets `43`/`44` for the robustness expansion. |
| `SEQ_COEXIST` | `1` | Coexistence mode (sustained-idle gate + GPU lease). Set `0` for immediate-grab, no lease. |
| `SEQ_SUSTAINED_POLLS` | `3` | Consecutive usable polls before grabbing a lease-pool card. |
| `IMIT_LOCAL_ONLY` | `0` | Restrict the queue to local-source cells (keeps the tinker worker idle). |
