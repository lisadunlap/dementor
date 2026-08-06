#!/usr/bin/env python3
"""Portable, env-driven config + logging shared by every sequencer module.

All roots are ENV-OVERRIDABLE with our-box values as defaults (see PARTNER_SETUP.md). The canonical
shared env vars (identical names on the eval/steering halves) are DEMENTOR_REPO, DEMENTOR_DATA,
DEMENTOR_HF_HOME, DEMENTOR_GPUS. Because the live campaign runs from its own state dir, a fresh clone
with defaults gets an independent state and never clobbers it.

Importing this module has side effects (by design, once): it puts the repo root on sys.path, sets
HF_HOME for the child training procs, and creates the local-log dir. Every other sequencer module
imports it first, so this always runs before `from dementor import ...`.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _gpu_set(name: str, default: str) -> set[int]:
    # Distinguish UNSET (use default) from set-to-empty (a partner disabling a policy, e.g.
    # DEMENTOR_BLOCK_GPUS="" -> no block cards). Empty string must NOT fall back to the default.
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return {int(x) for x in raw.split(",") if str(x).strip()}


_HERE = Path(__file__).resolve().parent                          # experiments/imitation_train
REPO = Path(_env("DEMENTOR_REPO", str(_HERE.parent.parent)))     # repo root: two levels up
PY = _env("DEMENTOR_PY", sys.executable)                         # interpreter for training subprocesses
DATA_ROOT = Path(_env("DEMENTOR_DATA", str(REPO / "data")))      # big-disk data/outputs root (may be symlink)
# State (queue_state.json, local_logs/, markers). Default a dedicated dir under DEMENTOR_DATA; point
# DEMENTOR_TRAIN_STATE at an existing campaign dir to RESUME it (e.g. our /data exp_imitation_square).
STATE_DIR = Path(_env("DEMENTOR_TRAIN_STATE", str(DATA_ROOT / "imitation_train")))
QUEUE_PATH = STATE_DIR / "queue_state.json"
# Registry the training writes/reads MUST match dementor.training.matrix.DATA (= DEMENTOR_DATA/data).
REGISTRY_PATH = DATA_ROOT / "tinker_adapters.json"
REGISTRY_BACKUP_MARKER = STATE_DIR / "registry_backup_done.marker"
LOCAL_LOG_DIR = STATE_DIR / "local_logs"

# Which dataset THIS daemon instance trains. Default chatbot_arena, so an unset env reproduces the
# previous behaviour byte-for-byte. Overridable because the 16x16 grid target is 960 cells = 16
# sources x 15 targets x **4 datasets**, while this daemon only ever queued one of the four: the J4
# gap is precisely the gsm8k/oasst1/writingprompts cells that a chatbot_arena-only queue can never
# see (with the override unset, compute_queue() returns 6 cells; the real gap is ~200). The
# preference data for all four is already on disk under DPO_OUTPUT_DIR/<dataset>, and a cell only
# launches once its sft/dpo csv exist, so pointing an instance at another dataset cannot invent work.
# Run one instance per dataset, each with its own IMIT_DATASET.
DATASET = os.environ.get("IMIT_DATASET", "chatbot_arena")
SEED = int(os.environ.get("SEED", "42"))  # multi-seed: wrapper sets SEED=43,44 for robustness expansion
TINKER_PARALLEL = 4
GPU_POLL_INTERVAL = 30.0
HEARTBEAT_INTERVAL = 120.0
ME = os.environ.get("USER", "ethantsliu")

# GPU4 is compute-prohibited on OUR box; NEVER usable. Env DEMENTOR_FORBIDDEN_GPUS (default "4");
# a partner with no banned card sets it empty.
FORBIDDEN_GPUS = _gpu_set("DEMENTOR_FORBIDDEN_GPUS", "4")
# Slugs whose LOCAL training must shard across 2 GPUs (device_map model-parallel) because the
# student is too big for one 80GB card. gemma-4-31b (256K-vocab, untied lm_head VLM text tower)
# needs it. granite-4-h-small (32B-A9B) is INTENTIONALLY single-GPU: bf16 ~65GB fits one 80GB H100
# with gradient_checkpointing -- like the other 32B locals. If it ever OOMs on one card, add
# "granite-4-h-small" here to give its cells the 2-GPU model-parallel path.
MP_SOURCE_SLUGS = {"gemma-4-31b", "llama-3.3-70b"}
MEM_FREE_THRESHOLD_MIB = 1500.0

# --- disk pressure policy -------------------------------------------------------
# /data is a SHARED, chronically-near-full disk. When it hits 0 bytes free, adapter/data writes fail
# with OSError [Errno 28]. Two defenses: (1) a proactive guard -- never LAUNCH local work below
# MIN_FREE_DISK_GB, so we throttle instead of failing; (2) reactive -- a cell that dies with ENOSPC
# is RE-QUEUED (pending with backoff), never marked terminally failed.
DISK_PATH = str(DATA_ROOT)          # free-space probed on the data/outputs disk
MIN_FREE_DISK_GB = 20.0             # don't launch new local work below this
ENOSPC_RETRY_BACKOFF_SEC = 180.0    # re-queued ENOSPC cell won't relaunch before this
SQUARE_DATASETS = ["chatbot_arena"]  # the square is chatbot_arena-only; don't build the others

# --- js_park block policy -------------------------------------------------------
# GPUs 0-3 are js_park's active block on OUR box (he cycles vLLM across them all night), so a card
# there that momentarily shows no foreign process is NOT safe to grab. We treat 0-3 as OFF-LIMITS to
# local training whenever any foreign user is present on ANY of them, and only re-open once 0-3 has
# been continuously foreign-free for a SUSTAINED window. Tinker (remote) cells are unaffected.
# Env DEMENTOR_BLOCK_GPUS (default "0,1,2,3"); a partner on a DEDICATED box sets it empty.
BLOCK_GPUS = _gpu_set("DEMENTOR_BLOCK_GPUS", "0,1,2,3")
BLOCK_SUSTAINED_CLEAR_SECONDS = 600.0  # 0-3 must be foreign-free this long before we trust them

# --- shared GPU lease (coexistence with erosion_daemon / steering roster) -------
# Cards this daemon may lease-arbitrate. Canonical DEMENTOR_GPUS (default 5,6,7 on our box; a partner
# on a dedicated box sets DEMENTOR_GPUS=0,1,2,3). erosion/retry daemons share exactly these.
LEASE_GPUS = _gpu_set("DEMENTOR_GPUS", "5,6,7")
LEASE_HOLDER = "granite_imit"   # our label written into the lease meta
# Coexistence mode (default ON). When ON, before launching local work on a shared card the daemon (a)
# requires the card to read usable for SUSTAINED_POLLS consecutive polls -- so the steering roster's
# no-lease first-idle scheduler wins transient frees -- AND (b) wins the shared gpu_lease. Set
# SEQ_COEXIST=0 to restore the old immediate-grab / no-lease behavior. Never changes WHICH cells run,
# only the timing of grabbing a card.
COEXIST = os.environ.get("SEQ_COEXIST", "1") not in ("0", "false", "False", "")
SUSTAINED_POLLS = int(os.environ.get("SEQ_SUSTAINED_POLLS", "3"))
# Local-only mode (default OFF). When ON, the queue is restricted to backend==local cells so the
# tinker worker never touches Tinker-only-source cells. Set IMIT_LOCAL_ONLY=1 for the granite LOCAL
# run; left OFF, other seeds/models keep processing their tinker cells unchanged.
LOCAL_ONLY = os.environ.get("IMIT_LOCAL_ONLY", "0") in ("1", "true", "True")

# --- import-time setup (runs once, before any `from dementor import ...`) -------
sys.path.insert(0, str(REPO))
# Canonical DEMENTOR_HF_HOME (default our-box path) -> standard HF_HOME the child training procs read.
os.environ.setdefault("HF_HOME", _env("DEMENTOR_HF_HOME", "/data/ethantsliu/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with _log_lock:
        print(line, flush=True)
