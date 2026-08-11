#!/usr/bin/env python
"""Central path/env config for the STEERING RDO refusal-cone pipeline (path-portable).

Every root here is ENV-OVERRIDABLE with a sensible default, so the pipeline runs on a machine that
does NOT have our /data layout. The DEFAULTS reproduce the exact hard-coded paths the live scripts
used on our shared H100 box, so behaviour is UNCHANGED when no env var is set (semantics preserved).
Mirrors experiments/imitation_safety/erosion_common.py's `_env` pattern.

Canonical env vars (shared verbatim with the imitation package; see .env.example at the repo root):
  DEMENTOR_REPO        repo root (auto-detected from this file's location)
  DEMENTOR_MODELS_DIR  local model-weights root (per-slug subdirs); default /data/ethantsliu/models_dl
  DEMENTOR_HF_HOME     HF cache root; default ~/.cache/huggingface
  DEMENTOR_DATA        big-disk data/outputs root; default <repo>/data
  DEMENTOR_GPUS        comma list of usable GPU ids; default 5,6,7 (partner sets 0,1,2,3)
Steering-specific roots are documented below and in README.md. The three that make a fresh clone
self-contained for cone training default to the in-repo experiments/steering/data/ assets:
  DEMENTOR_STEER_DATA        train_benign.csv / harmful300.csv / gen/*_benign.csv  (default: ./data)
  DEMENTOR_REPL80            dir holding run_model.py + optional per-slug reuse   (default: this pkg)
  DEMENTOR_SALADBENCH_SPLITS harmful/harmless train+val json                      (default: ./data/saladbench_splits)
Other roots (DEMENTOR_STEER_ROOT / DEMENTOR_STEER_WORK / DEMENTOR_RTL_JUDGE_DIR / DEMENTOR_JUDGE_ALL /
DEMENTOR_STEER_BENCH_DIR) point at the larger eval/benchmark tree and are documented below.
"""
import glob
import os
import re
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))   # experiments/steering
_EXP = os.path.dirname(_HERE)                         # experiments/
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import _paths  # shared path/env primitives


_env = _paths.env  # backward-compatible alias (identical semantics)


_first = _paths.first  # first-existing-path helper (identical)


# In-repo shipped steering assets live next to this module under experiments/steering/data/
# (train_benign.csv, harmful300.csv, gen/*_benign.csv, saladbench_splits/*.json). A fresh clone
# reads them from here; our shared box's live campaign tree is only a fallback if the repo copy is
# missing (e.g. `git lfs pull` not run). The live tree is READ-ONLY here -- never written.
_REPO_DATA = os.path.join(_HERE, "data")
_LIVE_STEER = "/data/ethantsliu/exp_steer_safety"


# Repo root (for `dementor` imports). Default: two levels up from here (experiments/steering -> repo).
REPO = _paths.REPO

# Big-disk data/outputs root. Canonical env DEMENTOR_DATA (shared with the imitation package);
# DEMENTOR_DATA_ROOT kept as a legacy alias. Default: the repo's data/ dir (a symlink to big disk here).
DATA_ROOT = _paths.DATA_ROOT

# Steering experiment tree root -- holds judge_all.py + benchmarks/ that cone_eval.py consumes. NOT
# in git (lives in the steering experiment tree on our box). Override with DEMENTOR_STEER_ROOT.
STEER_ROOT = _paths.STEER_ROOT

# Where the roster reads rdo_worklist.json + writes per-model <slug>/ dirs. Default: the LIVE run
# location on our box. A partner points DEMENTOR_STEER_WORK at their own scratch dir.
WORK_ROOT = _env("DEMENTOR_STEER_WORK", os.path.join(STEER_ROOT, "repl80_rdo"))

# Steering INPUT DATA dir consumed by run_model.stage_benign/stage_derive (train_benign.csv,
# gen/llama_benign.csv reference) + cone_eval's AdvBench read (harmful300.csv). Default: the in-repo
# shipped data dir (so a fresh clone is self-contained), falling back to our box's live campaign tree
# only if the repo copy is absent. Env DEMENTOR_STEER_DATA. Byte-identical to the live copy on our
# box, so defaults reproduce our-box behaviour unchanged.
STEER_DATA = _env("DEMENTOR_STEER_DATA", _first(_REPO_DATA, _LIVE_STEER))

# Dir that holds run_model.py (imported by run_rdo_model for the benign+fingerprint DERIVE stages)
# AND, optionally, per-slug benign.csv/vectors_ml.pt to REUSE. Default: this repo package dir, which
# ships run_model.py. On a fresh box the per-slug reuse dirs (REPL80/<slug>/) are absent, so those
# stages derive fresh -- correct for a new model like nemotron-super. Env DEMENTOR_REPL80 (our box
# can point it at STEER_ROOT/repl80 to reuse the campaign's already-derived fingerprints).
REPL80 = _env("DEMENTOR_REPL80", _HERE)

# Per-slug local model-weights root. load_worklist() repoints a worklist entry to MODELS_DIR/<slug>
# when its literal (our-box) path is absent, so a partner names each slug's snapshot there and points
# DEMENTOR_MODELS_DIR at it. Default: our box's download dir (kept so behaviour is unchanged here).
MODELS_DIR = _env("DEMENTOR_MODELS_DIR", "/data/ethantsliu/models_dl")

# saladbench splits (harmful/harmless train+val json) read by compute_dim (DIM selection), rdo_port
# (cone training) and run_rdo_model/select_eval (cone-dim selection). Default: the in-repo shipped
# splits dir, falling back to our box's live geometry-of-refusal tree. Env DEMENTOR_SALADBENCH_SPLITS.
SPLITS_DIR = _env("DEMENTOR_SALADBENCH_SPLITS",
                  _first(os.path.join(_REPO_DATA, "saladbench_splits"),
                         "/data/ethantsliu/geometry-of-refusal/data/saladbench_splits"))

# RTL harm-judge infra dir (rtl_judge.py) + judge_all.py entrypoint (Qwen3-8B RTL judge subprocess).
RTL_JUDGE_DIR = _env("DEMENTOR_RTL_JUDGE_DIR", "/data/ethantsliu/exp3_safety/leak_fix")
# judge_all.py that cone_eval/run_model spawn. Default: our box's live campaign copy if present, else
# the in-repo shipped port/judge_all.py (path-portable) so a fresh clone's cone eval still judges.
JUDGE_ALL = _env("DEMENTOR_JUDGE_ALL",
                 _first(os.path.join(STEER_ROOT, "judge_all.py"),
                        os.path.join(_HERE, "port", "judge_all.py")))

# Benchmark CSV dir consumed by cone_eval.py. The in-repo data/benchmarks/ ships ONLY the 5 STANDARD
# prompts (harmbench/strongreject/xstest/sorrybench; advbench rides on harmful300.csv in STEER_DATA) --
# the SG-Bench/OR-Bench TAIL is large and deliberately not shipped. So prefer our box's COMPLETE live
# tree first, and fall back to the repo's 5-standard copy only on a fresh box with no live tree (a
# partner running the standard-5 eval). Env DEMENTOR_STEER_BENCH_DIR overrides.
BENCH_DIR = _env("DEMENTOR_STEER_BENCH_DIR",
                 _first(os.path.join(STEER_ROOT, "benchmarks"),
                        os.path.join(_REPO_DATA, "benchmarks")))

# Pre-fetched Llama-Guard-3-8B snapshot for the SG-Bench grader (auto-preferred if present).
SGBENCH_GUARD_LOCAL = _env("DEMENTOR_LLAMAGUARD_LOCAL",
                           "/data/ethantsliu/hf-cache/llamaguard3-8b-local")

# Repo .env (HF_TOKEN for gated judges; OPENAI_* only for the dormant OpenAI grader path).
REPO_ENV = _env("DEMENTOR_REPO_ENV", os.path.join(REPO, ".env"))

# HF caches. Canonical env DEMENTOR_HF_HOME (shared with the imitation package); standard HF_HOME
# kept as an alias. Default: our box's big-disk HF cache (so behaviour is unchanged here). The hub
# subcache is env-overridable and defaults to <HF_HOME>/hub.
HF_HOME = _paths.HF_HOME
HF_HUB_CACHE = _paths.HF_HUB_CACHE

# Python interpreter for subprocess re-launches. Default: the interpreter running this process, so
# the child inherits the same venv.
PY = _paths.PY

# RTL / local-classifier judge model (matches cone_eval / canonical_graders defaults exactly).
JUDGE_MODEL = _paths.RTL_JUDGE_MODEL

# GPU ids the sustained-idle daemons may use. Default 5,6,7 (our shared box: GPU4 banned, 0-3 belong
# to others). A partner on a dedicated box sets DEMENTOR_GPUS=0,1,2,3.
GPUS = _paths.GPUS

# Single fixed seed (NO multi-seed sweep for now). The RDO cone pins torch/random RNG to this in
# port/rdo_port.py; exposed here as the one canonical knob (env DEMENTOR_SEED). Default 42.
SEED = int(_env("DEMENTOR_SEED", "42"))

# The shared GPU-lease module lives with the imitation package (single shared copy, NOT duplicated
# here) so the steering daemons + the imitation-erosion daemon arbitrate over the SAME lock
# namespace. Override with DEMENTOR_IMITATION_PKG if the imitation package is elsewhere.
IMITATION_PKG = _env("DEMENTOR_IMITATION_PKG", os.path.join(REPO, "experiments", "imitation_safety"))


def worklist_path():
    """Locate rdo_worklist.json: prefer the run/scratch copy in WORK_ROOT (our box's live roster, or
    a partner's own edits), else the committed repo copy next to this module (so a fresh clone with an
    empty WORK_ROOT still finds the 32-model roster)."""
    for cand in (os.path.join(WORK_ROOT, "rdo_worklist.json"),
                 os.path.join(_HERE, "rdo_worklist.json")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"rdo_worklist.json not found in DEMENTOR_STEER_WORK ({WORK_ROOT}) or the repo ({_HERE})")


def _repo_id_from_hub_path(path):
    """Recover the HF repo id from a hub-cache path, else None.

    ".../hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/<sha>"
        -> "meta-llama/Llama-3.3-70B-Instruct"
    The org/name separator is the FIRST "--" after the "models--" prefix, so org names that
    themselves contain single hyphens ("meta-llama") survive the split.
    """
    m = re.search(r"models--([^/\\]+)", path or "")
    if not m:
        return None
    tail = m.group(1)
    return tail.replace("--", "/", 1) if "--" in tail else None


def _local_snapshot(repo_id):
    """A usable snapshot dir for repo_id inside THIS box's HF_HUB_CACHE, else None.

    Prefers the newest snapshot that actually carries weights, so a weightless config-only
    cache entry (the failure mode _resolve_rtl_judge() documents for Qwen3-8B under
    HF_HUB_OFFLINE=1) is skipped rather than returned.
    """
    folder = "models--" + repo_id.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(HF_HUB_CACHE, folder, "snapshots", "*")))
    for snap in reversed(snaps):
        if glob.glob(os.path.join(snap, "*.safetensors")) or glob.glob(
            os.path.join(snap, "*.bin")
        ):
            return snap
    return None


# --- slug aliases across the two rosters ---------------------------------------
# The steering roster (rdo_worklist.json) and the imitation roster (dementor.training.matrix
# MODEL_SLUG) name two checkpoints differently -- the steering side drops the active-parameter
# suffix. Same weights, so any join keyed on slug silently drops these models: nemotron-nano is
# CLEAN with a 0.843 cone AND has 32 imitation adapters, but nothing connected the two halves.
#
# Aliased rather than renamed on purpose: repl80_rdo/<steering slug>/ already holds each model's
# cone, vectors_ml, vectors_depths and depth sweep, and renaming those directories mid-campaign
# would orphan the artifacts every existing path expects.
SLUG_ALIASES = {
    "nemotron-nano": "nemotron-nano-30b-a3b",   # NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    "qwen3.6-35b": "qwen3.6-35b-a3b",           # Qwen3.6-35B-A3B
}
_SLUG_ALIASES_REV = {v: k for k, v in SLUG_ALIASES.items()}


def canonical_slug(slug):
    """Imitation-side (canonical) name for a slug. Unknown slugs pass through unchanged."""
    return SLUG_ALIASES.get(slug, slug)


def steering_slug(slug):
    """Steering-roster name for a slug -- i.e. the repl80_rdo/<dir> that holds its cone."""
    return _SLUG_ALIASES_REV.get(slug, slug)


def resolve_model_path(slug, path):
    """Point a worklist entry at THIS box's weights. Keep the literal path when it exists (our-box
    absolutes / local HF snapshots), else repoint to MODELS_DIR/<slug> if that exists (partner box),
    else re-resolve a stale hub-cache absolute against THIS box's HF cache, else leave unchanged.

    The third step matters because the worklist pins some entries to an absolute hub-cache snapshot
    from the box that downloaded them (e.g. llama-3.3-70b at
    /data/ethantsliu/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/<sha>).
    On a box without that layout the literal is absent and MODELS_DIR/<slug> may be too, and
    returning the dead absolute made transformers treat it as a repo id and raise
    "Repo id must be ..." BEFORE model load -- which idled the GPUs even though HF_HOME /
    HF_HUB_CACHE were pointing at a perfectly good cache. So: recover the repo id from the path and
    prefer a real snapshot in the live cache; failing that hand back the repo id itself, which HF
    resolves from the cache (or downloads, when not offline). Both are portable; the dead absolute
    never was.
    """
    if path and os.path.exists(path):
        return path
    cand = os.path.join(MODELS_DIR, slug)
    if os.path.isdir(cand):
        return cand
    repo_id = _repo_id_from_hub_path(path)
    if repo_id:
        return _local_snapshot(repo_id) or repo_id
    return path


def load_worklist(resolve=True):
    """Load the RDO roster (via worklist_path()). When resolve=True, rewrite each model's `path` for
    this box via resolve_model_path() so the SAME worklist runs against any box's weights layout."""
    wl = json.load(open(worklist_path()))
    if resolve:
        for m in wl.get("models", []):
            m["path"] = resolve_model_path(m["slug"], m.get("path", ""))
    return wl


def hf_env(offline=True):
    """The HF/PYTHONPATH env every subprocess relaunch shares (guards the HF Xet download hang).

    An explicit ambient HF_HUB_OFFLINE wins over the `offline` default, so a HF_HUB_OFFLINE=0
    relaunch can fetch cache-miss models/kernels (aya, phi-4, granite conv1d) with Xet still disabled.
    """
    e = {"HF_HOME": HF_HOME, "HF_HUB_CACHE": HF_HUB_CACHE, "HF_HUB_DISABLE_XET": "1",
         "PYTHONPATH": REPO}
    e["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1" if offline else "0")
    return e


def import_gpu_lease():
    """Import the SHARED gpu_lease module from the imitation package (reused, never duplicated).

    Both the steering daemons (benchmark_eval_daemon / retry_pc_fails) and the imitation-erosion
    daemon import this same module, so they arbitrate GPU claims over one lock namespace. The
    module's LOCK_ROOT is env-derived (DEMENTOR_IMITATION_ROOT / DEMENTOR_DATA_ROOT / DEMENTOR_REPO),
    so all daemons that inherit the same DEMENTOR_* env share the same leases.
    """
    if IMITATION_PKG not in sys.path:
        sys.path.insert(0, IMITATION_PKG)
    import gpu_lease
    return gpu_lease
