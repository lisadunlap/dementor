#!/usr/bin/env python
"""Central path/env config for the STEERING RDO refusal-cone pipeline (path-portable).

Every root here is ENV-OVERRIDABLE with a sensible default, so the pipeline runs on a machine that
does NOT have our /data layout. The DEFAULTS reproduce the exact hard-coded paths the live scripts
used on our shared H100 box, so behaviour is UNCHANGED when no env var is set (semantics preserved).
Mirrors experiments/imitation_safety/erosion_common.py's `_env` pattern.

Canonical env vars (shared verbatim with the imitation package; see .env.example at the repo root):
  DEMENTOR_REPO        repo root (auto-detected from this file's location)
  DEMENTOR_MODELS_DIR  local model-weights root (per-slug subdirs); default /data/ethantsliu/models_dl
  DEMENTOR_HF_HOME     HF cache root; default /data/ethantsliu/huggingface
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
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))   # experiments/steering


def _env(name, default):
    v = os.environ.get(name)
    return v if v else default


def _first(*cands):
    """First candidate path that EXISTS, else the last one (so the value is always defined).
    Lets a default prefer the in-repo shipped asset while falling back to our box's live tree."""
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[-1]


# In-repo shipped steering assets live next to this module under experiments/steering/data/
# (train_benign.csv, harmful300.csv, gen/*_benign.csv, saladbench_splits/*.json). A fresh clone
# reads them from here; our shared box's live campaign tree is only a fallback if the repo copy is
# missing (e.g. `git lfs pull` not run). The live tree is READ-ONLY here -- never written.
_REPO_DATA = os.path.join(_HERE, "data")
_LIVE_STEER = "/data/ethantsliu/exp_steer_safety"


# Repo root (for `dementor` imports). Default: two levels up from here (experiments/steering -> repo).
REPO = _env("DEMENTOR_REPO", os.path.dirname(os.path.dirname(_HERE)))

# Big-disk data/outputs root. Canonical env DEMENTOR_DATA (shared with the imitation package);
# DEMENTOR_DATA_ROOT kept as a legacy alias. Default: the repo's data/ dir (a symlink to big disk here).
DATA_ROOT = _env("DEMENTOR_DATA", _env("DEMENTOR_DATA_ROOT", os.path.join(REPO, "data")))

# Steering experiment tree root -- holds judge_all.py + benchmarks/ that cone_eval.py consumes. NOT
# in git (lives in the steering experiment tree on our box). Override with DEMENTOR_STEER_ROOT.
STEER_ROOT = _env("DEMENTOR_STEER_ROOT", "/data/ethantsliu/exp_steer_safety")

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
HF_HOME = _env("DEMENTOR_HF_HOME", _env("HF_HOME", "/data/ethantsliu/huggingface"))
HF_HUB_CACHE = _env("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))

# Python interpreter for subprocess re-launches. Default: the interpreter running this process, so
# the child inherits the same venv.
PY = _env("DEMENTOR_PY", sys.executable)

# RTL / local-classifier judge model (matches cone_eval / canonical_graders defaults exactly).
JUDGE_MODEL = _env("RTL_JUDGE_MODEL", "Qwen/Qwen3-8B")

# GPU ids the sustained-idle daemons may use. Default 5,6,7 (our shared box: GPU4 banned, 0-3 belong
# to others). A partner on a dedicated box sets DEMENTOR_GPUS=0,1,2,3.
GPUS = [int(x) for x in _env("DEMENTOR_GPUS", "5,6,7").split(",") if str(x).strip()]

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


def resolve_model_path(slug, path):
    """Point a worklist entry at THIS box's weights. Keep the literal path when it exists (our-box
    absolutes / local HF snapshots), else repoint to MODELS_DIR/<slug> if that exists (partner box),
    else leave unchanged (an HF repo id resolved from the HF cache)."""
    if path and os.path.exists(path):
        return path
    cand = os.path.join(MODELS_DIR, slug)
    if os.path.isdir(cand):
        return cand
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
    """The HF/PYTHONPATH env every subprocess relaunch shares (guards the HF Xet download hang)."""
    e = {"HF_HOME": HF_HOME, "HF_HUB_CACHE": HF_HUB_CACHE, "HF_HUB_DISABLE_XET": "1",
         "PYTHONPATH": REPO}
    if offline:
        e["HF_HUB_OFFLINE"] = "1"
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
