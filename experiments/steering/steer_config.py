#!/usr/bin/env python
"""Central path/env config for the STEERING RDO refusal-cone pipeline (path-portable).

Every root here is ENV-OVERRIDABLE with a sensible default, so the pipeline runs on a machine that
does NOT have our /data layout. The DEFAULTS reproduce the exact hard-coded paths the live scripts
used on our shared H100 box, so behaviour is UNCHANGED when no env var is set (semantics preserved).
Mirrors experiments/imitation_safety/erosion_common.py's `_env` pattern.

Partner setup: set DEMENTOR_STEER_ROOT (the steering experiment tree holding judge_all.py +
benchmarks/), DEMENTOR_STEER_WORK (per-model <slug>/ outputs + rdo_worklist.json),
DEMENTOR_SALADBENCH_SPLITS, DEMENTOR_RTL_JUDGE_DIR, and the HF caches. See README.md.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # experiments/steering


def _env(name, default):
    v = os.environ.get(name)
    return v if v else default


# Repo root (for `dementor` imports). Default: two levels up from here (experiments/steering -> repo).
REPO = _env("DEMENTOR_REPO", os.path.dirname(os.path.dirname(_HERE)))

# Big-disk data root. Default: <repo>/data.
DATA_ROOT = _env("DEMENTOR_DATA_ROOT", os.path.join(REPO, "data"))

# Steering experiment tree root -- holds judge_all.py + benchmarks/ that cone_eval.py consumes. NOT
# in git (lives in the steering experiment tree on our box). Override with DEMENTOR_STEER_ROOT.
STEER_ROOT = _env("DEMENTOR_STEER_ROOT", "/data/ethantsliu/exp_steer_safety")

# Where the roster reads rdo_worklist.json + writes per-model <slug>/ dirs. Default: the LIVE run
# location on our box. A partner points DEMENTOR_STEER_WORK at their own scratch dir.
WORK_ROOT = _env("DEMENTOR_STEER_WORK", os.path.join(STEER_ROOT, "repl80_rdo"))

# repl80 fingerprint/benign artifacts reused by run_rdo_model (vectors_ml.pt / benign.csv).
REPL80 = _env("DEMENTOR_REPL80", os.path.join(STEER_ROOT, "repl80"))

# saladbench splits (harmful/harmless train+val json) used for cone training + dim selection.
SPLITS_DIR = _env("DEMENTOR_SALADBENCH_SPLITS",
                  "/data/ethantsliu/geometry-of-refusal/data/saladbench_splits")

# RTL harm-judge infra dir (rtl_judge.py) + judge_all.py entrypoint (Qwen3-8B RTL judge subprocess).
RTL_JUDGE_DIR = _env("DEMENTOR_RTL_JUDGE_DIR", "/data/ethantsliu/exp3_safety/leak_fix")
JUDGE_ALL = _env("DEMENTOR_JUDGE_ALL", os.path.join(STEER_ROOT, "judge_all.py"))

# Benchmark CSV dir consumed by cone_eval.py (advbench harmful300, orbench_*, sgbench, ...).
BENCH_DIR = _env("DEMENTOR_STEER_BENCH_DIR", os.path.join(STEER_ROOT, "benchmarks"))

# Pre-fetched Llama-Guard-3-8B snapshot for the SG-Bench grader (auto-preferred if present).
SGBENCH_GUARD_LOCAL = _env("DEMENTOR_LLAMAGUARD_LOCAL",
                           "/data/ethantsliu/hf-cache/llamaguard3-8b-local")

# Repo .env (HF_TOKEN for gated judges; OPENAI_* only for the dormant OpenAI grader path).
REPO_ENV = _env("DEMENTOR_REPO_ENV", os.path.join(REPO, ".env"))

# HF caches (env-first; standard HF defaults otherwise).
HF_HOME = _env("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HF_HUB_CACHE = _env("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))

# Python interpreter for subprocess re-launches. Default: the interpreter running this process, so
# the child inherits the same venv.
PY = _env("DEMENTOR_PY", sys.executable)

# RTL / local-classifier judge model (matches cone_eval / canonical_graders defaults exactly).
JUDGE_MODEL = _env("RTL_JUDGE_MODEL", "Qwen/Qwen3-8B")

# GPU ids the sustained-idle daemons may use. Default 5,6,7 (our shared box: GPU4 banned, 0-3 belong
# to others). A partner on a dedicated box sets DEMENTOR_GPUS=0,1,2,3.
GPUS = [int(x) for x in _env("DEMENTOR_GPUS", "5,6,7").split(",") if str(x).strip()]

# The shared GPU-lease module lives with the imitation package (single shared copy, NOT duplicated
# here) so the steering daemons + the imitation-erosion daemon arbitrate over the SAME lock
# namespace. Override with DEMENTOR_IMITATION_PKG if the imitation package is elsewhere.
IMITATION_PKG = _env("DEMENTOR_IMITATION_PKG", os.path.join(REPO, "experiments", "imitation_safety"))


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
