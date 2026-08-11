"""Shared path/env primitives for the Dementor experiment packages.

Both experiments/imitation_safety/erosion_common.py and experiments/steering/steer_config.py
independently declared the same `_env` helper and the same env-overridable roots (REPO, DATA_ROOT,
HF_HOME, HF_HUB_CACHE, PY, GPUS, STEER_ROOT, RTL_JUDGE_MODEL). Those are centralized here so there is
one source of truth. Every value stays environment-overridable. Machine-specific campaign roots keep
their existing defaults, while the Hugging Face cache uses the portable per-user default.

NOTE: roots whose resolution DIFFERS between the two packages (RTL_JUDGE_DIR, JUDGE_ALL, PORT, and the
package-specific registry/work/model roots) are deliberately NOT centralized here — each package keeps
its own definition (using `env`/`first` below) because their in-repo-vs-fallback candidate order
differs. This module only owns the byte-identical shared roots.
"""
import os
import sys

# experiments/_paths.py -> dirname = experiments/ -> dirname = repo root.
_REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(name, default):
    """Return $name if set and non-empty, else default. (Was `_env` in both packages.)"""
    v = os.environ.get(name)
    return v if v else default


def first(*cands):
    """First candidate path that EXISTS, else the last one (always defined). Lets a default prefer an
    in-repo shipped asset while falling back to the live tree. (Was `_first`/`_prefer`.)"""
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[-1]


# ---- byte-identical shared roots (env-overridable; defaults preserved verbatim) ----
REPO = env("DEMENTOR_REPO", _REPO_DEFAULT)
DATA_ROOT = env("DEMENTOR_DATA", env("DEMENTOR_DATA_ROOT", os.path.join(REPO, "data")))
STEER_ROOT = env("DEMENTOR_STEER_ROOT", "/data/ethantsliu/exp_steer_safety")
HF_HOME = env(
    "DEMENTOR_HF_HOME",
    env("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
)
HF_HUB_CACHE = env("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
PY = env("DEMENTOR_PY", sys.executable)
GPUS = [int(x) for x in env("DEMENTOR_GPUS", "5,6,7").split(",") if str(x).strip()]
def _resolve_rtl_judge():
    """The RTL judge model. An explicit RTL_JUDGE_MODEL wins. Otherwise prefer a COMPLETE local
    checkout under DEMENTOR_MODELS_DIR: the HF hub-cache entry for Qwen3-8B is frequently weightless
    under HF_HUB_OFFLINE=1 (config only, no *.safetensors), which makes from_pretrained raise
    'does not appear to have a file named model.safetensors' at judge time. Falling back to the repo
    id only when no local weights exist keeps the offline box working without an env var."""
    explicit = env("RTL_JUDGE_MODEL", None)
    if explicit:
        return explicit
    local = os.path.join(env("DEMENTOR_MODELS_DIR", "/data/ethantsliu/models_dl"), "Qwen3-8B")
    try:
        if os.path.isdir(local) and any(f.endswith(".safetensors") for f in os.listdir(local)):
            return local
    except OSError:
        pass
    return "Qwen/Qwen3-8B"


RTL_JUDGE_MODEL = _resolve_rtl_judge()
