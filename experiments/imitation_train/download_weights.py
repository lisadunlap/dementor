"""One-shot helper to pre-download the small local-backend student weights for the imitation
square into the HF cache, so the sequencer's local track can train them offline.

Fetches the current local-backend imitation students that download cleanly via snapshot_download
(Llama-3.1-8B, OLMo-3-7B, aya-expanse-8b, phi-4), disabling the HF Xet backend (which hangs on this
box) and falling back to an ungated mirror for gated repos. Run by hand once per box:

    DEMENTOR_HF_HOME=/big/disk/hf python experiments/imitation_train/download_weights.py
"""
import os, sys, time
from pathlib import Path

# Portable roots: canonical DEMENTOR_REPO / DEMENTOR_HF_HOME (our-box defaults). Repo root is two
# levels up (experiments/imitation_train/); .env at the repo root supplies HF_TOKEN for gated repos.
_REPO = Path(os.environ.get("DEMENTOR_REPO") or Path(__file__).resolve().parents[2])
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault(
    "HF_HOME",
    os.environ.get("DEMENTOR_HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
)
sys.path.insert(0, str(_REPO))
from dotenv import load_dotenv
load_dotenv(str(_REPO / ".env"))
from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN")

# A download script cannot honour an ambient HF_HUB_OFFLINE=1 (the steering pipeline's default via
# steer_config.hf_env()); offline would turn every fetch below into a cache-miss error.
os.environ["HF_HUB_OFFLINE"] = "0"

TARGETS = [
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("allenai/OLMo-3-7B-Instruct", False),
    ("adamo1139/aya-expanse-8b-ungated", False),
    ("microsoft/phi-4", False),
]

# Opt-in, named-only targets: too large (or too situational) for the default sweep. Fetch with
#   python download_weights.py llama-3.3-70b
#   python download_weights.py j4-qwen          # the three J4 imitation sources, all ungated
# llama-3.3-70b bootstraps a bare box for J5 adapter steering (~140 GB, GATED: needs HF_TOKEN).
# The J4 Qwen sources are UNGATED, so a box with no token can still fetch them and run J4 locally
# (config.yaml marks them backend=tinker only because no box held local weights; see
# dementor.config._local_backend_overrides).
EXTRA = {
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct", True),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", False),
    "qwen3.6-27b": ("Qwen/Qwen3.6-27B", False),
    "qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", False),
}

# Named bundles, expanded by _selected().
GROUPS = {
    "j4-qwen": ["qwen3.5-4b", "qwen3.6-27b", "qwen3.6-35b-a3b"],
}

FALLBACKS = {
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

def dl(repo_id, gated):
    print(f"[{time.strftime('%H:%M:%S')}] START {repo_id}", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            token=TOKEN if gated else None,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        print(f"[{time.strftime('%H:%M:%S')}] DONE {repo_id} -> {path}", flush=True)
        return True
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] FAILED {repo_id}: {type(e).__name__}: {e}", flush=True)
        return False

def _selected(argv):
    """No args -> the default TARGETS sweep (unchanged). Args -> only those, by GROUPS bundle,
    EXTRA key, default-target repo id, or bare repo id."""
    if not argv:
        return list(TARGETS)
    names = []
    for a in argv:
        names.extend(GROUPS.get(a, [a]))
    picked = []
    known = {r: g for r, g in TARGETS}
    for name in names:
        if name in EXTRA:
            picked.append(EXTRA[name])
        elif name in known:
            picked.append((name, known[name]))
        else:
            picked.append((name, True))  # unknown repo id: try it gated (token is used if present)
    return picked


for repo_id, gated in _selected(sys.argv[1:]):
    ok = dl(repo_id, gated)
    if not ok and repo_id in FALLBACKS:
        mirror = FALLBACKS[repo_id]
        print(f"[{time.strftime('%H:%M:%S')}] Trying ungated mirror {mirror} for {repo_id}", flush=True)
        dl(mirror, False)

print(f"[{time.strftime('%H:%M:%S')}] ALL DOWNLOADS COMPLETE", flush=True)
