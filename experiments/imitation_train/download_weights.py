import os, sys, time
from pathlib import Path

# Portable roots: canonical DEMENTOR_REPO / DEMENTOR_HF_HOME (our-box defaults). Repo root is two
# levels up (experiments/imitation_train/); .env at the repo root supplies HF_TOKEN for gated repos.
_REPO = Path(os.environ.get("DEMENTOR_REPO") or Path(__file__).resolve().parents[2])
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", os.environ.get("DEMENTOR_HF_HOME") or "/data/ethantsliu/huggingface")
sys.path.insert(0, str(_REPO))
from dotenv import load_dotenv
load_dotenv(str(_REPO / ".env"))
from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN")

TARGETS = [
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("allenai/OLMo-3-7B-Instruct", False),
    ("adamo1139/aya-expanse-8b-ungated", False),
    ("microsoft/phi-4", False),
]

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

for repo_id, gated in TARGETS:
    ok = dl(repo_id, gated)
    if not ok and repo_id in FALLBACKS:
        mirror = FALLBACKS[repo_id]
        print(f"[{time.strftime('%H:%M:%S')}] Trying ungated mirror {mirror} for {repo_id}", flush=True)
        dl(mirror, False)

print(f"[{time.strftime('%H:%M:%S')}] ALL DOWNLOADS COMPLETE", flush=True)
