import os
import subprocess
import sys
from pathlib import Path


def test_default_huggingface_cache_is_user_writable():
    env = os.environ.copy()
    for name in ("DEMENTOR_HF_HOME", "HF_HOME", "HF_HUB_CACHE"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", "from experiments import _paths; print(_paths.HF_HOME)"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert Path(result.stdout.strip()) == Path.home() / ".cache" / "huggingface"
