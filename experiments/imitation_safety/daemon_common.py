"""Shared primitives for the sustained-idle imitation daemons (erosion_daemon, prompt_erosion_daemon).

These three helpers were byte-identical copies in each daemon:
  * gpu_stat(g)      -- one-shot nvidia-smi (utilization.gpu %, memory.used MB); conservative
                        "busy" fallback on any error so a card we can't read is never claimed.
  * make_dlog(path)  -- factory for the timestamped stdout+file logger each daemon defines.
  * hf_offline_env(repo, hf_home, hf_hub_cache) -- the BASE_ENV builder: inherit os.environ, pin the
                        HF caches, disable Xet (avoids the download hang), and DEFAULT HF_HUB_OFFLINE=1
                        (an ambient HF_HUB_OFFLINE=0 still wins, so an uncached base can fetch).

Extracted verbatim so behavior is unchanged. The scheduling loop itself is deliberately NOT shared
here: erosion (external-worker detection + fixed worklist with an ALL-DONE break), prompt_erosion, and
the steering benchmark_eval poller (re-scans forever, no terminal break) diverge in load-bearing ways.
"""
import os
import time
import subprocess


def gpu_stat(g):
    """(utilization.gpu %, memory.used MB) for card g; conservative fallback (busy) on error."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", str(g)],
                             stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        util, mem = [int(x.strip()) for x in out.split(",")]
        return util, mem
    except Exception:
        return 100, 999999


def make_dlog(logpath):
    """Return a dlog(m) that prints a timestamped line to stdout AND appends it to logpath."""
    def dlog(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        open(logpath, "a").write(line + "\n")
    return dlog


def hf_offline_env(repo, hf_home, hf_hub_cache):
    """BASE_ENV for the item runners: inherit env, pin HF caches, disable Xet, default OFFLINE=1."""
    env = dict(os.environ, HF_HOME=hf_home, HF_HUB_CACHE=hf_hub_cache,
               HF_HUB_DISABLE_XET="1", PYTHONPATH=repo)
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env
