from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parents[1] / "experiments" / "imitation_safety"


def tinker_module():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("tinker_erosion")
    finally:
        sys.path.remove(str(HERE))


def test_parallel_judge_launches_disjoint_batches_and_releases_leases(tmp_path, monkeypatch):
    module = tinker_module()
    adapters = [{"id": f"item-{index}"} for index in range(4)]
    done = set()
    claimed = set()
    released = set()
    launched = []
    active = 0
    max_active = 0

    class FakeProcess:
        returncode = 0

        def __init__(self, ids):
            nonlocal active, max_active
            self.ids = ids
            self.reaped = False
            active += 1
            max_active = max(max_active, active)

        def poll(self):
            nonlocal active
            if not self.reaped:
                done.update(self.ids)
                active -= 1
                self.reaped = True
            return self.returncode

    def fake_popen(cmd, **_kwargs):
        ids = cmd[3].split(",")
        launched.append(ids)
        return FakeProcess(ids)

    def try_claim(gpu, holder):
        assert holder == module.LEASE_HOLDER
        if gpu in claimed:
            return False
        claimed.add(gpu)
        return True

    def release(gpu, holder):
        assert holder == module.LEASE_HOLDER
        released.add(gpu)

    monkeypatch.setattr(module, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(module, "dlog", lambda _message: None)
    monkeypatch.setattr(module, "_done", lambda item_id: item_id in done)
    monkeypatch.setattr(module, "_sampled", lambda _item_id, _benchmarks: True)
    monkeypatch.setattr(module, "gpu_stat", lambda _gpu: (0, 0))
    monkeypatch.setattr(module.gpu_lease, "reap", lambda: [])
    monkeypatch.setattr(module.gpu_lease, "try_claim", try_claim)
    monkeypatch.setattr(module.gpu_lease, "release", release)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    args = SimpleNamespace(
        gpus="0,1,2,3",
        parallel=4,
        batch_size=1,
        util_max=5,
        mem_max=5000,
        sustained_polls=1,
        interval=0,
        gpu=None,
        once=False,
        max_prompts=200,
        subsample_seed=42,
    )
    module.phase_judge(args, adapters, [], ["advbench"], set())

    assert max_active == 4
    assert sorted(item for batch in launched for item in batch) == [
        "item-0", "item-1", "item-2", "item-3"
    ]
    assert claimed == released == {0, 1, 2, 3}


def test_reserved_campaign_defaults_to_four_way_judging(tmp_path, monkeypatch):
    module = tinker_module()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(module.EC, "WORK", str(work))
    monkeypatch.setattr(module, "GPUS_DEFAULT", [0, 1, 2, 3, 5, 7])
    assert module.default_judge_parallel() == 1
    (tmp_path / ".priority_active").write_text("exclusive evaluation pool\n")
    assert module.default_judge_parallel() == 4
    monkeypatch.setattr(module, "GPUS_DEFAULT", [5, 6, 7])
    assert module.default_judge_parallel("0,1,2,3,5,7") == 4
