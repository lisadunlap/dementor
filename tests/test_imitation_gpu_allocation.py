"""gpu_usability() must honour a hard per-daemon card allocation (DEMENTOR_ONLY_GPUS).

Regression test for the multi-daemon scheduler bug: DEMENTOR_GPUS governs only lease ARBITRATION,
and gpu_usability() enumerates every card nvidia-smi reports, so the launch path was never scoped to
a daemon's allocation. Four per-dataset daemons each saw all four cards and each launched baseline
generators on all of them -- 16 competing processes on 4 GPUs. lease_claim() is additionally a
documented no-op when SEQ_COEXIST=0, so coexistence-off left no arbitration to catch it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments", "imitation_train"))


@pytest.fixture()
def gpus_mod(monkeypatch):
    import gpus

    def fake_smi(args):
        joined = " ".join(args)
        # NB: the compute-apps query string also contains "uuid", so it must be matched FIRST.
        if "compute-apps" in joined:
            return []                                      # no compute apps -> no foreign users
        if "memory.used" in joined:
            return [f"{i}, 10" for i in range(4)]          # 4 cards, all nearly empty
        if "uuid" in joined:
            return [f"{i}, GPU-{i}" for i in range(4)]
        return []

    monkeypatch.setattr(gpus, "_nvidia_smi", fake_smi)
    monkeypatch.setattr(gpus, "BLOCK_GPUS", set())          # dedicated-box semantics
    monkeypatch.setattr(gpus, "FORBIDDEN_GPUS", set())
    return gpus


def test_unset_allocation_considers_every_card(gpus_mod, monkeypatch):
    """Unset must be byte-identical to the old behaviour: nothing excluded by allocation."""
    monkeypatch.setattr(gpus_mod, "ONLY_GPUS", None)
    u = gpus_mod.gpu_usability()
    assert sorted(u) == [0, 1, 2, 3]
    assert all(ok for ok, _ in u.values())
    assert not [i for i, (_, r) in u.items() if r == "not-in-allocation"]


def test_allocation_confines_daemon_to_its_cards(gpus_mod, monkeypatch):
    """A daemon given {0} must not consider 1-3 free -- that is what spawned 16 processes."""
    monkeypatch.setattr(gpus_mod, "ONLY_GPUS", {0})
    u = gpus_mod.gpu_usability()
    assert u[0][0] is True, "its own card must stay launchable"
    for idx in (1, 2, 3):
        assert u[idx] == (False, "not-in-allocation")


def test_four_daemons_partition_the_box(gpus_mod, monkeypatch):
    """Disjoint allocations => each card is launchable by exactly one daemon."""
    claims = {}
    for card in range(4):
        monkeypatch.setattr(gpus_mod, "ONLY_GPUS", {card})
        free = [i for i, (ok, _) in gpus_mod.gpu_usability().items() if ok]
        claims[card] = free
        assert free == [card]
    assert sorted(c for v in claims.values() for c in v) == [0, 1, 2, 3]


def test_prohibited_card_still_wins_inside_an_allocation(gpus_mod, monkeypatch):
    """Allocation must never re-enable a compute-prohibited card (GPU4 on box A)."""
    monkeypatch.setattr(gpus_mod, "FORBIDDEN_GPUS", {2})
    monkeypatch.setattr(gpus_mod, "ONLY_GPUS", {2})
    assert gpus_mod.gpu_usability()[2] == (False, "prohibited")
