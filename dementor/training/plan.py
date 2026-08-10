"""Dry-run planner for the full config-driven training matrix.

Enumerates every adapter job implied by ``config.yaml`` (roster x datasets x seeds
x {SFT, DPO} cross-pairs, plus self-SFT drift controls) and prints counts + the
tinker/local backend split — WITHOUT launching anything. Use it to sanity-check
the scale-up size/cost before spending.

  python -m dementor.training.plan                  # all configured datasets
  python -m dementor.training.plan --dataset gsm8k  # restrict to one dataset
"""
from __future__ import annotations

import argparse
from collections import Counter

from dementor import config


def enumerate_jobs(datasets: list[str] | None = None) -> list[dict]:
    """Every (stage, dataset, source, target, seed, backend) job in the active matrix."""
    roster = config.campaign_roster()
    backend = {m["slug"]: m["backend"] for m in roster}
    slugs = [m["slug"] for m in roster]
    ds_names = datasets or config.campaign_dataset_names()
    seeds = config.campaign_seeds()

    jobs: list[dict] = []
    for src in slugs:
        for tgt in slugs:
            if src == tgt:
                continue
            for ds in ds_names:
                for seed in seeds:
                    for stage in ("sft", "dpo"):
                        jobs.append({"stage": stage, "dataset": ds, "source": src,
                                     "target": tgt, "seed": seed, "backend": backend[src]})
    # self-SFT drift controls: each model imitates its own outputs.
    for m in slugs:
        for ds in ds_names:
            for seed in seeds:
                jobs.append({"stage": "self_sft", "dataset": ds, "source": m,
                             "target": m, "seed": seed, "backend": backend[m]})
    return jobs


def summarize(jobs: list[dict]) -> dict:
    return {
        "total": len(jobs),
        "by_stage": dict(Counter(j["stage"] for j in jobs)),
        "by_backend": dict(Counter(j["backend"] for j in jobs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-run the config-driven training matrix.")
    ap.add_argument("--dataset", action="append", help="restrict to dataset(s); repeatable")
    ap.add_argument("--limit", type=int, default=10, help="number of sample jobs to print")
    args = ap.parse_args()

    jobs = enumerate_jobs(args.dataset)
    s = summarize(jobs)
    roster = config.campaign_roster()
    n_tinker = sum(m["backend"] == "tinker" for m in roster)
    n_local = sum(m["backend"] == "local" for m in roster)

    print(f"Roster: {len(roster)} models ({n_tinker} tinker, {n_local} local) | "
          f"datasets: {args.dataset or config.campaign_dataset_names()} | "
          f"seeds: {config.campaign_seeds()}")
    print(f"Total jobs: {s['total']}")
    print(f"  by stage:   {s['by_stage']}")
    print(f"  by backend: {s['by_backend']}")
    print(f"\nSample ({min(args.limit, len(jobs))} of {len(jobs)}):")
    for j in jobs[: args.limit]:
        print(f"  [{j['backend']:6}] {j['stage']:8} {j['dataset']:13} "
              f"{j['source']} -> {j['target']} seed{j['seed']}")
    print("\n(dry-run only — no jobs launched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
