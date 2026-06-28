"""B2c verdict — does the gsm8k ->nemotron retention REPLICATE across datasets?

On gsm8k, the three Phase-B sources (Qwen3-4B, Llama-3.3-70B, Qwen3-32B) all
"retained" strongly into the nemotron target (DPO persistence 0.52-0.76) while
laundering into the other three targets. That single-target retention was the
ENTIRE basis for the B2b "capability drives durability" verdict. B2c re-runs the
same cells on writingprompts, chatbot_arena, and oasst1 to test whether it holds.

Pre-registered decision rule:
  - ->nemotron retention REPLICATES (stays high on wp/ca/oasst1)  -> real source x
    target (or nemotron-as-target) effect; report it as a finding.
  - It does NOT replicate (collapses to ~0 elsewhere)             -> gsm8k was a
    dataset-specific artifact; capability does NOT drive durability; Phase B is an
    honest null.

Reads every {source}_to_{target}/cell_summary.csv (method==dpo) across the four
datasets, prints the ->nemotron replication table + per-source per-dataset means
(overall and ex-nemotron), and the verdict. Writes results/durability/
b2c_cross_dataset.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["gsm8k", "writingprompts", "chatbot_arena", "oasst1"]
SOURCES = ["qwen3-4b", "llama-3.3-70b", "qwen3-32b"]
TARGETS = ["llama-3.1-8b", "qwen3.6-27b", "nemotron-nano-30b-a3b", "gpt-oss-20b"]
NEM = "nemotron-nano-30b-a3b"
RETAIN_MIN, LAUNDER_MAX = 0.15, 0.12


def dpo_persist(ds: str, src: str, tgt: str) -> float | None:
    f = ROOT / f"data/results/{ds}/analysis/cells/{src}_to_{tgt}/cell_summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    row = df[df["method"] == "dpo"]
    return None if row.empty else float(row["persistence"].iloc[0])


def main() -> None:
    rows = []
    for ds in DATASETS:
        for src in SOURCES:
            for tgt in TARGETS:
                p = dpo_persist(ds, src, tgt)
                if p is not None:
                    rows.append({"dataset": ds, "source": src, "target": tgt,
                                 "dpo_persistence": round(p, 4),
                                 "is_nemotron_target": tgt == NEM})
    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("no B2c cells scored yet")
    t.to_csv(ROOT / "results/durability/b2c_cross_dataset.csv", index=False)

    # --- the ->nemotron replication table (source x dataset) ---
    print("=== ->nemotron DPO persistence (the replication test) ===")
    nem = t[t["is_nemotron_target"]].pivot_table(
        index="source", columns="dataset", values="dpo_persistence", aggfunc="first")
    nem = nem.reindex(index=SOURCES, columns=DATASETS)
    print(nem.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "  -- "))

    # --- per-source per-dataset means: overall vs ex-nemotron ---
    print("\n=== per-source mean DPO persistence (overall | ex-nemotron) ===")
    for src in SOURCES:
        line = [f"{src:14s}"]
        for ds in DATASETS:
            cells = t[(t.source == src) & (t.dataset == ds)]
            if cells.empty:
                line.append(f"{ds[:4]}: -- ")
                continue
            allm = cells["dpo_persistence"].mean()
            exn = cells[~cells.is_nemotron_target]["dpo_persistence"].mean()
            line.append(f"{ds[:4]}: {allm:.3f}|{exn:.3f}")
        print("  " + "  ".join(line))

    # --- verdict ---
    nem_gsm = t[(t.is_nemotron_target) & (t.dataset == "gsm8k")]["dpo_persistence"]
    nem_other = t[(t.is_nemotron_target) & (t.dataset != "gsm8k")]["dpo_persistence"]
    print("\n=== VERDICT ===")
    print(f"  ->nemotron  gsm8k mean = {nem_gsm.mean():.3f} (n={len(nem_gsm)})")
    print(f"  ->nemotron  other-datasets mean = {nem_other.mean():.3f} (n={len(nem_other)})")
    if len(nem_other) and nem_other.mean() <= LAUNDER_MAX < nem_gsm.mean():
        print("  >>> NULL: the gsm8k ->nemotron retention does NOT replicate. It collapses to "
              f"~{nem_other.mean():.3f} on the other datasets. The B2b 'capability drives "
              "durability' verdict rested on a gsm8k-specific artifact. High-capability sources "
              "launder like everyone else. Phase B is an honest null.")
    elif len(nem_other) and nem_other.mean() >= RETAIN_MIN:
        print("  >>> REPLICATES: ->nemotron retention holds across datasets. Real source x target "
              "(or nemotron-as-target) effect worth a section.")
    else:
        print(f"  >>> PARTIAL/INCOMPLETE: other-datasets mean {nem_other.mean():.3f} in the "
              f"{LAUNDER_MAX}-{RETAIN_MIN} band, or not all cells scored yet.")
    print(f"\n  wrote {ROOT/'results/durability/b2c_cross_dataset.csv'}")


if __name__ == "__main__":
    main()
