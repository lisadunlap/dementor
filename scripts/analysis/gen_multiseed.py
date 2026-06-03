"""D2 step 1: generate seed2/3 SFT+DPO eval outputs for all 36 cells, so the
disguise rungs get 3-seed confidence intervals. Generate-ONLY (no eval, so the
matrix_ladder figures aren't clobbered); idempotent/resumable (cached seed1 + any
already-done seed2/3 are skipped). Outputs are clean because run_cell_pipeline now
applies the FIXED clean_response. After this finishes, evaluate with
`decontaminate.py --adapter-seeds 3` to get per-cell mean ± CI on de-confounded text.

Usage: python -m scripts.analysis.gen_multiseed
"""
from __future__ import annotations

import types
from pathlib import Path

from scripts.analysis.run_cell_pipeline import generate

DATA = Path("data")


def main() -> None:
    cells = [c for c in sorted(DATA.glob("results/*/analysis/cells/*")) if (c / "gen").is_dir()]
    for i, cell in enumerate(cells, 1):
        ds = cell.parts[cell.parts.index("results") + 1]
        src, tgt = cell.name.split("_to_")
        print(f"[{i}/{len(cells)}] {ds}/{cell.name}", flush=True)
        cargs = types.SimpleNamespace(dataset=ds, source=src, target=tgt, eval_size=200,
                                      parallel=8, temperature=0.7, adapter_seeds=3)
        try:
            generate(cargs, cell / "gen")  # seeds 1 cached -> only seed2/3 sft+dpo are sampled
        except Exception as exc:
            print(f"  [skip] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    print("DONE generating seed2/3 SFT+DPO")


if __name__ == "__main__":
    main()
