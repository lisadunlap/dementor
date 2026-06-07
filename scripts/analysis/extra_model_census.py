"""B1 — Native MATH-500 capability census for Tinker-TRAINABLE candidate source models.

The cached extra-model responses (gemma/mistral/phi/qwen2.5) are NOT Tinker-trainable, so
candidates come from Tinker's supported list. We sample each candidate's NATIVE MATH-500
accuracy to confirm a capability spread and to pick K=2 dissociators for B2a.

Dissociation logic: the original 4 confound capability with durability ({nemotron,gpt-oss}
high-cap+retain; {qwen,llama} low-cap+launder). The decisive new sources are HIGH-capability
models from the LAUNDERER lineages (llama, qwen): if they launder despite high capability ->
durability != capability (separate axis); if they retain -> capability drives durability.

Reuses census_mathbench.sample_base + load_math500 + grade_mathbench.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from workflows.run_matrix import CHAT_TEMPLATE_KWARGS  # noqa: E402
from scripts.analysis import grade_mathbench as G  # noqa: E402
from scripts.analysis.census_mathbench import load_math500, sample_base  # noqa: E402

GEN = ROOT / "results/durability/extra_census_gen"
OUT = ROOT / "results/durability/extra_census.csv"

# (base_model, slug, lineage, chat_template_kwargs). All verified in Tinker's supported list.
CANDIDATES = [
    ("meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b", "llama(launderer)", {}),
    ("Qwen/Qwen3-32B", "qwen3-32b", "qwen(launderer)", {"enable_thinking": False}),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3-4b", "qwen(launderer)", {}),
    ("openai/gpt-oss-120b", "gpt-oss-120b", "gpt-oss(retainer)", {"reasoning_effort": "low"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    load_dotenv(str(ROOT / ".env"))
    import tinker
    service = tinker.ServiceClient()

    items, _ = load_math500(args.n, args.seed)
    prompts = [it["prompt"] for it in items]
    golds = [it["gold"] for it in items]
    GEN.mkdir(parents=True, exist_ok=True)

    rows = []
    for base, slug, lineage, kwargs in CANDIDATES:
        CHAT_TEMPLATE_KWARGS[base] = kwargs  # register chat kwargs for sample_base
        out_path = GEN / f"{slug}.csv"
        if out_path.exists() and len(pd.read_csv(out_path)) == len(prompts):
            df = pd.read_csv(out_path)
            print(f"[skip] {slug}: cached", flush=True)
        else:
            print(f"[gen] {slug} ({base})", flush=True)
            resps = sample_base(service, base, prompts, 0.0, args.seed, args.parallel)
            recs = [{"prompt": p, "gold": g, "model_response": r,
                     "correct": int(G.is_correct(G.math500_extract(str(r)), g))}
                    for p, g, r in zip(prompts, golds, resps)]
            df = pd.DataFrame(recs)
            df.to_csv(out_path, index=False)
        acc = df["correct"].mean()
        rows.append({"base_model": base, "slug": slug, "lineage": lineage,
                     "math500_acc": round(float(acc), 4)})
        print(f"  {slug}: MATH-500 acc={acc:.3f}", flush=True)

    cen = pd.DataFrame(rows).sort_values("math500_acc", ascending=False)
    cen.to_csv(OUT, index=False)
    print("\n=== B1 candidate MATH-500 capability census ===")
    print(cen.to_string(index=False))
    print("\n  Original-4 reference: nemotron 0.80, gpt-oss 0.79 (retain) | qwen 0.50, llama 0.45 (launder)")
    print("  Pick: high-cap launderer-lineage models as the B2a dissociators.")


if __name__ == "__main__":
    main()
