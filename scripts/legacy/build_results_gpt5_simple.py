#!/usr/bin/env python3

import os
import pandas as pd

BASE = "disguising/scores/hierarchical_math_disguise"
MODELS = [
    ("source_meta-llama/Llama-3-8B-Instruct/target_gpt/gpt-5", "Meta-Llama-3-8B-Instruct"),
    ("source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5", "Microsoft-Phi-4-mini-instruct"),
]

def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    rows_llm = []
    rows_heu = []
    for rel, name in MODELS:
        d = os.path.join(BASE, rel)
        comp = read_csv(os.path.join(d, "comparison_results.csv"))
        h_disg = read_csv(os.path.join(d, "heuristic_table.csv"))
        h_src = read_csv(os.path.join(d, "heuristic_table_target_source.csv"))
        h_delta = read_csv(os.path.join(d, "heuristic_deltas.csv"))

        if comp is not None and not comp.empty:
            rows_llm.append({
                "Model": name,
                "N": len(comp),
                "SemanticMean": round(float(comp["semantic_score"].mean()), 3),
                "StylisticMean": round(float(comp["stylistic_score"].mean()), 3),
            })

        rows_heu.append({
            "Model": name,
            "Heuristics_disg_vs_tgt": None if h_disg is None or h_disg.empty else round(float(h_disg["match"].mean()), 3),
            "Heuristics_tgt_vs_src": None if h_src is None or h_src.empty else round(float(h_src["match"].mean()), 3),
            "Heuristics_delta": None if h_delta is None or h_delta.empty else round(float(h_delta["match_delta"].mean()), 3),
        })

    # Build compact markdown
    lines = []
    lines.append("## LLM (Disguised vs GPT-5)")
    lines.append("| Model | N | Semantic | Stylistic |")
    lines.append("|:--|--:|--:|--:|")
    for r in rows_llm:
        lines.append(f"| {r['Model']} | {r['N']} | {r['SemanticMean']:.3f} | {r['StylisticMean']:.3f} |")
    lines.append("")
    lines.append("## Heuristics (means)")
    lines.append("| Model | Disg vs Tgt | Tgt vs Src | Delta |")
    lines.append("|:--|--:|--:|--:|")
    for r in rows_heu:
        dv = "" if r["Heuristics_disg_vs_tgt"] is None else f"{r['Heuristics_disg_vs_tgt']:.3f}"
        tv = "" if r["Heuristics_tgt_vs_src"] is None else f"{r['Heuristics_tgt_vs_src']:.3f}"
        de = "" if r["Heuristics_delta"] is None else f"{r['Heuristics_delta']:.3f}"
        lines.append(f"| {r['Model']} | {dv} | {tv} | {de} |")

    out = os.path.join(BASE, "results_gpt5.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()


