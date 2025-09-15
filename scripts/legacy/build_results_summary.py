#!/usr/bin/env python3

import os
import pandas as pd

BASE = "disguising/scores/hierarchical_math_disguise"

MODELS = [
    ("source_meta-llama/Llama-3-8B-Instruct/target_gpt/gpt-5", "Meta-Llama-3-8B-Instruct"),
    ("source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5", "Microsoft-Phi-4-mini-instruct"),
]

def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def summarize_llm(dir_path):
    p = os.path.join(dir_path, "comparison_results.csv")
    df = safe_read_csv(p)
    if df is None or df.empty:
        return {"n": 0, "sem_mean": None, "sem_std": None, "sty_mean": None, "sty_std": None}
    return {
        "n": len(df),
        "sem_mean": df["semantic_score"].mean(),
        "sem_std": df["semantic_score"].std(),
        "sty_mean": df["stylistic_score"].mean(),
        "sty_std": df["stylistic_score"].std(),
    }

def summarize_heuristics(dir_path):
    h_disg = safe_read_csv(os.path.join(dir_path, "heuristic_table.csv"))
    h_src = safe_read_csv(os.path.join(dir_path, "heuristic_table_target_source.csv"))
    h_delta = safe_read_csv(os.path.join(dir_path, "heuristic_deltas.csv"))
    res = {"disg_tgt": None, "tgt_src": None, "delta": None}
    if h_disg is not None and not h_disg.empty:
        res["disg_tgt"] = h_disg["match"].mean()
    if h_src is not None and not h_src.empty:
        res["tgt_src"] = h_src["match"].mean()
    if h_delta is not None and not h_delta.empty and "match_delta" in h_delta.columns:
        res["delta"] = h_delta["match_delta"].mean()
    return res

def main():
    rows_llm = []
    rows_heu = []
    for rel, name in MODELS:
        dir_path = os.path.join(BASE, rel)
        llm = summarize_llm(dir_path)
        heu = summarize_heuristics(dir_path)
        rows_llm.append({
            "Model": name,
            "N": llm["n"],
            "Semantic": None if llm["sem_mean"] is None else f"{llm['sem_mean']:.3f} ± {llm['sem_std']:.3f}",
            "Stylistic": None if llm["sty_mean"] is None else f"{llm['sty_mean']:.3f} ± {llm['sty_std']:.3f}",
        })
        rows_heu.append({
            "Model": name,
            "Heuristics (disg vs tgt)": None if heu["disg_tgt"] is None else f"{heu['disg_tgt']:.3f}",
            "Heuristics (tgt vs src)": None if heu["tgt_src"] is None else f"{heu['tgt_src']:.3f}",
            "Heuristics Δ": None if heu["delta"] is None else f"{heu['delta']:.3f}",
        })

    # Build compact markdown
    lines = []
    lines.append("# Hierarchical Math Disguise - Results")
    lines.append("")
    lines.append("## LLM (Disguised vs GPT-5)")
    df_llm = pd.DataFrame(rows_llm)
    lines.append(df_llm.to_markdown(index=False))
    lines.append("")
    lines.append("## Heuristics")
    df_heu = pd.DataFrame(rows_heu)
    lines.append(df_heu.to_markdown(index=False))
    lines.append("")

    out_path = os.path.join(BASE, "results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()


