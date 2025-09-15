#!/usr/bin/env python3

import os
import pandas as pd

BASE = "disguising/scores/hierarchical_math_disguise"

MODELS = [
    ("source_meta-llama/Llama-3-8B-Instruct/target_gpt/gpt-5", "Meta-Llama-3-8B-Instruct"),
    ("source_microsoft/Phi-4-mini-instruct/target_gpt/gpt-5", "Microsoft-Phi-4-mini-instruct"),
]

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def compute_heuristic_deltas(dir_path, model_name):
    h_disg_path = os.path.join(dir_path, "heuristic_table.csv")
    h_src_path = os.path.join(dir_path, "heuristic_table_target_source.csv")
    hd = load_csv(h_disg_path)
    hs = load_csv(h_src_path)
    if hd is None or hs is None:
        return None
    m = pd.merge(hd, hs, on="style_function", suffixes=("_disg_vs_tgt", "_tgt_vs_src"))
    # Keep only match columns and compute clear delta
    out = pd.DataFrame({
        "model": model_name,
        "style_function": m["style_function"],
        "disguised_vs_target_match": m["match_disg_vs_tgt"],
        "target_vs_source_match": m["match_tgt_vs_src"],
    })
    out["match_delta"] = out["disguised_vs_target_match"] - out["target_vs_source_match"]
    return out

def compute_llm_deltas(dir_path, model_name):
    # We have disguised-vs-target scores in comparison_results.csv
    # Original-vs-target baseline may be absent. If absent, skip.
    disguised_csv = os.path.join(dir_path, "comparison_results.csv")
    original_csv = os.path.join(dir_path, "comparison_results_original.csv")
    dd = load_csv(disguised_csv)
    od = load_csv(original_csv)
    if dd is None or od is None:
        return None
    # Align by prompt; compute deltas per row
    key = "prompt"
    cols = [key, "semantic_score", "stylistic_score"]
    dd_small = dd[cols].rename(columns={
        "semantic_score": "semantic_score_disg_vs_tgt",
        "stylistic_score": "stylistic_score_disg_vs_tgt",
    })
    od_small = od[cols].rename(columns={
        "semantic_score": "semantic_score_src_vs_tgt",
        "stylistic_score": "stylistic_score_src_vs_tgt",
    })
    j = pd.merge(dd_small, od_small, on=key, how="inner")
    j["delta_semantic"] = j["semantic_score_disg_vs_tgt"] - j["semantic_score_src_vs_tgt"]
    j["delta_stylistic"] = j["stylistic_score_disg_vs_tgt"] - j["stylistic_score_src_vs_tgt"]
    j.insert(0, "model", model_name)
    return j

def main():
    all_heu = []
    all_llm = []
    for rel, name in MODELS:
        dir_path = os.path.join(BASE, rel)
        # Heuristic deltas
        h = compute_heuristic_deltas(dir_path, name)
        if h is not None:
            out_h = os.path.join(dir_path, "heuristic_deltas.csv")
            h.to_csv(out_h, index=False)
            all_heu.append(h)
            print(f"Saved heuristic deltas -> {out_h}")
        else:
            print(f"Skipping heuristic deltas for {name} (missing files)")
        # LLM deltas
        l = compute_llm_deltas(dir_path, name)
        if l is not None:
            out_l = os.path.join(dir_path, "comparison_results_deltas.csv")
            l.to_csv(out_l, index=False)
            all_llm.append(l)
            print(f"Saved LLM deltas -> {out_l}")
        else:
            print(f"Skipping LLM deltas for {name} (missing comparison_results_original.csv)")
    # Combined summaries
    if all_heu:
        pd.concat(all_heu, ignore_index=True).to_csv(os.path.join(BASE, "heuristic_deltas_all.csv"), index=False)
        print(f"Saved heuristic_deltas_all.csv in {BASE}")
    if all_llm:
        pd.concat(all_llm, ignore_index=True).to_csv(os.path.join(BASE, "comparison_results_deltas_all.csv"), index=False)
        print(f"Saved comparison_results_deltas_all.csv in {BASE}")

if __name__ == "__main__":
    main()


