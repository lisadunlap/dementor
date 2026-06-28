"""D1-FIX CAPABILITY-SPREAD CENSUS driver.

Goal: find a HARDER math benchmark that actually SPREADS the 4 base models, so a
capability gap exists to lift (gsm8k ceiling'd all 4 at ~0.83 -> no gap).

For each candidate benchmark, sample ALL 4 BASE models on ~80 prompts via Tinker
(reusing the run_cell_pipeline base-sampling recipe: per-model chat template +
clean_response), grade with the deterministic extractor in grade_mathbench.py,
and compute per-model accuracy and spread = best-worst.

HARD CAP: <= 1600 total sampling calls across the whole census.
Default: 3 candidates x 4 models x 80 prompts = 960 calls.

Outputs:
  results/d1fix_census_gen/<benchmark>/<model_slug>.csv   per-item graded rows
  results/d1fix_capability_census.csv                     per-model acc + spread
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from dementor.training.matrix import CHAT_TEMPLATE_KWARGS, MODEL_SLUG, clean_response  # noqa: E402
from experiments.analysis import grade_mathbench as G  # noqa: E402

GEN_DIR = ROOT / "results/d1fix_census_gen"
CENSUS_CSV = ROOT / "results/d1fix_capability_census.csv"
MAX_TOKENS = 1024  # harder problems need more reasoning room than gsm8k's 512

MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b"),
    ("openai/gpt-oss-20b", "gpt-oss-20b"),
    ("Qwen/Qwen3.6-27B", "qwen3.6-27b"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron-nano-30b-a3b"),
]


# ---------------------------------------------------------------------------
# Benchmark loaders: return list of dicts {prompt, gold, [n_options]} + a grader.
# ---------------------------------------------------------------------------
def load_gsm_symbolic(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test").shuffle(seed=seed)
    items = []
    for r in ds:
        gold = G.gsm_gold(r["answer"])
        if gold is None:
            continue
        items.append({"prompt": r["question"], "gold": gold})
        if len(items) >= n:
            break
    grader = lambda resp, it: G.is_correct(G.gsm_extract(resp), it["gold"])
    return items, grader


def load_math500(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    hard = [r for r in ds if int(r["level"]) >= 3 and G.is_plain_num(r["answer"])]
    import random
    random.Random(seed).shuffle(hard)
    items = []
    for r in hard[:n]:
        items.append({"prompt": r["problem"], "gold": G.math500_gold(r["answer"]),
                      "level": int(r["level"])})
    grader = lambda resp, it: G.is_correct(G.math500_extract(resp), it["gold"])
    return items, grader


def load_mmlu_pro(n: int, seed: int, category: str = "math"):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [r for r in ds if r["category"] == category]
    import random
    random.Random(seed).shuffle(rows)
    items = []
    for r in rows[:n]:
        opts = r["options"]
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        body = r["question"] + "\n\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        body += ("\n\nThink step by step, then give your final answer as a single "
                 "letter in the form 'The answer is (X).'")
        items.append({"prompt": body, "gold": str(r["answer"]).strip().upper(),
                      "n_options": len(opts)})
    grader = lambda resp, it: G.is_correct(
        G.mmlu_extract(resp, it["n_options"]), it["gold"])
    return items, grader


LOADERS = {
    "gsm_symbolic": load_gsm_symbolic,
    "math500": load_math500,
    "mmlu_pro_math": lambda n, s: load_mmlu_pro(n, s, "math"),
}


# ---------------------------------------------------------------------------
# Tinker base sampling (recipe lifted from run_cell_pipeline._sample_messages)
# ---------------------------------------------------------------------------
def sample_base(service, base_model, prompts, temperature, seed, parallel):
    import tinker
    from tinker import types

    sampling = service.create_sampling_client(base_model=base_model)
    tok = sampling.get_tokenizer()
    chat_kwargs = CHAT_TEMPLATE_KWARGS.get(base_model, {})
    try:
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature,
                                      stop=None, seed=seed)
    except TypeError:
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature, stop=None)

    def submit(prompt):
        messages = [{"role": "user", "content": prompt}]
        rendered = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, **chat_kwargs)
        mi = types.ModelInput.from_ints(tokens=tok.encode(rendered, add_special_tokens=True))
        return sampling.sample(prompt=mi, sampling_params=params, num_samples=1)

    def result(prompt, fut, retries=4):
        cur, err = fut, None
        for attempt in range(retries):
            try:
                return cur.result()
            except Exception as exc:
                err = exc
                print(f"    [retry {attempt+1}] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                cur = submit(prompt)
        raise err

    out, t0 = [], time.time()
    for start in range(0, len(prompts), parallel):
        chunk = prompts[start:start + parallel]
        futs = [(p, submit(p)) for p in chunk]
        for p, f in futs:
            raw = tok.decode(result(p, f).sequences[0].tokens)
            out.append(clean_response(base_model, raw))
        done = start + len(chunk)
        print(f"    {done}/{len(prompts)} ({done/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    return out


# ---------------------------------------------------------------------------
def run_benchmark(service, bench, n, temperature, seed, parallel, call_budget):
    items, grader = LOADERS[bench](n, seed)
    prompts = [it["prompt"] for it in items]
    print(f"\n=== {bench}: {len(items)} prompts ===", flush=True)
    bdir = GEN_DIR / bench
    bdir.mkdir(parents=True, exist_ok=True)

    rows = []
    calls = 0
    for base_model, slug in MODELS:
        out_path = bdir / f"{slug}.csv"
        if out_path.exists() and len(pd.read_csv(out_path)) == len(items):
            print(f"  [skip] {slug}: cached", flush=True)
            df = pd.read_csv(out_path)
        else:
            if calls + len(prompts) > call_budget["remaining"]:
                print(f"  [BUDGET] would exceed cap; stopping at {slug}", flush=True)
                break
            print(f"  [gen] {slug}: {len(prompts)} prompts", flush=True)
            resps = sample_base(service, base_model, prompts, temperature, seed, parallel)
            calls += len(prompts)
            call_budget["remaining"] -= len(prompts)
            recs = []
            for it, resp in zip(items, resps):
                correct = grader(resp, it)
                rec = {"prompt": it["prompt"], "gold": it["gold"],
                       "model_response": resp, "correct": int(correct)}
                if "level" in it:
                    rec["level"] = it["level"]
                recs.append(rec)
            df = pd.DataFrame(recs)
            df.to_csv(out_path, index=False)
        acc = df["correct"].mean()
        rows.append({"benchmark": bench, "model": slug, "n": len(df),
                     "n_correct": int(df["correct"].sum()), "acc": round(float(acc), 4)})
        print(f"  {slug}: acc={acc:.3f} ({int(df['correct'].sum())}/{len(df)})", flush=True)
    return rows, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(LOADERS))
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--cap", type=int, default=1600, help="hard cap on total sampling calls")
    args = ap.parse_args()

    load_dotenv()
    import tinker
    service = tinker.ServiceClient()

    call_budget = {"remaining": args.cap}
    all_rows, total_calls = [], 0
    for bench in args.benchmarks:
        rows, calls = run_benchmark(service, bench, args.n, args.temperature,
                                    args.seed, args.parallel, call_budget)
        all_rows.extend(rows)
        total_calls += calls

    cdf = pd.DataFrame(all_rows)
    # spread per benchmark
    summary = []
    for bench, g in cdf.groupby("benchmark"):
        best = g.loc[g["acc"].idxmax()]
        worst = g.loc[g["acc"].idxmin()]
        spread = float(best["acc"] - worst["acc"])
        for _, r in g.iterrows():
            summary.append({**r.to_dict(),
                            "best_model": best["model"], "best_acc": float(best["acc"]),
                            "worst_model": worst["model"], "worst_acc": float(worst["acc"]),
                            "spread": round(spread, 4)})
    CENSUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(CENSUS_CSV, index=False)
    print(f"\n[census] wrote {len(sdf)} rows -> {CENSUS_CSV}")
    print(f"[calls] total sampling calls this run: {total_calls}")
    print("\n=== per-model accuracy ===")
    print(cdf.pivot_table(index="benchmark", columns="model", values="acc").round(3).to_string())
    print("\n=== spread per benchmark ===")
    print(sdf[["benchmark", "best_model", "best_acc", "worst_model", "worst_acc", "spread"]]
          .drop_duplicates("benchmark").to_string(index=False))


if __name__ == "__main__":
    main()
