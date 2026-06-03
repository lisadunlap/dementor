"""Parametrized end-to-end behavioral-inertia cell pipeline for the open-source
matrix: generate eval outputs (baselines + prompting rungs + adapter rungs) via
Tinker, stage them on shared endpoints, run the supervised-basis cell with both
controls, and print the calibrated ladder.

All generations are at one temperature (default 0.7) so the self-baseline's
inter-seed variance is comparable to the disguised draws. Idempotent: cached
outputs (right row count) are skipped, so reruns only fill gaps.

Example:
    python -m scripts.analysis.run_cell_pipeline \
        --dataset gsm8k --source llama-3.1-8b --target gpt-oss-20b --eval-size 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from scripts.analysis.behavioral_cell_evaluator import load_cell_spec, run_behavioral_cell
from scripts.methods.get_method import get_method
from workflows.run_matrix import CHAT_TEMPLATE_KWARGS, MODEL_SLUG, clean_response

SLUG_TO_MODEL = {v: k for k, v in MODEL_SLUG.items()}
DATA = Path("data")
EVAL_PROMPTS = {
    "gsm8k": DATA / "datasets/gsm8k/gsm8k_prompts_eval_200_seed42.csv",
    "chatbot_arena": DATA / "datasets/chatbot_arena/chatbot_arena_prompts_eval_1000_seed42.csv",
    "writingprompts": DATA / "datasets/writingprompts/writingprompts_eval_500_seed42.csv",
}
TRAIN_BASELINE = lambda ds, slug: DATA / f"model-responses/matrix_baselines/{ds}/{slug}_train.csv"
PROMPT_METHODS = ["just_name_it", "random_sampling", "stylistic"]
MAX_TOKENS = 512


def _cached(path: Path, n: int) -> bool:
    return path.exists() and len(pd.read_csv(path)) == n


def _tok_for(service, base_model, sampling):
    if hasattr(sampling, "get_tokenizer"):
        return sampling.get_tokenizer()
    return service.create_lora_training_client(base_model=base_model).get_tokenizer()


def _sample_messages(service, *, base_model, model_path, messages_by_prompt, prompts,
                     clean_model, render_model, temperature, seed, parallel, out_path, label):
    import tinker
    from tinker import types

    if _cached(out_path, len(prompts)):
        print(f"  [skip] {label}: cached", flush=True)
        return
    sampling = (service.create_sampling_client(model_path=model_path) if model_path
                else service.create_sampling_client(base_model=base_model))
    tok = _tok_for(service, base_model, sampling)
    chat_kwargs = CHAT_TEMPLATE_KWARGS.get(render_model, {})
    try:
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature, stop=None, seed=seed)
    except TypeError:
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=temperature, stop=None)

    def submit(messages):
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        mi = types.ModelInput.from_ints(tokens=tok.encode(rendered, add_special_tokens=True))
        return sampling.sample(prompt=mi, sampling_params=params, num_samples=1)

    def result(messages, fut, retries=4):
        cur, err = fut, None
        for attempt in range(retries):
            try:
                return cur.result()
            except Exception as exc:
                err = exc
                print(f"    [retry {attempt + 1}] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                cur = submit(messages)
        raise err

    print(f"  [gen] {label}: {len(prompts)} prompts", flush=True)
    rows, t0 = [], time.time()
    for start in range(0, len(prompts), parallel):
        chunk = list(zip(prompts[start : start + parallel], messages_by_prompt[start : start + parallel]))
        futs = [(p, m, submit(m)) for p, m in chunk]
        for p, m, f in futs:
            rows.append({"prompt": p, "model_response": clean_response(clean_model, tok.decode(result(m, f).sequences[0].tokens))})
        done = start + len(chunk)
        if done % 50 < parallel or done == len(prompts):
            print(f"    {done}/{len(prompts)} ({done/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def generate(args, gen: Path) -> None:
    load_dotenv()
    import tinker

    source = SLUG_TO_MODEL[args.source]
    target = SLUG_TO_MODEL[args.target]
    prompts = pd.read_csv(EVAL_PROMPTS[args.dataset])["prompt"].astype(str)
    if args.eval_size and args.eval_size < len(prompts):
        prompts = prompts.sample(n=args.eval_size, random_state=42)
    prompts = prompts.tolist()
    reg = json.loads(Path("data/tinker_adapters.json").read_text())
    service = tinker.ServiceClient()

    def base_msgs(plist):
        return [[{"role": "user", "content": p}] for p in plist]

    # Baselines: source x2 seeds, target x2 seeds.
    for model, slug, role in [(source, args.source, "source"), (target, args.target, "target")]:
        for seed in (1, 2):
            _sample_messages(service, base_model=model, model_path=None, messages_by_prompt=base_msgs(prompts),
                             prompts=prompts, clean_model=model, render_model=model, temperature=args.temperature,
                             seed=seed, parallel=args.parallel, out_path=gen / f"{role}_seed{seed}.csv",
                             label=f"{role} {slug} seed{seed}")

    # Prompting rungs: build disguise messages from TRAIN examples, sample SOURCE.
    tgt_train = pd.read_csv(TRAIN_BASELINE(args.dataset, args.target)).rename(columns={"model_response": "target_response"})
    src_train = pd.read_csv(TRAIN_BASELINE(args.dataset, args.source)).rename(columns={"model_response": "source_response"})
    for method_name in PROMPT_METHODS:
        out_path = gen / f"rung_{method_name}.csv"
        if _cached(out_path, len(prompts)):
            print(f"  [skip] rung {method_name}: cached", flush=True)
            continue
        method = get_method(method_name, source, target, disguise_df=tgt_train.copy(),
                            source_df=src_train.copy(), method_kwargs={"seed": 1})
        msgs = [method.forward(p) for p in prompts]
        _sample_messages(service, base_model=source, model_path=None, messages_by_prompt=msgs, prompts=prompts,
                         clean_model=source, render_model=source, temperature=args.temperature, seed=1,
                         parallel=args.parallel, out_path=out_path, label=f"rung {method_name}")

    # Adapter rungs: SFT, DPO via Tinker sampler paths, over `adapter_seeds` seeds.
    # seed1 keeps the legacy filename rung_{rung}.csv; seeds 2+ are rung_{rung}_seed{n}.csv.
    for rung in ("sft", "dpo"):
        for seed in range(1, args.adapter_seeds + 1):
            alias = f"{rung}_{args.dataset}_{args.source}_as_{args.target}_seed{seed}"
            path = reg.get(alias, {}).get("path")
            if not path:
                print(f"  [warn] {alias} missing sampler path; skipping", flush=True)
                continue
            out = gen / (f"rung_{rung}.csv" if seed == 1 else f"rung_{rung}_seed{seed}.csv")
            _sample_messages(service, base_model=source, model_path=path, messages_by_prompt=base_msgs(prompts),
                             prompts=prompts, clean_model=source, render_model=source, temperature=args.temperature,
                             seed=seed, parallel=args.parallel, out_path=out, label=f"rung {rung} seed{seed}")


def assemble_and_run(args, cell: Path, gen: Path) -> dict:
    staged = cell / "staged"
    staged.mkdir(parents=True, exist_ok=True)

    def col(csv, new):
        d = pd.read_csv(csv)[["prompt", "model_response"]].copy()
        d["prompt"] = d["prompt"].astype(str)
        return d.dropna(subset=["prompt"]).drop_duplicates("prompt").rename(columns={"model_response": new})

    source = col(gen / "source_seed1.csv", "source_response")
    target = col(gen / "target_seed1.csv", "target_response")
    adapter_labels = [
        rung if seed == 1 else f"{rung}_seed{seed}"
        for rung in ("sft", "dpo")
        for seed in range(1, args.adapter_seeds + 1)
    ]
    methods = []
    for label in PROMPT_METHODS + adapter_labels:
        rung = gen / f"rung_{label}.csv"
        if not rung.exists():
            continue
        st = col(rung, "model_response").merge(source, on="prompt").merge(target, on="prompt")
        path = staged / f"{label}.csv"
        st[["prompt", "source_response", "target_response", "model_response"]].to_csv(path, index=False)
        methods.append({"method": label, "comparison_csv": str(path)})

    manifest = {
        "dataset": args.dataset, "source_model": SLUG_TO_MODEL[args.source],
        "target_model": SLUG_TO_MODEL[args.target], "output_dir": str(cell),
        "feature_set": "full", "feature_ablation_sets": [], "basis_type": "supervised",
        "k": 5, "bootstrap_samples": args.bootstrap,
        "encoder_model": getattr(args, "encoder_model", None),
        "calibration_judge_model": args.calibration_judge,
        "calibration_sample_size": args.calibration_n,
        "self_baseline": {"source_runs": [str(gen / "source_seed1.csv"), str(gen / "source_seed2.csv")]},
        "identity_control": {"target_runs": [str(gen / "target_seed2.csv")]},
        "methods": methods,
    }
    (cell / "cell.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_behavioral_cell(load_cell_spec(cell / "cell.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(EVAL_PROMPTS))
    ap.add_argument("--source", required=True, choices=list(SLUG_TO_MODEL))
    ap.add_argument("--target", required=True, choices=list(SLUG_TO_MODEL))
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--adapter-seeds", type=int, default=1,
                    help="Number of SFT/DPO adapter seeds (1-3) to run as separate rungs for rung-level CIs.")
    ap.add_argument("--calibration-judge", default=None,
                    help="LLM judge model (e.g. openai/gpt-4.1-mini) to validate persistence against; off if unset.")
    ap.add_argument("--calibration-n", type=int, default=0,
                    help="Per-method (prompt,disguised,target) triples to score with the judge.")
    ap.add_argument("--gen-only", action="store_true")
    args = ap.parse_args()

    cell = DATA / f"results/{args.dataset}/analysis/cells/{args.source}_to_{args.target}"
    gen = cell / "gen"
    print(f"cell: {args.dataset}  {args.source} -> {args.target}  (eval {args.eval_size})\n")
    generate(args, gen)
    if args.gen_only:
        print("\ngeneration complete (--gen-only).")
        return
    summary = assemble_and_run(args, cell, gen)
    df = pd.read_csv(cell / "cell_summary.csv")
    cols = ["method", "persistence", "movement_raw", "over_assimilation",
            "anchored", "anchored_ci_low", "anchored_ci_high",
            "z_vs_baseline", "probe_cv", "trustworthy"]
    print(f"\n=== {args.dataset}  {args.source} -> {args.target} (supervised, anchored) ===")
    print(f"B={summary['baseline_persistence']:.3f} I={summary['identity_persistence']:.3f} "
          f"sep_captured={summary['sep_ratio']:.3f}")
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))

    corr_path = cell / "calibration" / "calibration_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        if not corr.empty:
            print("\n=== LLM-judge calibration (deterministic metric vs judge scores) ===")
            print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
