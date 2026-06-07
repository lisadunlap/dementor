"""Direction 3 — safety-behavior-laundering ladder runner (SCAFFOLD, SPEND-GUARDED).

Measures whether a model's refusal behavior survives black-box capability imitation.
For each base model and each A->B SFT/DPO adapter, sample the published refusal-benchmark
prompts (AdvBench harmful + XSTest over-refusal), classify each completion with the
deterministic Tier-A classifier, and persist ONLY a binary refused verdict + a redacted
snippet — never the full completion (dual-use mitigation).

!!! THIS SCRIPT DOES NOT SPEND BY DEFAULT. !!!
Every Tinker sampling call is gated behind `--i-have-approval-to-spend`. Without that
flag the script runs a free DRY-RUN: it resolves every adapter path, builds the exact
work plan, and prints the precise sampling-call count and a rough cost estimate, then
exits BEFORE any `service.create_sampling_client(...)` / `.sample(...)` call.

COMPARATOR CONVENTION (critical — see data/results/safety/README.md):
  A->B adapters are LoRA trained with `base_model = SOURCE` (verified: all 108
  registry entries have base_model_slug == source, matching run_cell_pipeline's
  `base_model=source` for adapter rungs). Therefore the "native" comparator for an
  A->B adapter rung is r_native(SOURCE), NOT r_native(target). Drift is computed
  against the SOURCE's own refusal rate. Do not invert this.

Free dry-run:
    ./.venv/bin/python scripts/safety/run_safety_ladder.py --pilot
    ./.venv/bin/python scripts/safety/run_safety_ladder.py --full

Gated real run (requires explicit PI approval; will spend Tinker sampling budget):
    ./.venv/bin/python scripts/safety/run_safety_ladder.py --pilot --i-have-approval-to-spend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.safety.refusal_classifier import classify_refusal, redact_snippet

DATA = Path("data")
REFUSAL_DIR = DATA / "datasets/benchmarks/refusal"
DEFAULT_EVAL = REFUSAL_DIR / "eval_120h_80o_seed42.csv"
ADAPTER_REGISTRY = DATA / "tinker_adapters.json"
OUT_DIR = Path("results/safety")

# Sampling config — mirrors the matrix (run_cell_pipeline): temp 0.7, max_tokens 512.
TEMPERATURE = 0.7
MAX_TOKENS = 512  # informational; the real value lives in run_cell_pipeline.MAX_TOKENS

# Matrix definition (the 4 open-source models, by slug).
MODELS = ["llama-3.1-8b", "qwen3.6-27b", "nemotron-nano-30b-a3b", "gpt-oss-20b"]
DATASETS = ["gsm8k", "writingprompts", "chatbot_arena"]
RUNGS = ["sft", "dpo"]

# Rough $ per 1k sampling calls @ ~512 max_tokens. This is an ORDER-OF-MAGNITUDE
# placeholder for the PI to replace with the real Tinker rate card before approving.
USD_PER_1K_CALLS_PLACEHOLDER = 0.0  # set to a real figure once the rate card is known


def ordered_pairs() -> list[tuple[str, str]]:
    return [(a, b) for a in MODELS for b in MODELS if a != b]


def build_plan(args) -> dict:
    """Resolve every unit of work WITHOUT sampling. Returns a plan dict."""
    reg = json.loads(ADAPTER_REGISTRY.read_text())
    eval_df = pd.read_csv(args.eval_csv)
    n_prompts = len(eval_df)

    # Pilot = gsm8k + seed1 adapters; baselines keep 2 seeds (reused matrix-wide, cheap).
    datasets = ["gsm8k"] if args.pilot else DATASETS
    baseline_seeds = (1, 2)
    adapter_seeds = (1,) if args.pilot else (1, 2, 3)

    # Baselines: each base model x prompts x baseline_seeds.
    baseline_units = []
    for model in MODELS:
        for seed in baseline_seeds:
            baseline_units.append({
                "kind": "baseline", "model": model, "seed": seed,
                "n_prompts": n_prompts,
                "out": str(OUT_DIR / f"native_refusal/{model}_seed{seed}.csv"),
            })

    # Adapter rungs: pairs x datasets x rungs x adapter_seeds.
    adapter_units, missing = [], []
    for ds in datasets:
        for (src, tgt) in ordered_pairs():
            for rung in RUNGS:
                for seed in adapter_seeds:
                    alias = f"{rung}_{ds}_{src}_as_{tgt}_seed{seed}"
                    entry = reg.get(alias)
                    path = entry.get("path") if entry else None
                    base_model = entry.get("base_model") if entry else None
                    if not path:
                        missing.append(alias)
                        continue
                    adapter_units.append({
                        "kind": "adapter", "alias": alias, "dataset": ds,
                        "source": src, "target": tgt, "rung": rung, "seed": seed,
                        "path": path, "base_model_slug": src,  # convention: base==source
                        "registry_base_model": base_model,
                        "n_prompts": n_prompts,
                        "out": str(OUT_DIR / f"adapter_refusal/{alias}.csv"),
                    })

    total_calls = sum(u["n_prompts"] for u in baseline_units + adapter_units)
    return {
        "tier": "pilot" if args.pilot else "full",
        "n_prompts": n_prompts,
        "datasets": datasets,
        "baseline_seeds": list(baseline_seeds),
        "adapter_seeds": list(adapter_seeds),
        "baseline_units": baseline_units,
        "adapter_units": adapter_units,
        "missing_aliases": missing,
        "n_baseline_calls": sum(u["n_prompts"] for u in baseline_units),
        "n_adapter_calls": sum(u["n_prompts"] for u in adapter_units),
        "total_sampling_calls": total_calls,
        "est_usd_placeholder": round(total_calls / 1000 * USD_PER_1K_CALLS_PLACEHOLDER, 4),
    }


def build_diagonal_plan(args) -> dict:
    """A3 placebo: the 12 self-SFT adapters (model imitating its OWN outputs) on the refusal
    prompts. Verdicts -> results/safety/self_adapter_refusal/. No baselines (native rates are
    already cached). Comparator stays r_native(SOURCE) = the model itself."""
    reg = json.loads(ADAPTER_REGISTRY.read_text())
    eval_df = pd.read_csv(args.eval_csv)
    n_prompts = len(eval_df)
    units, missing = [], []
    for ds in DATASETS:
        for m in MODELS:
            alias = f"self_sft_{ds}_{m}_as_{m}_seed1"
            entry = reg.get(alias)
            path = entry.get("path") if entry else None
            if not path:
                missing.append(alias)
                continue
            units.append({
                "kind": "adapter", "alias": alias, "dataset": ds, "source": m, "target": m,
                "rung": "self_sft", "seed": 1, "path": path, "base_model_slug": m,
                "n_prompts": n_prompts,
                "out": str(OUT_DIR / f"self_adapter_refusal/{alias}.csv"),
            })
    total = sum(u["n_prompts"] for u in units)
    return {"tier": "diagonal(self-SFT placebo)", "n_prompts": n_prompts, "datasets": DATASETS,
            "baseline_seeds": [], "adapter_seeds": [1], "baseline_units": [],
            "adapter_units": units, "missing_aliases": missing, "n_baseline_calls": 0,
            "n_adapter_calls": total, "total_sampling_calls": total,
            "est_usd_placeholder": 0.0}


def print_plan(plan: dict) -> None:
    print("=" * 72)
    print(f"SAFETY LADDER — DRY RUN ({plan['tier'].upper()} tier)  [NO SPEND]")
    print("=" * 72)
    print(f"  eval prompts per cell : {plan['n_prompts']}")
    print(f"  datasets              : {plan['datasets']}")
    print(f"  baseline seeds        : {plan['baseline_seeds']}")
    print(f"  adapter seeds         : {plan['adapter_seeds']}")
    print(f"  baseline cells        : {len(plan['baseline_units'])} "
          f"-> {plan['n_baseline_calls']:,} sampling calls")
    print(f"  adapter  cells        : {len(plan['adapter_units'])} "
          f"-> {plan['n_adapter_calls']:,} sampling calls")
    print(f"  TOTAL sampling calls  : {plan['total_sampling_calls']:,} "
          f"(@ temp {TEMPERATURE}, max_tokens {MAX_TOKENS})")
    if USD_PER_1K_CALLS_PLACEHOLDER > 0:
        print(f"  est. cost             : ~${plan['est_usd_placeholder']:,} "
              f"(@ ${USD_PER_1K_CALLS_PLACEHOLDER}/1k calls)")
    else:
        print("  est. cost             : set USD_PER_1K_CALLS_PLACEHOLDER to the real "
              "Tinker rate to get a $ figure")
    if plan["missing_aliases"]:
        print(f"  [warn] {len(plan['missing_aliases'])} adapter aliases missing a path:")
        for a in plan["missing_aliases"][:10]:
            print(f"         - {a}")
    print("-" * 72)
    print("DRY RUN complete. No sampling client created, no tokens generated, $0 spent.")
    print("To actually run (and spend), re-invoke with --i-have-approval-to-spend.")


def _run_with_spend(args, plan: dict) -> None:
    """GATED. Only reached when --i-have-approval-to-spend is passed.

    Reuses run_cell_pipeline._sample_messages (the exact retry-wrapped, idempotent
    Tinker loop), then classifies + discards completions, persisting only verdicts.
    """
    # Imports deferred so the dry-run path never touches Tinker / secrets.
    from dotenv import load_dotenv
    import tinker
    from scripts.analysis.run_cell_pipeline import _sample_messages
    from workflows.run_matrix import MODEL_SLUG

    load_dotenv()
    slug_to_model = {v: k for k, v in MODEL_SLUG.items()}
    service = tinker.ServiceClient()

    eval_df = pd.read_csv(args.eval_csv)
    prompts = eval_df["prompt"].astype(str).tolist()
    categories = eval_df["category"].astype(str).tolist()

    def base_msgs(plist):
        return [[{"role": "user", "content": p}] for p in plist]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "native_refusal").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adapter_refusal").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "self_adapter_refusal").mkdir(parents=True, exist_ok=True)

    def _already_done(out_path: str) -> bool:
        """Verdict-level idempotency: skip a unit whose verdict CSV already exists
        with the full prompt set (so a resumed run doesn't re-spend on finished cells)."""
        p = Path(out_path)
        if not p.exists():
            return False
        try:
            return len(pd.read_csv(p)) == len(prompts)
        except Exception:
            return False

    def _emit(raw_csv: Path, verdict_csv: Path) -> None:
        """Classify the raw completions, write only verdicts + redacted snippets,
        then delete the raw file so no full harmful completion persists."""
        raw = pd.read_csv(raw_csv)
        out = pd.DataFrame({
            "prompt": raw["prompt"].astype(str),
            "category": categories[: len(raw)],
            "refused": raw["model_response"].map(classify_refusal).astype(int),
            "snippet_redacted": raw["model_response"].map(lambda t: redact_snippet(t, 200)),
        })
        out.to_csv(verdict_csv, index=False)
        raw_csv.unlink(missing_ok=True)  # dual-use: discard full completions

    # Baselines.
    for u in plan["baseline_units"]:
        if _already_done(u["out"]):
            print(f"  [skip] native {u['model']} seed{u['seed']}: cached", flush=True)
            continue
        model = slug_to_model[u["model"]]
        raw = OUT_DIR / f"native_refusal/_raw_{u['model']}_seed{u['seed']}.csv"
        _sample_messages(
            service, base_model=model, model_path=None,
            messages_by_prompt=base_msgs(prompts), prompts=prompts,
            clean_model=model, render_model=model, temperature=args.temperature,
            seed=u["seed"], parallel=args.parallel, out_path=raw,
            label=f"native {u['model']} seed{u['seed']}",
        )
        _emit(raw, Path(u["out"]))

    # Adapter rungs (base_model = SOURCE; neutral user turn, no disguise wrapping).
    for u in plan["adapter_units"]:
        if _already_done(u["out"]):
            print(f"  [skip] adapter {u['alias']}: cached", flush=True)
            continue
        source = slug_to_model[u["source"]]
        raw = Path(u["out"]).parent / f"_raw_{u['alias']}.csv"  # raw next to verdict (handles self_adapter_refusal/)
        _sample_messages(
            service, base_model=source, model_path=u["path"],
            messages_by_prompt=base_msgs(prompts), prompts=prompts,
            clean_model=source, render_model=source, temperature=args.temperature,
            seed=u["seed"], parallel=args.parallel, out_path=raw,
            label=f"adapter {u['alias']}",
        )
        _emit(raw, Path(u["out"]))

    print("\nGated run complete. Verdicts written under results/safety/; "
          "raw completions discarded.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--pilot", action="store_true",
                      help="gsm8k + seed1 only (cost-bounded go/no-go pilot).")
    tier.add_argument("--full", action="store_true",
                      help="Full matrix: 3 datasets x 3 adapter seeds x 2 baseline seeds.")
    tier.add_argument("--diagonal", action="store_true",
                      help="A3 placebo: the 12 self-SFT adapters (model imitating itself) on refusals.")
    ap.add_argument("--eval-csv", dest="eval_csv", type=Path, default=DEFAULT_EVAL,
                    help="Refusal eval prompt CSV (prompt,category,expected).")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--i-have-approval-to-spend", dest="i_have_approval_to_spend",
                    action="store_true", default=False,
                    help="EXPLICIT spend gate. Without this, the script dry-runs only.")
    args = ap.parse_args()

    if not (args.pilot or args.full or args.diagonal):
        args.pilot = True  # default to the cheaper plan for the dry run

    if not args.eval_csv.exists():
        raise SystemExit(
            f"Eval CSV not found: {args.eval_csv}\n"
            "Run scripts/safety/fetch_refusal_prompts.py first (free download).")

    plan = build_diagonal_plan(args) if args.diagonal else build_plan(args)
    print_plan(plan)

    if not args.i_have_approval_to_spend:
        # Hard stop before any spend. This is the scaffold boundary.
        return

    # ARMED (PI-approved 2026-06-05): pilot AND full matrix authorized by Ethan.
    # Verdict-level idempotency skips already-complete cells, so a resumed --full run
    # only pays for the fresh cells (gsm8k+seed1 pilot already done).
    print(f"\nARMED: planned {plan['total_sampling_calls']:,} Tinker sampling calls "
          f"({plan['tier']} tier); already-complete cells are skipped. PI-approved.")
    _run_with_spend(args, plan)


if __name__ == "__main__":
    main()
