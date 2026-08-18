"""Matrix dispatcher for the config-defined imitation campaign.

The named campaign in ``config.yaml`` supplies the model cohort, datasets, seeds,
and training/control stages. Use ``dementor-plan`` or ``list-cells`` to inspect the
resolved matrix without launching work.

Subcommands:
  generate-target-responses: generate baseline responses on TRAIN splits for use as SFT completions
  build-sft-data: join (train_prompt, target_response) pairs into SFT CSVs
  launch-sft: submit SFT jobs to Tinker (one per cell × seed)
  list-cells: print the matrix without launching anything
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dementor import config

from ._cells import (
    Cell,
    iter_cells,
    iter_self_sft_cells,
    safety_sft_data_path,
    self_sft_data_path,
)
from ._constants import (
    DATA,
    DEFAULT_SAFETY_EXCLUDE_PROMPTS,
    DEFAULT_SAFETY_PROMPTS,
    DEFAULT_SAFETY_REPLAY_SIZE,
    MODELS,
    SAFETY_SFT_OUTPUT_DIR,
    SELF_SFT_OUTPUT_DIR,
    TRAIN_DATASETS,
)
from ._dpo import backfill_register_dpo, build_dpo_data, launch_dpo, launch_safety_dpo
from ._local_cell import launch_local_cell
from ._push import backfill_export_adapters, push_adapters_to_hf
from ._safety import build_safety_dpo_data, build_safety_sft_data
from ._sft import build_self_sft_data, build_sft_data, launch_sft, monitor_self_sft_controls
from ._target_responses import generate_target_responses


def _resolve_model_arg(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return config.model(value)["id"]
    except KeyError as exc:
        raise SystemExit(f"Unknown model id/slug: {value}") from exc


def _require_campaign_stages(*stages: str) -> None:
    """Refuse stage-specific campaign work excluded by ``config.yaml``."""
    configured = set(config.campaign_stages())
    missing = [stage for stage in stages if stage not in configured]
    if missing:
        raise SystemExit(
            "Campaign stage(s) disabled in config.yaml: " + ", ".join(missing)
        )


def _filtered_cells_from_args(args) -> list[Cell]:
    cells = list(iter_cells())
    source = _resolve_model_arg(getattr(args, "source", None))
    target = _resolve_model_arg(getattr(args, "target", None))
    dataset = getattr(args, "dataset", None)
    seed = getattr(args, "seed", None)
    if source is not None:
        cells = [c for c in cells if c.source == source]
    if target is not None:
        cells = [c for c in cells if c.target == target]
    if dataset is not None:
        cells = [c for c in cells if c.dataset == dataset]
    if seed is not None:
        cells = [c for c in cells if c.seed == seed]
    if getattr(args, "only_llama", False):
        cells = [c for c in cells if c.llama_critical]
    max_cells = getattr(args, "max_cells", None)
    if max_cells is not None:
        cells = cells[:max_cells]
    return cells


def _add_cell_filter_args(p) -> None:
    p.add_argument("--source", default=None, help="Source model id or slug")
    p.add_argument("--target", default=None, help="Target model id or slug")
    p.add_argument("--dataset", choices=list(TRAIN_DATASETS), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--only-llama", action="store_true")
    p.add_argument("--max-cells", type=int, default=None)


def _add_safety_data_args(p) -> None:
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--safety-prompts-file", type=Path, default=DEFAULT_SAFETY_PROMPTS)
    p.add_argument("--exclude-prompts-file", type=Path, default=DEFAULT_SAFETY_EXCLUDE_PROMPTS)
    p.add_argument(
        "--include-eval-prompts",
        action="store_true",
        help="Do not exclude the default refusal eval prompts from replay data.",
    )
    p.add_argument("--safety-replay-size", type=int, default=DEFAULT_SAFETY_REPLAY_SIZE)
    p.add_argument("--safety-prompt-seed", type=int, default=42)
    p.add_argument("--safety-response-seed", type=int, default=1)
    _add_cell_filter_args(p)


def _safety_exclude_path(args) -> Path | None:
    return None if getattr(args, "include_eval_prompts", False) else args.exclude_prompts_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-cells", help="Print the matrix without launching anything")
    list_p.add_argument("--only-llama", action="store_true")

    self_list_p = sub.add_parser("list-self-sft-controls", help="Print self-SFT drift-control cells")
    self_list_p.add_argument("--models", nargs="*", default=MODELS)
    self_list_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS))

    gen_p = sub.add_parser(
        "generate-target-responses",
        help="Generate target-model responses on training prompts (SFT completions).",
    )
    gen_p.add_argument("--dry-run", action="store_true")
    gen_p.add_argument("--models", nargs="*", default=MODELS, help="Subset of models to run")
    gen_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS), help="Subset of datasets")
    gen_p.add_argument("--parallel", type=int, default=1, help="In-flight Tinker sample() requests per model")

    build_p = sub.add_parser("build-sft-data", help="Build SFT training CSVs from baselines.")
    build_p.add_argument("--dry-run", action="store_true")

    self_build_p = sub.add_parser("build-self-sft-data", help="Build self-SFT drift-control CSVs.")
    self_build_p.add_argument("--dry-run", action="store_true")

    sft_p = sub.add_parser("launch-sft", help="Submit SFT jobs to Tinker.")
    sft_p.add_argument("--dry-run", action="store_true")
    sft_p.add_argument("--only-llama", action="store_true")
    sft_p.add_argument("--max-jobs", type=int, default=None)
    sft_p.add_argument("--manifest-out", type=Path, default=None)
    sft_p.add_argument("--parallel", type=int, default=1, help="Number of concurrent SFT jobs (ThreadPoolExecutor)")
    # Cell filters prevent an unscoped command from submitting the whole configured matrix.
    # --only-llama already exists above, so these mirror _add_cell_filter_args minus that flag.
    sft_p.add_argument("--source", default=None, help="Source model id or slug (shard by source)")
    sft_p.add_argument("--target", default=None, help="Target model id or slug")
    sft_p.add_argument("--dataset", choices=list(TRAIN_DATASETS), default=None)
    sft_p.add_argument("--seed", type=int, default=None)
    sft_p.add_argument("--max-cells", type=int, default=None)

    self_sft_p = sub.add_parser("launch-self-sft", help="Submit self-SFT drift-control jobs to Tinker.")
    self_sft_p.add_argument("--dry-run", action="store_true")
    self_sft_p.add_argument("--max-jobs", type=int, default=None)
    self_sft_p.add_argument("--manifest-out", type=Path, default=None)
    self_sft_p.add_argument("--parallel", type=int, default=1)
    self_sft_p.add_argument("--models", nargs="*", default=MODELS)
    self_sft_p.add_argument("--datasets", nargs="*", default=list(TRAIN_DATASETS))
    self_sft_p.add_argument("--output-root", type=Path, default=SELF_SFT_OUTPUT_DIR)
    self_sft_p.add_argument("--registry-path", type=Path, default=None)

    self_mon_p = sub.add_parser("monitor-self-sft", help="Summarize self-SFT drift-control registry/log status.")
    self_mon_p.add_argument("--json", action="store_true")

    dpd_p = sub.add_parser("build-dpo-data", help="Build DPO preference CSVs from baselines.")
    dpd_p.add_argument("--dry-run", action="store_true")

    dpo_p = sub.add_parser("launch-dpo", help="Submit DPO jobs to Tinker (on top of SFT adapters).")
    dpo_p.add_argument("--dry-run", action="store_true")
    dpo_p.add_argument("--only-llama", action="store_true")
    dpo_p.add_argument("--max-jobs", type=int, default=None)
    dpo_p.add_argument("--manifest-out", type=Path, default=None)
    dpo_p.add_argument("--parallel", type=int, default=1)
    # Cell filters, mirroring launch-sft. Without them launch-dpo submits the WHOLE matrix and the
    # only thing stopping a re-submit is the registry skip -- which does not cover off-roster
    # targets, so a roster-scoped campaign pays Tinker for cells it will never report.
    dpo_p.add_argument("--source", default=None, help="Source model id or slug (shard by source)")
    dpo_p.add_argument("--target", default=None, help="Target model id or slug")
    dpo_p.add_argument("--dataset", choices=list(TRAIN_DATASETS), default=None)
    dpo_p.add_argument("--seed", type=int, default=None)
    dpo_p.add_argument("--max-cells", type=int, default=None)

    safety_list_p = sub.add_parser(
        "list-safety-cells",
        help="Print filtered safety-constrained imitation cells without launching anything.",
    )
    _add_cell_filter_args(safety_list_p)

    safety_sft_build_p = sub.add_parser(
        "build-safety-sft-data",
        help="Build safety-constrained SFT CSVs: target imitation + refusal replay.",
    )
    _add_safety_data_args(safety_sft_build_p)

    safety_dpo_build_p = sub.add_parser(
        "build-safety-dpo-data",
        help="Build safety-constrained DPO CSVs: imitation pairs + refusal replay pairs.",
    )
    _add_safety_data_args(safety_dpo_build_p)

    safety_sft_p = sub.add_parser(
        "launch-safety-sft",
        help="Submit safety-constrained SFT jobs.",
    )
    safety_sft_p.add_argument("--dry-run", action="store_true")
    safety_sft_p.add_argument("--manifest-out", type=Path, default=None)
    safety_sft_p.add_argument("--parallel", type=int, default=1)
    _add_cell_filter_args(safety_sft_p)

    safety_dpo_p = sub.add_parser(
        "launch-safety-dpo",
        help="Submit safety-constrained DPO jobs on top of safety_sft adapters.",
    )
    safety_dpo_p.add_argument("--dry-run", action="store_true")
    safety_dpo_p.add_argument("--manifest-out", type=Path, default=None)
    safety_dpo_p.add_argument("--parallel", type=int, default=1)
    _add_cell_filter_args(safety_dpo_p)

    cell_p = sub.add_parser(
        "launch-local-cell",
        help="Run ONE local (gemma-4) cell's LoRA SFT and/or DPO. Wrap in `accelerate launch` for multi-GPU.",
    )
    cell_p.add_argument("--source", required=True, help="Local-backend source model id (e.g. google/gemma-4-E4B-it)")
    cell_p.add_argument("--target", required=True, help="Target model id the source is disguised as")
    cell_p.add_argument("--dataset", required=True, choices=list(TRAIN_DATASETS))
    cell_p.add_argument("--seed", type=int, required=True)
    cell_p.add_argument("--phase", choices=["sft", "dpo", "all"], default="all")
    cell_p.add_argument("--per-device-batch-size", type=int, default=None,
                        help="Override per-device train batch size (lower for the 31B/26B models).")
    cell_p.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (preserve effective batch when lowering per-device).")
    cell_p.add_argument("--epochs", type=int, default=None, help="Override SFT epochs.")

    self_cell_p = sub.add_parser(
        "launch-local-self-sft-cell",
        help="Run ONE local diagonal self-SFT control with explicit output and registry roots.",
    )
    self_cell_p.add_argument("--source", required=True, help="Local-backend source model id")
    self_cell_p.add_argument("--dataset", required=True, choices=list(TRAIN_DATASETS))
    self_cell_p.add_argument("--seed", type=int, required=True)
    self_cell_p.add_argument("--output-root", type=Path, default=SELF_SFT_OUTPUT_DIR)
    self_cell_p.add_argument("--registry-path", type=Path, default=None)
    self_cell_p.add_argument("--per-device-batch-size", type=int, default=None)
    self_cell_p.add_argument("--grad-accum", type=int, default=None)
    self_cell_p.add_argument("--epochs", type=int, default=None)

    bf_p = sub.add_parser(
        "backfill-export",
        help="Download already-registered SFT adapters to local PEFT.",
    )
    bf_p.add_argument("--only-llama", action="store_true")

    bd_p = sub.add_parser(
        "backfill-register-dpo",
        help="Scan dpo_runs/*/logs/checkpoints.jsonl and register DPO adapter URIs.",
    )

    hf_p = sub.add_parser(
        "push-to-hf",
        help="Upload all registered adapters to HuggingFace Hub.",
    )
    hf_p.add_argument("--namespace", default="dementor-research", help="HF user or org name")
    hf_p.add_argument("--only-llama", action="store_true")
    hf_p.add_argument(
        "--kinds",
        nargs="+",
        default=config.campaign_stages(),
        choices=["sft", "dpo", "self_sft"],
    )
    hf_p.add_argument("--keep-local", action="store_true", help="Don't delete local PEFT after upload")
    hf_p.add_argument("--private", action="store_true", help="Create private repos (default: public)")
    hf_p.add_argument("--parallel", type=int, default=1, help="Concurrent uploads")

    args = parser.parse_args()

    if args.cmd == "list-cells":
        cells = list(iter_cells())
        if args.only_llama:
            cells = [c for c in cells if c.llama_critical]
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} {c.slug}")
        print(f"\nTotal cells: {len(cells)}")
        return 0

    if args.cmd == "list-self-sft-controls":
        cells = list(iter_self_sft_cells(models=args.models, datasets=args.datasets))
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} self_sft_{c.slug}")
        print(f"\nTotal self-SFT controls: {len(cells)}")
        return 0

    if args.cmd == "generate-target-responses":
        n = generate_target_responses(
            models=args.models,
            datasets=args.datasets,
            dry_run=args.dry_run,
            parallel=args.parallel,
        )
        print(f"\nGenerated {n} new responses")
        return 0

    if args.cmd == "build-sft-data":
        _require_campaign_stages("sft")
        n = build_sft_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} SFT training CSVs")
        return 0

    if args.cmd == "build-self-sft-data":
        _require_campaign_stages("self_sft")
        n = build_self_sft_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} self-SFT control CSVs")
        return 0

    if args.cmd == "launch-sft":
        _require_campaign_stages("sft")
        cells = _filtered_cells_from_args(args)
        print(f"launch-sft: {len(cells)} cells after filters "
              f"(source={args.source} target={args.target} dataset={args.dataset} seed={args.seed})")
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=args.only_llama,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
        )
        # A FILTERED or dry run must not clobber the canonical manifest: that file is the
        # campaign-wide plan other tooling reads to compute what is still pending, and overwriting
        # it with a small shard silently makes the campaign look almost finished.
        # An explicit --manifest-out always wins; otherwise a scoped run gets a scoped filename.
        _scope = [p for p in (args.source, args.target, args.dataset,
                              str(args.seed) if args.seed is not None else None) if p]
        if args.manifest_out:
            out_path = args.manifest_out
        elif _scope or args.dry_run:
            tag = "-".join(str(s).replace("/", "__") for s in _scope) or "all"
            suffix = "dryrun" if args.dry_run else "shard"
            out_path = DATA / "results" / "matrix" / f"sft_manifest.{suffix}.{tag}.json"
        else:
            out_path = DATA / "results" / "matrix" / "sft_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "launch-self-sft":
        _require_campaign_stages("self_sft")
        cells = list(iter_self_sft_cells(models=args.models, datasets=args.datasets))
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
            alias_prefix="self_sft",
            data_path_fn=self_sft_data_path,
            output_root=args.output_root,
            registry_path=args.registry_path,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "self_sft_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "monitor-self-sft":
        _require_campaign_stages("self_sft")
        result = monitor_self_sft_controls()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Self-SFT control status:")
            for key, value in sorted(result["counts"].items()):
                print(f"  {key}: {value}")
            for row in result["controls"]:
                print(f"  {row['status']:10s} {row['alias']}")
        return 0

    if args.cmd == "build-dpo-data":
        _require_campaign_stages("dpo")
        n = build_dpo_data(dry_run=args.dry_run)
        print(f"\nBuilt {n} DPO preference CSVs")
        return 0

    if args.cmd == "launch-dpo":
        _require_campaign_stages("dpo")
        cells = _filtered_cells_from_args(args)
        print(f"launch-dpo: {len(cells)} cells after filters "
              f"(source={args.source} target={args.target} dataset={args.dataset} seed={args.seed})")
        manifest = launch_dpo(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=args.only_llama,
            max_jobs=args.max_jobs,
            parallel=args.parallel,
        )
        # Same rule as launch-sft: a FILTERED or dry run must not clobber the canonical manifest.
        _scope = [p for p in (args.source, args.target, args.dataset,
                              str(args.seed) if args.seed is not None else None) if p]
        if args.manifest_out:
            out_path = args.manifest_out
        elif _scope or args.dry_run:
            tag = "-".join(str(x).replace("/", "__") for x in _scope) or "all"
            out_path = DATA / "results" / "matrix" / f"dpo_manifest.{'dryrun' if args.dry_run else 'shard'}.{tag}.json"
        else:
            out_path = DATA / "results" / "matrix" / "dpo_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        return 0

    if args.cmd == "list-safety-cells":
        cells = _filtered_cells_from_args(args)
        for c in cells:
            tag = "[LLAMA-CRIT]" if c.llama_critical else "           "
            print(f"  {tag} safety_sft/safety_dpo {c.slug}")
        print(f"\nTotal safety-constrained cells: {len(cells)}")
        return 0

    if args.cmd == "build-safety-sft-data":
        cells = _filtered_cells_from_args(args)
        n = build_safety_sft_data(
            dry_run=args.dry_run,
            cells=cells,
            safety_prompts_file=args.safety_prompts_file,
            exclude_prompts_file=_safety_exclude_path(args),
            replay_size=args.safety_replay_size,
            prompt_seed=args.safety_prompt_seed,
            response_seed=args.safety_response_seed,
        )
        print(f"\nBuilt {n} safety-constrained SFT training CSVs")
        return 0

    if args.cmd == "build-safety-dpo-data":
        cells = _filtered_cells_from_args(args)
        n = build_safety_dpo_data(
            dry_run=args.dry_run,
            cells=cells,
            safety_prompts_file=args.safety_prompts_file,
            exclude_prompts_file=_safety_exclude_path(args),
            replay_size=args.safety_replay_size,
            prompt_seed=args.safety_prompt_seed,
            response_seed=args.safety_response_seed,
        )
        print(f"\nBuilt {n} safety-constrained DPO preference CSVs")
        return 0

    if args.cmd == "launch-safety-sft":
        cells = _filtered_cells_from_args(args)
        manifest = launch_sft(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=False,
            max_jobs=None,
            parallel=args.parallel,
            alias_prefix="safety_sft",
            data_path_fn=safety_sft_data_path,
            output_root=SAFETY_SFT_OUTPUT_DIR,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "safety_sft_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "launch-safety-dpo":
        cells = _filtered_cells_from_args(args)
        manifest = launch_safety_dpo(
            cells=cells,
            dry_run=args.dry_run,
            only_llama=False,
            max_jobs=None,
            parallel=args.parallel,
        )
        out_path = args.manifest_out or (DATA / "results" / "matrix" / "safety_dpo_manifest.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest: {out_path}")
        print(f"Total jobs: {manifest['n_jobs']}")
        if args.dry_run:
            print("(dry-run — no jobs submitted)")
        return 0

    if args.cmd == "launch-local-cell":
        phases = ("sft", "dpo") if args.phase == "all" else (args.phase,)
        _require_campaign_stages(*phases)
        summary = launch_local_cell(
            source=args.source,
            target=args.target,
            dataset=args.dataset,
            seed=args.seed,
            phase=args.phase,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.cmd == "launch-local-self-sft-cell":
        _require_campaign_stages("self_sft")
        source = _resolve_model_arg(args.source)
        summary = launch_local_cell(
            source=source,
            target=source,
            dataset=args.dataset,
            seed=args.seed,
            phase="sft",
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
            self_sft=True,
            sft_output_root=args.output_root,
            registry_path=args.registry_path,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.cmd == "backfill-export":
        result = backfill_export_adapters(
            only_llama=args.only_llama,
            kinds=tuple(config.campaign_stages()),
        )
        print(f"\nExported {result['exported']}, skipped {result['skipped']}, errors {len(result['errors'])}")
        for err in result["errors"]:
            print(f"  ERROR {err['alias']}: {err['error']}")
        return 0 if not result["errors"] else 1

    if args.cmd == "backfill-register-dpo":
        result = backfill_register_dpo()
        print(f"\nRegistered {result['registered']} DPO adapters; {result['missing']} missing checkpoint logs")
        return 0

    if args.cmd == "push-to-hf":
        _require_campaign_stages(*args.kinds)
        result = push_adapters_to_hf(
            namespace=args.namespace,
            only_llama=args.only_llama,
            kinds=tuple(args.kinds),
            delete_local_after=not args.keep_local,
            private=args.private,
            parallel=args.parallel,
        )
        print(f"\nUploaded {result['uploaded']}, errors {len(result['errors'])}")
        for err in result["errors"][:10]:
            print(f"  ERROR {err['alias']}: {err['error']}")
        return 0 if not result["errors"] else 1

    parser.error("Unknown command")
    return 1
