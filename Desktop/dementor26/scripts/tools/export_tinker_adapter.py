from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tinker sampler weights and export them for local Transformers/vLLM use."
    )
    parser.add_argument(
        "--tinker-path",
        required=True,
        help="Tinker sampler path, e.g. tinker://<run-id>/sampler_weights/<name>.",
    )
    parser.add_argument("--base-model", required=True, help="HF base model used for the Tinker run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=["peft", "merged"],
        default="peft",
        help="Export a lightweight PEFT adapter or a merged HuggingFace model.",
    )
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--dtype", default="bfloat16", help="Used only for merged export.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--merge-strategy", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tinker_cookbook import weights
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Exporting Tinker adapters requires `tinker_cookbook`. "
            "Install it before running this helper."
        ) from exc

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    download_dir = args.download_dir or (args.output_dir.parent / f"{args.output_dir.name}_download")
    adapter_dir = weights.download(tinker_path=args.tinker_path, output_dir=str(download_dir))

    if args.format == "peft":
        if not hasattr(weights, "build_lora_adapter"):
            raise AttributeError(
                "This tinker_cookbook version does not expose weights.build_lora_adapter(). "
                "Upgrade tinker_cookbook or export with --format merged."
            )
        weights.build_lora_adapter(
            base_model=args.base_model,
            adapter_path=str(adapter_dir),
            output_path=str(args.output_dir),
        )
    else:
        weights.build_hf_model(
            base_model=args.base_model,
            adapter_path=str(adapter_dir),
            output_path=str(args.output_dir),
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            merge_strategy=args.merge_strategy,
        )

    metadata = {
        "tinker_path": args.tinker_path,
        "base_model": args.base_model,
        "download_dir": str(download_dir),
        "output_dir": str(args.output_dir),
        "format": args.format,
    }
    (args.output_dir / "dementor_tinker_export.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
