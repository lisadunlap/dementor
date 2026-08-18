#!/usr/bin/env python3
"""Materialize the exact sum of two plain LoRA adapters.

This is primarily used by the vLLM erosion evaluator.  Local DPO training starts
from a base model with its SFT adapter merged, then trains a fresh DPO LoRA.  At
inference time the resulting model is therefore::

    W + scale_sft * B_sft @ A_sft + scale_dpo * B_dpo @ A_dpo

The same update can be represented without loading ``W`` by concatenating the A
matrices by rows and the scaled B matrices by columns.  The emitted adapter uses
``alpha == rank`` (unit PEFT scaling), so it is directly loadable by vLLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _load_config(adapter_dir: Path) -> dict:
    path = adapter_dir / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing PEFT config: {path}")
    return json.loads(path.read_text())


def _validate_plain_lora(left: dict, right: dict) -> None:
    for name, cfg in (("left", left), ("right", right)):
        if cfg.get("peft_type") != "LORA":
            raise ValueError(f"{name} adapter is not LoRA")
        unsupported = {
            "rank_pattern": cfg.get("rank_pattern"),
            "alpha_pattern": cfg.get("alpha_pattern"),
            "modules_to_save": cfg.get("modules_to_save"),
            "use_dora": cfg.get("use_dora", False),
            "use_qalora": cfg.get("use_qalora", False),
            "use_rslora": cfg.get("use_rslora", False),
        }
        bad = {key: value for key, value in unsupported.items() if value}
        if bad:
            raise ValueError(f"{name} adapter uses unsupported LoRA features: {bad}")
        if cfg.get("bias", "none") != "none" or cfg.get("lora_bias", False):
            raise ValueError(f"{name} adapter contains LoRA bias parameters")

    comparable = ("base_model_name_or_path", "fan_in_fan_out", "target_modules", "task_type")
    mismatches = [key for key in comparable if left.get(key) != right.get(key)]
    if mismatches:
        raise ValueError(f"Adapter configs disagree on: {', '.join(mismatches)}")


def _weights_path(adapter_dir: Path) -> Path:
    for name in WEIGHTS_NAMES:
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No PEFT adapter weights found in {adapter_dir}")


def _load_weights(adapter_dir: Path) -> dict:
    import torch

    path = _weights_path(adapter_dir)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def compose_lora_adapters(left_dir: str | Path, right_dir: str | Path,
                          output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Write an exact additive composition of two ordinary PEFT LoRAs."""
    import torch
    from safetensors.torch import save_file

    left_dir, right_dir, output_dir = map(Path, (left_dir, right_dir, output_dir))
    left_cfg, right_cfg = _load_config(left_dir), _load_config(right_dir)
    _validate_plain_lora(left_cfg, right_cfg)
    left, right = _load_weights(left_dir), _load_weights(right_dir)
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))[:5]
        only_right = sorted(set(right) - set(left))[:5]
        raise ValueError(f"Adapter tensors differ (left-only={only_left}, right-only={only_right})")

    allowed = (".lora_A.weight", ".lora_B.weight")
    unexpected = sorted(key for key in left if not key.endswith(allowed))
    if unexpected:
        raise ValueError(f"Unsupported non-LoRA tensors: {unexpected[:5]}")

    scale_left = float(left_cfg["lora_alpha"]) / float(left_cfg["r"])
    scale_right = float(right_cfg["lora_alpha"]) / float(right_cfg["r"])
    output: dict[str, torch.Tensor] = {}
    ranks: set[int] = set()
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        if key.endswith(".lora_A.weight"):
            if a.shape[1:] != b.shape[1:]:
                raise ValueError(f"Incompatible A tensors for {key}: {a.shape} vs {b.shape}")
            output[key] = torch.cat((a, b), dim=0).contiguous()
            ranks.add(a.shape[0] + b.shape[0])
        else:
            if a.shape[:-1] != b.shape[:-1]:
                raise ValueError(f"Incompatible B tensors for {key}: {a.shape} vs {b.shape}")
            output[key] = torch.cat((a * scale_left, b * scale_right), dim=-1).contiguous()

    if len(ranks) != 1:
        raise ValueError(f"Combined adapter has non-uniform ranks: {sorted(ranks)}")
    combined_rank = ranks.pop()
    cfg = dict(left_cfg)
    cfg.update({"r": combined_rank, "lora_alpha": combined_rank, "inference_mode": True,
                "rank_pattern": {}, "alpha_pattern": {}})

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    save_file(output, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (output_dir / CONFIG_NAME).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    provenance = {
        "composition": "exact_additive_lora_cat",
        "left_adapter": str(left_dir.resolve()),
        "right_adapter": str(right_dir.resolve()),
        "left_scale": scale_left,
        "right_scale": scale_right,
        "combined_rank": combined_rank,
    }
    (output_dir / "composition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First PEFT LoRA (the SFT parent)")
    parser.add_argument("right", type=Path, help="Second PEFT LoRA (the DPO adapter)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(compose_lora_adapters(args.left, args.right, args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
