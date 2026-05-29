from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .behavioral_inertia_metrics import train_source_target_probe
from .common import CONDITIONS, condition_texts, git_commit, normalize_comparison_df, write_json


def _parse_layers(value: str | None, n_hidden_states: int) -> list[int]:
    if not value:
        raw = [0, n_hidden_states // 4, n_hidden_states // 2, (3 * n_hidden_states) // 4, n_hidden_states - 1]
    elif value == "all":
        raw = list(range(n_hidden_states))
    else:
        raw = [int(part.strip()) for part in value.split(",") if part.strip()]
    out = []
    for idx in raw:
        if idx < 0:
            idx = n_hidden_states + idx
        idx = max(0, min(n_hidden_states - 1, idx))
        if idx not in out:
            out.append(idx)
    return out


def _load_encoder(model_name: str, device: str | None = None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if resolved_device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(resolved_device)
    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, resolved_device


def _format_text(prompt: str, response: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


def encode_hidden_states(
    texts: list[str],
    *,
    model_name: str,
    layers: str | None = None,
    batch_size: int = 4,
    max_length: int = 1024,
    device: str | None = None,
) -> tuple[np.ndarray, list[int]]:
    import torch

    tokenizer, model, resolved_device = _load_encoder(model_name, device=device)
    layer_ids: list[int] | None = None
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            if layer_ids is None:
                layer_ids = _parse_layers(layers, len(hidden_states))
            mask = inputs["attention_mask"].unsqueeze(-1)
            layer_vecs = []
            for layer_idx in layer_ids:
                hidden = hidden_states[layer_idx].float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                layer_vecs.append(pooled.cpu().numpy())
            batches.append(np.stack(layer_vecs, axis=1))
    if not batches:
        return np.zeros((0, 0, 0), dtype=np.float32), []
    assert layer_ids is not None
    return np.concatenate(batches, axis=0), layer_ids


def _build_texts(df: pd.DataFrame) -> dict[str, list[str]]:
    texts = {}
    for cond in CONDITIONS:
        response_col = f"{cond}_response"
        texts[cond] = [
            _format_text(prompt, response)
            for prompt, response in zip(df["prompt"].astype(str), df[response_col].astype(str))
        ]
    return texts


def run_activation_bridge(
    *,
    comparison_csv: str | Path,
    output_dir: str | Path,
    mode: str = "fixed-encoder",
    encoder_model: str = "intfloat/e5-small-v2",
    source_model: str | None = None,
    target_model: str | None = None,
    source_responses: str | Path | None = None,
    source_col: str = "source_response",
    disguised_col: str = "model_response",
    target_col: str = "target_response",
    layers: str | None = None,
    batch_size: int = 4,
    max_length: int = 1024,
    max_rows: int | None = None,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    if mode not in {"fixed-encoder", "native-probe"}:
        raise ValueError("--mode must be fixed-encoder or native-probe")
    if mode == "native-probe":
        if not source_model:
            raise ValueError("--source-model is required for native-probe mode")
        encoder_model = source_model

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_comparison_df(
        comparison_csv,
        source_responses=source_responses,
        source_col=source_col,
        disguised_col=disguised_col,
        target_col=target_col,
    )
    if max_rows is not None:
        df = df.head(max_rows).copy()
    texts = _build_texts(df)

    condition_arrays = {}
    selected_layers: list[int] | None = None
    for cond in CONDITIONS:
        arr, layer_ids = encode_hidden_states(
            texts[cond],
            model_name=encoder_model,
            layers=layers,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        condition_arrays[cond] = arr
        selected_layers = layer_ids

    assert selected_layers is not None
    np.savez_compressed(
        out_dir / "activations.npz",
        source=condition_arrays["source"],
        disguised=condition_arrays["disguised"],
        target=condition_arrays["target"],
        layers=np.asarray(selected_layers, dtype=int),
    )

    rows = []
    for local_layer_idx, layer_id in enumerate(selected_layers):
        probe = train_source_target_probe(
            condition_arrays["source"][:, local_layer_idx, :],
            condition_arrays["target"][:, local_layer_idx, :],
            condition_arrays["disguised"][:, local_layer_idx, :],
            seed=seed,
        )
        rows.append({"layer": int(layer_id), **probe})
    layer_df = pd.DataFrame(rows)
    layer_df.to_csv(out_dir / "activation_layer_curve.csv", index=False)
    layer_df.to_csv(out_dir / "activation_probe_results.csv", index=False)

    if not layer_df.empty and "probe_cv" in layer_df:
        best_idx = layer_df["probe_cv"].fillna(-1).idxmax()
        best = layer_df.loc[best_idx].to_dict()
    else:
        best = {}
    summary = {
        "activation_bridge_mode": mode,
        "encoder_model": encoder_model,
        "source_model": source_model,
        "target_model": target_model,
        "n": int(len(df)),
        "layers": [int(x) for x in selected_layers],
        "best_layer": int(best.get("layer")) if best else None,
        "probe_cv": float(best.get("probe_cv")) if best else None,
        "activation_source_prob": float(best.get("mean_source_prob")) if best else None,
        "activation_target_prob": float(best.get("mean_target_prob")) if best else None,
        "git_commit": git_commit(),
        "interpretation_note": (
            "fixed-encoder uses a shared measurement model for cross-family comparisons"
            if mode == "fixed-encoder"
            else "native-probe uses the source model as encoder and reports scalar probe metrics"
        ),
    }
    write_json(out_dir / "activation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run activation bridge probes for behavioral inertia.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["fixed-encoder", "native-probe"], default="fixed-encoder")
    parser.add_argument("--encoder-model", default="intfloat/e5-small-v2")
    parser.add_argument("--source-model")
    parser.add_argument("--target-model")
    parser.add_argument("--source-responses")
    parser.add_argument("--source-col", default="source_response")
    parser.add_argument("--disguised-col", default="model_response")
    parser.add_argument("--target-col", default="target_response")
    parser.add_argument("--layers", help="Comma-separated hidden-state indices, 'all', or omitted for quartiles.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()

    summary = run_activation_bridge(
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir,
        mode=args.mode,
        encoder_model=args.encoder_model,
        source_model=args.source_model,
        target_model=args.target_model,
        source_responses=args.source_responses,
        source_col=args.source_col,
        disguised_col=args.disguised_col,
        target_col=args.target_col,
        layers=args.layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_rows=args.max_rows,
        seed=args.seed,
        device=args.device,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
