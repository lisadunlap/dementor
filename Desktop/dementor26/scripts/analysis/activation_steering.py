from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .common import git_commit, normalize_comparison_df, write_json


def parse_strengths(value: str) -> list[float]:
    strengths = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not strengths:
        raise ValueError("At least one steering strength is required.")
    return strengths


def _resolve_device(device: str | None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _load_causal_lm(
    model_name: str,
    *,
    peft_adapter_path: str | Path | None = None,
    device: str | None = None,
    dtype: str = "auto",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype == "auto":
        if resolved_device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype != "default":
        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if peft_adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading PEFT adapters requires `peft`. Install it or pass a merged HF model path."
            ) from exc
        model = PeftModel.from_pretrained(model, str(peft_adapter_path))

    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def _candidate_layer_containers(model: Any) -> list[Any]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
        ("language_model", "model", "layers"),
    ]
    out = []
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            out.append(obj)
    return out


def get_transformer_layers(model: Any) -> Any:
    for layers in _candidate_layer_containers(model):
        if len(layers) > 0:
            return layers
    raise ValueError("Could not locate transformer block list on this model.")


def resolve_layer_index(layer: int, n_layers: int) -> int:
    if layer < 0:
        layer = n_layers + layer
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Layer index {layer} out of range for {n_layers} layers.")
    return layer


def _format_pair_text(prompt: str, response: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


def _format_generation_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    return f"{system_prompt.strip()}\n\n{prompt}" if system_prompt else prompt


def _pool_hidden(hidden: Any, attention_mask: Any, mode: str):
    import torch

    if mode == "last":
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError("--pool must be 'last' or 'mean'")


def collect_layer_activations(
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    layer: int,
    device: str,
    batch_size: int,
    max_length: int,
    pool: str,
) -> np.ndarray:
    import torch

    rows = []
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
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[layer].float()
            rows.append(_pool_hidden(hidden, inputs["attention_mask"], pool).cpu().numpy())
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def compute_steering_vector(
    *,
    df: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    layer: int,
    device: str,
    batch_size: int,
    max_length: int,
    pool: str,
) -> dict[str, np.ndarray | float]:
    source_texts = [
        _format_pair_text(prompt, response)
        for prompt, response in zip(df["prompt"].astype(str), df["source_response"].astype(str))
    ]
    target_texts = [
        _format_pair_text(prompt, response)
        for prompt, response in zip(df["prompt"].astype(str), df["target_response"].astype(str))
    ]
    source = collect_layer_activations(
        texts=source_texts,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pool=pool,
    )
    target = collect_layer_activations(
        texts=target_texts,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pool=pool,
    )
    pair_deltas = target - source
    vector = pair_deltas.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        unit_vector = vector / norm
    else:
        unit_vector = vector
    return {
        "vector": vector.astype(np.float32),
        "unit_vector": unit_vector.astype(np.float32),
        "vector_norm": norm,
        "source_mean": source.mean(axis=0).astype(np.float32),
        "target_mean": target.mean(axis=0).astype(np.float32),
    }


@contextmanager
def steering_hook(
    *,
    model: Any,
    layer: int,
    vector: np.ndarray,
    strength: float,
    token_position: str,
    device: str,
) -> Iterator[None]:
    import torch

    layers = get_transformer_layers(model)
    layer = resolve_layer_index(layer, len(layers))
    steer = torch.as_tensor(vector, dtype=torch.float32, device=device) * float(strength)

    def _apply(hidden):
        steer_cast = steer.to(dtype=hidden.dtype)
        if hidden.ndim == 3:
            edited = hidden.clone()
            if token_position == "last":
                edited[:, -1, :] = edited[:, -1, :] + steer_cast
            elif token_position == "all":
                edited = edited + steer_cast.view(1, 1, -1)
            else:
                raise ValueError("--token-position must be 'last' or 'all'")
            return edited
        if hidden.ndim == 2:
            return hidden + steer_cast.view(1, -1)
        return hidden

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (_apply(output[0]), *output[1:])
        return _apply(output)

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_responses(
    *,
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    system_prompt: str | None,
) -> list[str]:
    import torch

    outputs: list[str] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            rendered = [_format_generation_prompt(tokenizer, prompt, system_prompt) for prompt in chunk]
            inputs = tokenizer(rendered, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = inputs["input_ids"].shape[1]
            for row in generated:
                completion = row[prompt_width:]
                outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def run_activation_steering(
    *,
    comparison_csv: str | Path,
    output_dir: str | Path,
    model_name: str,
    peft_adapter_path: str | Path | None = None,
    source_responses: str | Path | None = None,
    source_col: str = "source_response",
    disguised_col: str = "model_response",
    target_col: str = "target_response",
    layer: int = -8,
    strengths: str = "0,0.5,1,2",
    vector_output: str | Path | None = None,
    vector_input: str | Path | None = None,
    max_rows: int | None = None,
    batch_size: int = 1,
    activation_batch_size: int = 2,
    max_length: int = 1024,
    max_new_tokens: int = 256,
    pool: str = "last",
    token_position: str = "last",
    system_prompt: str | None = None,
    device: str | None = None,
    dtype: str = "auto",
) -> dict:
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

    tokenizer, model, resolved_device = _load_causal_lm(
        model_name,
        peft_adapter_path=peft_adapter_path,
        device=device,
        dtype=dtype,
    )
    layers = get_transformer_layers(model)
    layer = resolve_layer_index(layer, len(layers))

    if vector_input:
        payload = np.load(vector_input)
        vector = payload["vector"].astype(np.float32)
        vector_norm = float(np.linalg.norm(vector))
    else:
        payload = compute_steering_vector(
            df=df,
            tokenizer=tokenizer,
            model=model,
            layer=layer + 1,
            device=resolved_device,
            batch_size=activation_batch_size,
            max_length=max_length,
            pool=pool,
        )
        # Use the raw (unnormalized) mean pair-delta as the steering vector so
        # that `--strengths` are interpretable: strength=1.0 reproduces the full
        # average source->target shift at the model's natural activation scale.
        # The unit vector is still saved alongside for downstream cosine work.
        vector = payload["vector"].astype(np.float32)  # type: ignore[union-attr]
        vector_norm = float(payload["vector_norm"])  # type: ignore[arg-type]
        if vector_output:
            np.savez_compressed(
                vector_output,
                vector=vector,
                unit_vector=payload["unit_vector"],
                source_mean=payload["source_mean"],
                target_mean=payload["target_mean"],
                layer=np.asarray([layer], dtype=int),
                vector_norm=np.asarray([vector_norm], dtype=np.float32),
            )

    summaries = []
    for strength in parse_strengths(strengths):
        with steering_hook(
            model=model,
            layer=layer,
            vector=vector,
            strength=strength,
            token_position=token_position,
            device=resolved_device,
        ):
            responses = generate_responses(
                prompts=df["prompt"].astype(str).tolist(),
                tokenizer=tokenizer,
                model=model,
                device=resolved_device,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                system_prompt=system_prompt,
            )
        out = pd.DataFrame(
            {
                "prompt": df["prompt"].astype(str),
                "source_response": df["source_response"].astype(str),
                "model_response": responses,
                "target_response": df["target_response"].astype(str),
                "method": "activation_steering",
                "source_model": model_name,
                "target_model": df["target_model"].iloc[0] if "target_model" in df.columns and len(df) else None,
                "steering_layer": layer,
                "steering_strength": strength,
                "peft_adapter_path": str(peft_adapter_path) if peft_adapter_path else None,
            }
        )
        safe_strength = str(strength).replace("-", "neg").replace(".", "p")
        csv_path = out_dir / f"activation_steering_layer{layer}_strength{safe_strength}.csv"
        out.to_csv(csv_path, index=False)
        summaries.append({"strength": strength, "output_csv": str(csv_path)})

    summary = {
        "model_name": model_name,
        "peft_adapter_path": str(peft_adapter_path) if peft_adapter_path else None,
        "comparison_csv": str(comparison_csv),
        "n": int(len(df)),
        "layer": int(layer),
        "n_layers": int(len(layers)),
        "strengths": parse_strengths(strengths),
        "vector_norm": vector_norm,
        "pool": pool,
        "token_position": token_position,
        "outputs": summaries,
        "git_commit": git_commit(),
        "note": (
            "This runner requires local Transformers hooks. For Tinker-trained adapters, "
            "first export the tinker:// sampler weights to a local PEFT adapter or merged HF model."
        ),
    }
    write_json(out_dir / "activation_steering_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and inject activation steering vectors during HF generation.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--peft-adapter-path")
    parser.add_argument("--source-responses")
    parser.add_argument("--source-col", default="source_response")
    parser.add_argument("--disguised-col", default="model_response")
    parser.add_argument("--target-col", default="target_response")
    parser.add_argument("--layer", type=int, default=-8)
    parser.add_argument("--strengths", default="0,0.5,1,2")
    parser.add_argument("--vector-output")
    parser.add_argument("--vector-input")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--activation-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--token-position", choices=["last", "all"], default="last")
    parser.add_argument("--system-prompt")
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="auto", help="auto, default, float32, float16, bfloat16, etc.")
    args = parser.parse_args()

    summary = run_activation_steering(
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        peft_adapter_path=args.peft_adapter_path,
        source_responses=args.source_responses,
        source_col=args.source_col,
        disguised_col=args.disguised_col,
        target_col=args.target_col,
        layer=args.layer,
        strengths=args.strengths,
        vector_output=args.vector_output,
        vector_input=args.vector_input,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        activation_batch_size=args.activation_batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        pool=args.pool,
        token_position=args.token_position,
        system_prompt=args.system_prompt,
        device=args.device,
        dtype=args.dtype,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
