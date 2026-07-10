"""Activation-steering disguise rung (Rung 5): projection-ablation + additive.

Productionized, tested, reusable version of the validated scratch implementation
(``/data/ethantsliu/steering/{derive_vector,steer_gen}.py``). This is the *scorable
rung*: it derives a per-layer source->target steering vector, runs a forward-hook
generator (projection-ablation OR additive) on the SOURCE model, and writes a
``{prompt, model_response}`` CSV that ``run_behavioral_cell`` scores alongside sft/dpo.

Why a separate module from ``activation_steering.py``: that module is an additive-only
sweep tool (last-token / "Prompt:\\nResponse:" pooling). The validated recipe differs in
two load-bearing ways, both reproduced here:

  1. **Derivation** — the source (Qwen2.5-7B, 3584-d) and target (Qwen3-8B, 4096-d) live
     in DIFFERENT residual spaces, so a literal ``mean(target_hidden) - mean(source_hidden)``
     is impossible. Since steering edits the SOURCE forward pass, the vector must live in
     SOURCE residual space: we teacher-force ``[chat_prompt + response]`` for BOTH response
     sets through the SOURCE model and difference their per-layer means over RESPONSE tokens:

         v_L = mean_p act_src(prompt_p + target_resp_p)[L] - mean_p act_src(prompt_p + source_resp_p)[L]

     ``hidden_states[L+1]`` is the output of decoder layer ``L`` (so ``v_L`` is applied by a
     forward hook on ``layers[L]`` — same index).

  2. **Operator** — projection-ablation ``h <- h - beta*(h.v_hat) v_hat`` (erase the source
     fingerprint direction) is the regime that works; additive ``h <- h + alpha*v_hat`` has
     no clean disguise regime. Both are supported via ``mode="ablate"|"add"``.

Validated result to reproduce (Qwen2.5-7B-Instruct -> Qwen3-8B, layer 14 of 28):
    beta 0.6 -> persistence ~0.43 (coherent) · beta 0.7 -> ~0.35 (coherent, beats DPO ~0.38)
    · beta 0.8 -> ~0.09 (~= real target) · beta >= 1.0 -> catastrophic collapse
The ``over_assimilation`` cell flag (not perplexity) is what catches the degenerate collapse.

torch/transformers imports are kept lazy (like ``training/local_backend.py``) so analysis
code can import this module without a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

# Architecture-agnostic decoder-layer discovery + index resolution and the shared HF
# causal-LM loader now live in `_common` (they locate the `*.layers.<int>` ModuleList
# generically, in the same spirit as training.local_backend.fsdp_wrap_layer_names -- no
# hardcoded layer class). This previously imported them from `activation_steering`, which
# was a backwards dependency (this is the newer production module).
from dementor.steering._common import get_transformer_layers, load_causal_lm, resolve_layer_index

__all__ = [
    "derive_steering_vector",
    "generate_steered_responses",
    "make_additive_hook",
    "make_ablation_hook",
    "save_steering_vectors",
    "load_steering_vectors",
    "run_steering_rung",
]


# ---------------------------------------------------------------------------
# Forward hooks (pure; CPU-testable). Each edits ``output[0]`` (or a bare tensor)
# and preserves the rest of a HF decoder block's output tuple.
# ---------------------------------------------------------------------------
def make_additive_hook(vector: Any, alpha: float):
    """Additive steering hook: ``h <- h + alpha * v_hat`` at every token.

    ``alpha`` is in residual-norm units (the added vector has L2 norm ``alpha``).
    """
    import torch

    v = torch.as_tensor(vector)
    v_unit = v / v.norm()

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        add = (float(alpha) * v_unit).to(device=h.device, dtype=h.dtype)
        h_new = h + add
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return hook


def make_ablation_hook(vector: Any, beta: float):
    """Projection-ablation hook: ``h <- h - beta * (h . v_hat) v_hat`` at every token.

    Removes the source->target fingerprint component. ``beta=1`` fully zeroes the
    component along ``v_hat``; ``beta>1`` overshoots into the anti-target half-space.
    Post-hook projection satisfies ``h_new . v_hat == (1 - beta) * (h . v_hat)``.
    Computed in fp32 then cast back to the block dtype.
    """
    import torch

    v = torch.as_tensor(vector)
    v_unit_f = (v / v.norm()).float()

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        v_cast = v_unit_f.to(device=hf.device)
        coef = (hf @ v_cast).unsqueeze(-1)  # projection coefficient, [..., 1]
        h_new = (hf - float(beta) * coef * v_cast).to(h.dtype)
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new

    return hook


# ---------------------------------------------------------------------------
# Vector derivation: per-layer source->target diff-of-means over response tokens.
# ---------------------------------------------------------------------------
def _load_causal_encoder(model_name: str, device: str | None, dtype: str):
    """Load an HF causal LM + tokenizer for teacher-forced activation capture.

    padding_side='right' so the exact response span ``[p_len:tot]`` can be sliced.
    """
    tokenizer, model, resolved_device = load_causal_lm(
        model_name, padding_side="right", device=device, dtype=dtype
    )
    n_layers = len(get_transformer_layers(model))
    return tokenizer, model, resolved_device, n_layers


def _collect_response_means(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    layers: list[int],
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[int, Any]:
    """Mean residual over RESPONSE tokens, per requested decoder layer.

    Teacher-forces ``[chat_prompt + response]`` and averages ``hidden_states[L+1]``
    (output of decoder layer ``L``) over the response-token span, then over prompts.
    Returns ``{L: Tensor[hidden]}`` (fp64 accumulation, returned fp32 on CPU).
    """
    import torch

    def build(prompt: str, response: str):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        p_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[:max_length]
        p_len = min(len(p_ids), max_length)
        return ids, p_len, len(ids)

    n = len(prompts)
    sums = {L: None for L in layers}
    count = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            specs = [build(prompts[i], responses[i]) for i in range(start, min(start + batch_size, n))]
            maxlen = max(s[2] for s in specs)
            input_ids = torch.full((len(specs), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(specs), maxlen), dtype=torch.long)
            for j, (ids, _p_len, tot) in enumerate(specs):
                input_ids[j, :tot] = torch.tensor(ids, dtype=torch.long)
                attn[j, :tot] = 1
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            hidden_states = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False
            ).hidden_states
            for j, (_ids, p_len, tot) in enumerate(specs):
                if tot <= p_len:  # empty-response guard
                    continue
                for L in layers:
                    vec = hidden_states[L + 1][j, p_len:tot, :].mean(0).double().cpu()
                    sums[L] = vec if sums[L] is None else sums[L] + vec
                count += 1
    if count == 0:
        raise ValueError("No non-empty responses to derive a steering vector from.")
    return {L: (sums[L] / count).float() for L in layers}


def derive_steering_vector(
    source_model: str,
    target_model: str | None = None,
    prompts: list[str] | None = None,
    *,
    source_responses: list[str] | None = None,
    target_responses: list[str] | None = None,
    layers: Iterable[int] | None = None,
    device: str | None = None,
    batch_size: int = 8,
    max_length: int = 2048,
    dtype: str = "auto",
    gen_max_new_tokens: int = 512,
) -> dict[int, Any]:
    """Per-layer source->target diff-of-means steering vector (SOURCE residual space).

    Both response sets are encoded with ``source_model`` (source/target have different
    hidden dims, so the vector must live in source space). For each decoder layer ``L``:

        v_L = mean_p act_src(prompt_p + target_resp_p)[L] - mean_p act_src(prompt_p + source_resp_p)[L]

    Args:
        source_model: HF id/path of the disguising (source) model; also the encoder.
        target_model: HF id/path of the target; only needed to *generate* target
            responses when ``target_responses`` is not supplied.
        prompts: derivation prompts (the train split).
        source_responses / target_responses: precomputed greedy responses aligned to
            ``prompts``. When ``None`` they are generated greedily from the respective
            model (via ``training.local_backend.generate_local_responses``).
        layers: decoder-layer indices to derive (negatives allowed). ``None`` -> all layers.
        device: e.g. ``"cuda"``. Defaults to cuda-if-available.

    Returns:
        ``{L: Tensor[hidden]}`` on CPU (fp32), keyed by decoder-layer index. Apply ``v_L``
        with a forward hook on ``get_transformer_layers(model)[L]``.
    """
    if prompts is None:
        raise ValueError("`prompts` is required.")
    if source_responses is None or target_responses is None:
        from dementor.training.local_backend import generate_local_responses

        if source_responses is None:
            source_responses = generate_local_responses(
                model=source_model, prompts=prompts, temperature=0.0, max_new_tokens=gen_max_new_tokens, device=device
            )
        if target_responses is None:
            if target_model is None:
                raise ValueError("Provide `target_responses` or a `target_model` to generate them.")
            target_responses = generate_local_responses(
                model=target_model, prompts=prompts, temperature=0.0, max_new_tokens=gen_max_new_tokens, device=device
            )
    if not (len(prompts) == len(source_responses) == len(target_responses)):
        raise ValueError("prompts, source_responses, target_responses must be the same length.")

    tokenizer, model, resolved_device, n_layers = _load_causal_encoder(source_model, device, dtype)
    if layers is None:
        layer_list = list(range(n_layers))
    else:
        layer_list = [resolve_layer_index(int(L), n_layers) for L in layers]

    src_means = _collect_response_means(
        model=model, tokenizer=tokenizer, prompts=list(map(str, prompts)),
        responses=list(map(str, source_responses)), layers=layer_list,
        device=resolved_device, batch_size=batch_size, max_length=max_length,
    )
    tgt_means = _collect_response_means(
        model=model, tokenizer=tokenizer, prompts=list(map(str, prompts)),
        responses=list(map(str, target_responses)), layers=layer_list,
        device=resolved_device, batch_size=batch_size, max_length=max_length,
    )
    return {L: (tgt_means[L] - src_means[L]).float() for L in layer_list}


def save_steering_vectors(vectors: Mapping[int, Any], path: str | Path, meta: dict | None = None) -> None:
    import torch

    payload = {"vectors": {int(k): v.cpu() for k, v in vectors.items()}}
    payload["vnorm"] = {int(k): float(v.norm()) for k, v in vectors.items()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, str(path))


def load_steering_vectors(path: str | Path) -> dict[int, Any]:
    """Load ``{L: Tensor}``. Accepts a bare mapping or a ``{"vectors": {...}}`` payload
    (compatible with the scratch ``vector.pt``)."""
    import torch

    obj = torch.load(str(path), map_location="cpu")
    vectors = obj["vectors"] if isinstance(obj, dict) and "vectors" in obj else obj
    return {int(k): v for k, v in vectors.items()}


# ---------------------------------------------------------------------------
# Steered generation.
# ---------------------------------------------------------------------------
def _resolve_vector(vector: Any, layer: int):
    import torch

    if isinstance(vector, Mapping):
        if layer not in vector:
            raise KeyError(f"No steering vector for layer {layer}; have {sorted(vector)}.")
        return torch.as_tensor(vector[layer])
    return torch.as_tensor(vector)


def _ensure_model_tokenizer(model: Any, tokenizer: Any, device: str | None, dtype: str):
    """Accept a loaded model (+tokenizer) or an HF id/path; return (model, tokenizer, device)."""
    import torch

    if isinstance(model, str):
        # Left-padded (decoder-only generation) load; a passed-in `tokenizer` is reused
        # (and still pad/padding-side normalized) rather than reloaded.
        tokenizer, model, resolved_device = load_causal_lm(
            model, padding_side="left", device=device, dtype=dtype, tokenizer=tokenizer
        )
        return model, tokenizer, resolved_device
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        raise ValueError("A tokenizer is required when passing a loaded model object.")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only generation
    return model, tokenizer, resolved_device


def generate_steered_responses(
    model: Any,
    prompts: list[str],
    vector: Any,
    layer: int,
    *,
    mode: str = "ablate",
    strength: float,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    tokenizer: Any = None,
    seed: int = 1,
    top_p: float = 0.95,
    batch_size: int = 16,
    device: str | None = None,
    dtype: str = "auto",
    verbose: bool = False,
) -> list[str]:
    """Generate with a steering hook on decoder ``layer`` of the SOURCE model.

    Args:
        model: a loaded HF causal LM, or an HF id/path (then ``tokenizer`` is optional).
        vector: ``{L: Tensor}`` mapping (uses ``vector[layer]``) or a single ``Tensor[hidden]``.
        mode: ``"ablate"`` -> ``h -= strength*(h.v_hat)v_hat`` (projection-ablation, the
            validated disguise operator); ``"add"`` -> ``h += strength*v_hat`` (additive).
        strength: ``beta`` for ablation, ``alpha`` for additive.
        temperature: ``>0`` samples (top_p), ``0`` is greedy.

    Returns:
        Decoded completions (prompt stripped), one per input prompt.
    """
    import torch

    if mode not in ("ablate", "add"):
        raise ValueError("mode must be 'ablate' or 'add'.")
    model, tokenizer, resolved_device = _ensure_model_tokenizer(model, tokenizer, device, dtype)

    layers = get_transformer_layers(model)
    layer = resolve_layer_index(int(layer), len(layers))
    v = _resolve_vector(vector, layer).to(resolved_device)
    hook = make_ablation_hook(v, strength) if mode == "ablate" else make_additive_hook(v, strength)
    handle = layers[layer].register_forward_hook(hook)

    try:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(p)}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        do_sample = temperature > 0
        torch.manual_seed(seed)
        responses: list[str] = [""] * len(prompts)
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = rendered[start : start + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(resolved_device)
                gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=False,
                                  pad_token_id=tokenizer.pad_token_id)
                if do_sample:
                    gen_kwargs.update(temperature=temperature, top_p=top_p)
                out = model.generate(**enc, **gen_kwargs)
                new = out[:, enc["input_ids"].shape[1] :]
                for j, text in enumerate(tokenizer.batch_decode(new, skip_special_tokens=True)):
                    responses[start + j] = text.strip()
                if verbose:
                    print(f"  [steer] gen {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return responses
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Scorable rung: write a {prompt, model_response} CSV (mirrors rung_sft.csv / rung_dpo.csv).
# ---------------------------------------------------------------------------
def run_steering_rung(
    *,
    source_model: str,
    out_csv: str | Path,
    eval_prompts_csv: str | Path,
    layer: int,
    strength: float,
    mode: str = "ablate",
    vector_path: str | Path | None = None,
    vectors: Mapping[int, Any] | None = None,
    train_prompts_csv: str | Path | None = None,
    source_responses_csv: str | Path | None = None,
    target_responses_csv: str | Path | None = None,
    target_model: str | None = None,
    save_vector_to: str | Path | None = None,
    temperature: float = 0.7,
    seed: int = 1,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    device: str | None = None,
    dtype: str = "auto",
) -> dict:
    """Derive-or-load a steering vector, generate steered eval responses, write the rung CSV.

    The output CSV has exactly ``prompt,model_response`` (same format as
    ``gen/rung_sft.csv`` / ``gen/rung_dpo.csv``) so ``run_behavioral_cell`` scores it as a
    disguise rung alongside sft/dpo. Provide the vector one of three ways:
    ``vectors`` (in-memory) > ``vector_path`` (a .pt) > derive from
    ``train_prompts_csv`` + response CSVs (or generated responses).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_prompts = pd.read_csv(eval_prompts_csv)["prompt"].astype(str).tolist()

    if vectors is None:
        if vector_path is not None:
            vectors = load_steering_vectors(vector_path)
        else:
            if train_prompts_csv is None:
                raise ValueError("Provide `vectors`, `vector_path`, or `train_prompts_csv` to derive.")
            train_prompts = pd.read_csv(train_prompts_csv)["prompt"].astype(str).tolist()
            src_resp = (
                pd.read_csv(source_responses_csv)["model_response"].astype(str).tolist()
                if source_responses_csv else None
            )
            tgt_resp = (
                pd.read_csv(target_responses_csv)["model_response"].astype(str).tolist()
                if target_responses_csv else None
            )
            vectors = derive_steering_vector(
                source_model, target_model, train_prompts,
                source_responses=src_resp, target_responses=tgt_resp,
                device=device, dtype=dtype,
            )
            if save_vector_to:
                save_steering_vectors(vectors, save_vector_to, meta={"source_model": source_model, "target_model": target_model})

    responses = generate_steered_responses(
        source_model, eval_prompts, vectors, layer,
        mode=mode, strength=strength, temperature=temperature, seed=seed,
        max_new_tokens=max_new_tokens, batch_size=batch_size, device=device, dtype=dtype, verbose=True,
    )
    df = pd.DataFrame({"prompt": eval_prompts, "model_response": responses})
    df["model_response"] = df["model_response"].fillna("").replace("", " ")
    df.to_csv(out_csv, index=False)
    n_empty = int((df["model_response"].str.strip() == "").sum())
    return {
        "out_csv": str(out_csv),
        "n": len(df),
        "layer": int(layer),
        "mode": mode,
        "strength": float(strength),
        "n_empty": n_empty,
        "median_chars": int(df["model_response"].str.len().median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation-steering disguise rung (projection-ablation / additive).")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model")
    parser.add_argument("--eval-prompts-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True, help="beta (ablate) or alpha (add).")
    parser.add_argument("--mode", choices=["ablate", "add"], default="ablate")
    parser.add_argument("--vector-path", help="Load a saved {L:Tensor} .pt (e.g. the scratch vector.pt).")
    parser.add_argument("--train-prompts-csv", help="Derivation prompts (if deriving the vector).")
    parser.add_argument("--source-responses-csv", help="Precomputed source greedy responses (prompt,model_response).")
    parser.add_argument("--target-responses-csv", help="Precomputed target greedy responses (prompt,model_response).")
    parser.add_argument("--save-vector-to")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    summary = run_steering_rung(
        source_model=args.source_model,
        out_csv=args.out_csv,
        eval_prompts_csv=args.eval_prompts_csv,
        layer=args.layer,
        strength=args.strength,
        mode=args.mode,
        vector_path=args.vector_path,
        train_prompts_csv=args.train_prompts_csv,
        source_responses_csv=args.source_responses_csv,
        target_responses_csv=args.target_responses_csv,
        target_model=args.target_model,
        save_vector_to=args.save_vector_to,
        temperature=args.temperature,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
