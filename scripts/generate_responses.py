from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

# Optional provider imports (only needed for basic generation)
try:
    from litellm import completion
    import litellm
except Exception:  # pragma: no cover - litellm only needed for basic mode
    completion = None  # type: ignore
    litellm = None  # type: ignore

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _read_prompts(prompts_file: Path, column: str = "prompt") -> List[str]:
    # Accept either CSV with a 'prompt' column or a .txt file with one prompt per line
    if prompts_file.suffix.lower() == ".txt":
        return [line.rstrip("\n") for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Default: CSV
    df = pd.read_csv(prompts_file)
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in CSV {prompts_file}")
    return df[column].astype(str).tolist()


def _reproduce_train_eval_split(
    dataset_csv: Path,
    *,
    prompt_column: str,
    train_size: int,
    eval_size: int,
    seed: int,
) -> tuple[List[str], List[str]]:
    import pandas as pd

    df = pd.read_csv(dataset_csv)
    if prompt_column not in df.columns:
        raise ValueError(f"Expected column '{prompt_column}' in CSV {dataset_csv}")
    dataset_size = train_size + eval_size
    if len(df) < dataset_size:
        raise ValueError(
            f"Requested dataset_size {dataset_size} but dataset only has {len(df)} rows."
        )
    sampled = df.sample(n=dataset_size, random_state=seed).reset_index(drop=True)
    train_df = sampled.iloc[:train_size].reset_index(drop=True)
    eval_df = sampled.iloc[train_size:].reset_index(drop=True)
    train_prompts = train_df[prompt_column].astype(str).tolist()
    eval_prompts = eval_df[prompt_column].astype(str).tolist()
    return train_prompts, eval_prompts


def _write_rows(output_csv: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "model_response", "model"])
            writer.writeheader()
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing(output_csv: Path) -> Dict[str, str]:
    if not output_csv.exists():
        return {}
    try:
        df = pd.read_csv(output_csv, quoting=csv.QUOTE_ALL)
    except Exception:
        return {}
    if {'prompt', 'model_response'}.issubset(df.columns):
        return {str(row['prompt']): str(row['model_response']) for _, row in df.iterrows()}
    return {}


def _get_cached_completion():
    try:
        from scripts.cache_llm import cached_completion  # type: ignore

        return cached_completion
    except Exception:
        return None


def _gen_litellm(
    *,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    api_base: Optional[str],
    api_key: Optional[str],
) -> str:
    if completion is None or litellm is None:
        raise RuntimeError("litellm is not installed. pip install litellm")

    if not hasattr(litellm, "cache") or litellm.cache is None:
        litellm.cache = litellm.Cache()

    cached_completion = _get_cached_completion()
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    if cached_completion:
        response = cached_completion(**kwargs)
        return response.choices[0].message.content or ""
    response = completion(**kwargs)
    return response["choices"][0]["message"]["content"]


def _gen_hf(model_id: str, prompt_text: str, max_tokens: int, temperature: float) -> str:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers not installed. pip install transformers accelerate") from exc

    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    out = pipe(prompt_text, max_new_tokens=max_tokens, do_sample=(temperature > 0), temperature=max(temperature, 1e-6))
    text = out[0]['generated_text']
    return text[len(prompt_text):].strip()


def _gen_vllm(model_id: str, prompt_text: str, max_tokens: int, temperature: float) -> str:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("vllm not installed. pip install vllm") from exc

    llm = LLM(model=model_id, trust_remote_code=True)
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate([prompt_text], params)
    return outputs[0].outputs[0].text


def _strip_chat_markup(text: str) -> str:
    """Remove common Llama-style chat markers for cleaner outputs."""
    cleaned = text
    for marker in (
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|assistant|>",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1]
            break
    for token in ("<|eot_id|>", "<|eom_id|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _gen_with_tinker(
    *,
    adapter_name: Optional[str],
    model_path: Optional[str],
    base_model: Optional[str],
    prompts: List[str],
    prompt_template: str,
    stop: Optional[List[str]],
    max_tokens: int,
    renderer_name: Optional[str],
    model_label: Optional[str] = None,
) -> List[dict]:
    import tinker
    from tinker import types

    service = tinker.ServiceClient()
    sampling = None
    if adapter_name:
        try:
            if hasattr(service, "get_sampling_client"):
                sampling = service.get_sampling_client(name=adapter_name)  # type: ignore[assignment]
            elif hasattr(service, "create_sampling_client"):
                # Some SDKs accept names positionally
                sampling = service.create_sampling_client(adapter_name)  # type: ignore[assignment]
        except Exception:
            sampling = None
    if sampling is None:
        if not model_path:
            raise ValueError("No valid adapter_name on server and no --model-path provided for Tinker backend.")
        sampling = service.create_sampling_client(model_path)
    # Obtain tokenizer compatible with the underlying base model
    if hasattr(sampling, "get_tokenizer"):
        tokenizer = sampling.get_tokenizer()  # type: ignore[assignment]
    else:
        bm = base_model or "meta-llama/Llama-3.1-8B-Instruct"
        training_for_tokenizer = service.create_lora_training_client(base_model=bm)
        tokenizer = training_for_tokenizer.get_tokenizer()

    def _render_prompt(raw_prompt: str) -> str:
        template_applied = prompt_template.format(prompt=raw_prompt)
        if renderer_name and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": template_applied}]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass
        return template_applied

    rows: List[dict] = []
    for p in prompts:
        text = _render_prompt(p)
        encoded = tokenizer.encode(text, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens=encoded)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=stop)
        future = sampling.sample(prompt=model_input, sampling_params=params, num_samples=1)
        result = future.result()
        reply = tokenizer.decode(result.sequences[0].tokens)
        if renderer_name:
            reply = _strip_chat_markup(reply)
        rows.append(
            {
                "prompt": p,
                "model_response": reply,
                "model": model_label or adapter_name or model_path,
            }
        )
    return rows


def _gen_with_openai(
    model: str,
    prompts: List[str],
    *,
    prompt_template: str,
    system_prompt: str,
    base_url: Optional[str],
    max_tokens: int,
    temperature: float,
) -> List[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=(base_url or "https://api.openai.com/v1"))
    rows: List[dict] = []
    for p in prompts:
        text = prompt_template.format(prompt=p)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        reply = resp.choices[0].message.content or ""
        rows.append({"prompt": p, "model_response": reply, "model": model})
    return rows


def _run_basic_generation(args: argparse.Namespace, prompts: List[str]) -> None:
    out_path = args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {} if args.overwrite else _load_existing(out_path)
    mode = "w" if args.overwrite or not out_path.exists() else "a"
    print(f"Writing {len(prompts)} prompts to {out_path} (mode={mode})")

    api_base = args.openai_api_base
    api_key = args.openai_api_key
    model_lower = args.model.lower()
    if model_lower.startswith("openai/gpt-") and not api_base:
        api_base = "https://api.openai.com/v1"

    with open(out_path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        if mode == "w":
            writer.writerow(["prompt", "model_response", "model"])

        for prompt in tqdm(prompts, desc=f"Generating with {args.model}"):
            if prompt in existing:
                continue

            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": prompt})

            sys_text = (args.system + "\n\n" if args.system else "")
            if model_lower.startswith("hf:"):
                model_id = args.model.split(":", 1)[1]
                text = _gen_hf(model_id, sys_text + prompt, args.max_tokens, args.temperature)
            elif model_lower.startswith("vllm:"):
                model_id = args.model.split(":", 1)[1]
                text = _gen_vllm(model_id, sys_text + prompt, args.max_tokens, args.temperature)
            else:
                text = _gen_litellm(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    api_base=api_base,
                    api_key=api_key,
                )

            writer.writerow([prompt, text, args.model])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses via finetuned adapters (Tinker/OpenAI) or generic provider/HF/vLLM backends."
        )
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Path to prompts file (CSV with 'prompt' column or .txt).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the outputs CSV.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Question: {prompt}\nAnswer:",
        help="Template applied to each prompt (available key: prompt).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional stop sequences; can specify multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of prompts to generate (after filtering).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output instead of appending/skipping existing prompts (basic mode only).",
    )

    # Optional: reproduce seed-42 split to exclude train prompts
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help=(
            "Original dataset CSV used for SFT/DPO (with a 'prompt' column). If set,"
            " we'll reproduce the same seed split and exclude the training prompts."
        ),
    )
    parser.add_argument(
        "--dataset-prompt-column",
        type=str,
        default="prompt",
        help="Prompt column name in --dataset-csv (default: prompt).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=300,
        help="Train size used during fine-tune (default: 300).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=200,
        help="Eval size used during fine-tune (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during fine-tune split (default: 42).",
    )

    sub = parser.add_subparsers(dest="backend", required=True)

    # Tinker backend
    tinker_p = sub.add_parser("tinker", help="Use a Tinker adapter name or model path")
    tinker_p.add_argument("--adapter-name", type=str, required=False, help="Adapter name saved on Tinker.")
    tinker_p.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Full Tinker model path (e.g., tinker://.../sampler_weights/final). Overrides --adapter-name if given.",
    )
    tinker_p.add_argument(
        "--adapter-registry",
        type=Path,
        default=Path("data/tinker_adapters.json"),
        help="Optional local registry JSON mapping adapter name -> model path.",
    )
    tinker_p.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model to fetch tokenizer if SamplingClient lacks one.",
    )
    tinker_p.add_argument(
        "--renderer-name",
        type=str,
        default=None,
        help="Optional renderer name (e.g., llama3) to mimic the training chat template.",
    )

    # OpenAI(-compatible) backend
    oai_p = sub.add_parser("openai", help="Use OpenAI or OpenAI-compatible endpoint (optionally with base_url)")
    oai_p.add_argument("--model", type=str, required=True, help="Model id, e.g. openai/gpt-4.1-mini or local name")
    oai_p.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    oai_p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override API base URL (useful for vLLM/OpenAI-compatible servers)",
    )
    oai_p.add_argument("--temperature", type=float, default=0.0)
    # Basic backend (direct provider/HF/vLLM)
    basic_p = sub.add_parser("basic", help="Generate responses via LiteLLM/HF/vLLM backends.")
    basic_p.add_argument("--model", required=True, help="Model identifier (e.g., openai/gpt-4.1-mini, hf:meta-llama/..., vllm:<path>).")
    basic_p.add_argument("--system", default=None, help="Optional system message for chat models.")
    basic_p.add_argument("--max-tokens", type=int, default=512)
    basic_p.add_argument("--temperature", type=float, default=0.0)
    basic_p.add_argument("--openai-api-base", default=None, help="Override OPENAI_API_BASE for LiteLLM provider.")
    basic_p.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for LiteLLM provider.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _read_prompts(args.prompts_file)

    # If provided, reproduce the train/eval split and exclude the training prompts
    if args.dataset_csv is not None:
        train_prompts, eval_prompts = _reproduce_train_eval_split(
            args.dataset_csv,
            prompt_column=args.dataset_prompt_column,
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
        )
        train_set = set(train_prompts)
        # Keep order from prompts file; drop anything in train_set
        prompts = [p for p in prompts if p not in train_set]
        # If we still have more than eval_size after filtering, truncate deterministically
        if args.eval_size is not None and len(prompts) > args.eval_size:
            prompts = prompts[: args.eval_size]

    # Optional limit override
    if args.limit is not None and len(prompts) > args.limit:
        prompts = prompts[: args.limit]

    if args.backend == "basic":
        _run_basic_generation(args, prompts)
        return

    if args.backend == "tinker":
        if "TINKER_API_KEY" not in os.environ:
            raise EnvironmentError("Please set TINKER_API_KEY for the Tinker backend.")
        # Resolve model_path from registry if adapter_name provided
        model_path = args.model_path
        resolved_adapter_name = args.adapter_name
        if not model_path and args.adapter_name and args.adapter_registry and args.adapter_registry.exists():
            try:
                with args.adapter_registry.open("r", encoding="utf-8") as fh:
                    reg = json.load(fh)
                entry = reg.get(args.adapter_name)
                registry_path = None
                registry_renderer = None
                if isinstance(entry, dict):
                    registry_path = entry.get("path")
                    registry_renderer = entry.get("renderer_name")
                    actual_name = entry.get("adapter_name")
                    if actual_name:
                        resolved_adapter_name = actual_name
                else:
                    registry_path = entry
                if registry_path:
                    model_path = registry_path
                    if args.renderer_name is None and registry_renderer:
                        args.renderer_name = registry_renderer
                    print(f"Resolved adapter '{args.adapter_name}' to model path from registry: {model_path}")
            except Exception:
                pass

        if not model_path and not args.adapter_name:
            raise ValueError("Provide either --model-path or --adapter-name for Tinker backend.")
        rows = _gen_with_tinker(
            adapter_name=resolved_adapter_name,
            model_path=(model_path or None),
            base_model=args.base_model,
            prompts=prompts,
            prompt_template=args.prompt_template,
            stop=(args.stop if args.stop else None),
            max_tokens=args.max_tokens,
            renderer_name=args.renderer_name,
            model_label=args.adapter_name,
        )
    elif args.backend == "openai":
        if "OPENAI_API_KEY" not in os.environ and not args.base_url:
            # Still allow if using a local OpenAI-compatible server that doesn't need a key
            pass
        rows = _gen_with_openai(
            model=args.model,
            prompts=prompts,
            prompt_template=args.prompt_template,
            system_prompt=args.system_prompt,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    _write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
