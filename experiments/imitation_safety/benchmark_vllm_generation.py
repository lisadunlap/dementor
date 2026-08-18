#!/usr/bin/env python3
"""Benchmark vLLM+LoRA against a completed HF erosion-generation checkpoint.

The reference CSV supplies both prompts and the exact greedy outputs produced by
the current Transformers evaluator.  Loading time and generation time are
reported separately, and a JSON artifact records exact text agreement.  This is
an adoption gate, not part of the paper's result matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--actual-csv", type=Path,
                        help="Optionally save prompts and generated responses for a paired gate")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--attention-backend", choices=("FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"))
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--spec-model")
    parser.add_argument("--spec-method")
    parser.add_argument("--spec-tokens", type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from dementor import config
    from experiments.imitation_safety.erosion_common import _render

    with args.reference_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:args.limit]
    prompts = [row["prompt"] for row in rows]
    expected = [row["model_response"] for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    chat_kwargs = dict(config.model(args.base_model).get("chat_template_kwargs", {}))
    rendered = [_render(tokenizer, args.base_model, prompt, chat_kwargs) for prompt in prompts]

    started = time.perf_counter()
    engine_kwargs = {}
    if args.attention_backend:
        engine_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.spec_model:
        engine_kwargs.update(
            spec_model=args.spec_model,
            spec_method=args.spec_method or "eagle3",
            spec_tokens=args.spec_tokens or 3,
        )
    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        **engine_kwargs,
    )
    loaded = time.perf_counter()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42)
    request = LoRARequest("composed_sft_dpo", 1, str(args.adapter.resolve()))
    outputs = llm.generate(rendered, sampling, lora_request=request, use_tqdm=False)
    finished = time.perf_counter()

    def clean_response(raw: str) -> str:
        # Keep this benchmark environment independent of the training extras. This is the same
        # generic scaffold cleanup as matrix.clean_response. It deliberately does not implement
        # gpt-oss's special final-channel selection because local campaign sources do not use it.
        return re.sub(r"<\|(?:start|end|return|channel|message|constrain|eot_id|eom_id|"
                      r"im_start|im_end|endoftext)\|>", "", raw).strip()

    actual = [clean_response(item.outputs[0].text) for item in outputs]
    exact = [a == b for a, b in zip(actual, expected)]
    generated_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    result = {
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
        "n": len(rows),
        "max_tokens": args.max_tokens,
        "attention_backend": args.attention_backend or "auto",
        "enforce_eager": args.enforce_eager,
        "spec_model": args.spec_model,
        "spec_method": (args.spec_method or "eagle3") if args.spec_model else None,
        "spec_tokens": (args.spec_tokens or 3) if args.spec_model else None,
        "load_seconds": loaded - started,
        "generation_seconds": finished - loaded,
        "prompts_per_second": len(rows) / (finished - loaded),
        "generated_tokens": generated_tokens,
        "output_tokens_per_second": generated_tokens / (finished - loaded),
        "exact_text_matches": sum(exact),
        "exact_text_match_rate": sum(exact) / len(exact) if exact else None,
        "first_mismatches": [
            {"index": i, "expected": expected[i], "actual": actual[i]}
            for i, matches in enumerate(exact) if not matches
        ][:5],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.actual_csv:
        args.actual_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.actual_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("prompt", "model_response"))
            writer.writeheader()
            writer.writerows(
                {"prompt": prompt, "model_response": response}
                for prompt, response in zip(prompts, actual)
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
