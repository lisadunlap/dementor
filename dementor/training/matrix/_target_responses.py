"""Phase A.2: generate one canonical target-model response per training prompt.

These responses become the SFT completions (and the DPO chosen/rejected sides).
Local-backend models run through ``dementor.training.local_backend``; Tinker
models sample through the Tinker service (imported lazily so all-local runs never
require the optional ``tinker`` package).
"""
from __future__ import annotations

import time

import pandas as pd

from dementor import config

from ._cells import baseline_path
from ._constants import CHAT_TEMPLATE_KWARGS, TRAIN_DATASETS
from ._models import clean_response, renderer_for


def generate_target_responses(
    *,
    models: list[str],
    datasets: list[str],
    dry_run: bool = False,
    parallel: int = 1,
) -> int:
    """Generate one canonical response per (model, train_prompt) for SFT training data.

    Skips combos that already have a cache file with the right number of rows.
    Returns total number of new generations.

    parallel: number of in-flight Tinker sample() requests per model. 1 = sequential.
    """
    from dotenv import load_dotenv

    load_dotenv()
    # `tinker` is imported lazily inside the Tinker branch below so an all-local
    # data-generation run does not require the optional `tinker` package.
    service = None  # created lazily on first Tinker model (an all-local run needs no Tinker)

    total_generated = 0
    for model in models:
        renderer = renderer_for(model)  # guarded helper (cookbook with dict fallback)
        sampling = None
        tok = None
        for dataset in datasets:
            train_csv = TRAIN_DATASETS[dataset]
            prompts_df = pd.read_csv(train_csv)
            n_expected = len(prompts_df)
            out_path = baseline_path(model, dataset)

            if out_path.exists():
                existing = pd.read_csv(out_path)
                if len(existing) == n_expected:
                    print(f"  [skip] {model} on {dataset}: {n_expected} responses already cached")
                    continue

            if dry_run:
                print(f"  [dry-run] would generate {n_expected} responses: {model} on {dataset}")
                continue

            print(f"  [gen] {model} on {dataset}: {n_expected} prompts (renderer={renderer})")
            if config.backend_for(model) == "local":
                from dementor.training.local_backend import generate_local_responses
                prompts_list = prompts_df["prompt"].tolist()
                t0 = time.time()
                replies = generate_local_responses(
                    model=model, prompts=prompts_list,
                    chat_template_kwargs=CHAT_TEMPLATE_KWARGS.get(model, {}),
                    max_new_tokens=512, temperature=0.0,
                )
                rows = [{"prompt": p, "model_response": clean_response(model, r), "model": model}
                        for p, r in zip(prompts_list, replies)]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                total_generated += len(rows)
                print(f"  [wrote] {out_path} ({len(rows)} rows, {time.time() - t0:.0f}s, local)")
                continue
            # Tinker path — import lazily so all-local runs never need the `tinker` package.
            import tinker
            from tinker import types
            if sampling is None:
                if service is None:
                    service = tinker.ServiceClient()
                sampling = service.create_sampling_client(base_model=model)
                if hasattr(sampling, "get_tokenizer"):
                    tok = sampling.get_tokenizer()
                else:
                    train_client = service.create_lora_training_client(base_model=model)
                    tok = train_client.get_tokenizer()

            params = types.SamplingParams(max_tokens=512, temperature=0.0, stop=None)
            chat_kwargs = CHAT_TEMPLATE_KWARGS.get(model, {})
            prompts_list = prompts_df["prompt"].tolist()
            rows: list[dict] = []
            t0 = time.time()

            def _submit(prompt: str):
                messages = [{"role": "user", "content": prompt}]
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
                encoded = tok.encode(rendered, add_special_tokens=True)
                mi = types.ModelInput.from_ints(tokens=encoded)
                return sampling.sample(prompt=mi, sampling_params=params, num_samples=1)

            def _result_with_retry(prompt: str, fut, max_retries: int = 4):
                """Collect future; on transient API errors, resubmit + retry."""
                current = fut
                last_err = None
                for attempt in range(max_retries):
                    try:
                        return current.result()
                    except Exception as e:
                        last_err = e
                        print(
                            f"    [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)[:100]}",
                            flush=True,
                        )
                        current = _submit(prompt)
                raise last_err

            # Chunked pipelining: submit `parallel` futures, then drain. Order preserved.
            for chunk_start in range(0, len(prompts_list), parallel):
                chunk = prompts_list[chunk_start : chunk_start + parallel]
                futures = [(p, _submit(p)) for p in chunk]
                for p, f in futures:
                    reply_raw = tok.decode(_result_with_retry(p, f).sequences[0].tokens)
                    reply = clean_response(model, reply_raw)
                    rows.append({"prompt": p, "model_response": reply, "model": model})
                completed = chunk_start + len(chunk)
                # Report progress at ~every 25 prompts
                if completed // 25 != (completed - len(chunk)) // 25 or completed == n_expected:
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (n_expected - completed) / max(rate, 1e-6)
                    print(f"    {completed}/{n_expected} ({rate:.2f} gens/s, ETA {eta:.0f}s)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            total_generated += len(rows)
            print(f"  [wrote] {out_path} ({len(rows)} rows, {time.time() - t0:.0f}s)")

    return total_generated
