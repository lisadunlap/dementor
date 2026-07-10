"""Single local (gemma-4) cell launcher for ``accelerate launch`` / ``torchrun`` runs.

Unlike the matrix-wide ``launch-sft`` / ``launch-dpo`` commands (which fan every
cell out over a thread pool and must NOT be wrapped in ``accelerate launch``),
this runs exactly ONE cell's LoRA SFT and/or DPO in the current process so the HF
Trainer can own DDP device placement and rank-aware saving. The rank-0 DPO
data-prep handshake helpers below coordinate the shared-filesystem preference
JSONL so only one process writes it.
"""
from __future__ import annotations

import time
from pathlib import Path

from dementor import config

from ._cells import Cell, dpo_data_path, sft_data_path
from ._constants import (
    DATASET_TEMPLATES,
    DPO_OUTPUT_DIR,
    MODEL_SLUG,
    SFT_OUTPUT_DIR,
)


def _dist_writer_rank() -> int:
    """Global process rank for the *shared-filesystem* single-writer guard.

    Prefer the GLOBAL ``RANK`` over ``LOCAL_RANK`` so exactly one process (global rank
    0) writes the shared preference JSONL. On multi-node every node has a local-rank-0,
    so keying the guard off ``LOCAL_RANK`` would let each node's local-rank-0 write the
    same shared file concurrently and corrupt it. Falls back to ``LOCAL_RANK`` only when
    ``RANK`` is unset, and to 0 when not launched distributed.
    """
    import os

    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _dpo_prep_token(seed: int, explicit: str | None = None) -> str:
    """Per-launch token stamped into the rank-0 DPO data-prep handshake flag.

    Must be IDENTICAL across every rank of one launch (so non-rank-0 waiters recognize
    *this* invocation's flag) yet DIFFER across launches (so a stale ``.prepared`` left
    in a non-empty ``preference_artifacts/`` by a prior run is rejected instead of waved
    through on mere existence). ``accelerate launch`` / ``torchrun`` hand every rank the
    same ``TORCHELASTIC_RUN_ID`` -- a fresh uuid4 per launch in the single-node default
    -- which is exactly such a value; an operator can force one via ``$DEMENTOR_PREP_TOKEN``
    (multi-node / pinned rdzv-id). The uuid4 fallback is reached only when NOT launched
    distributed (a lone process with no waiters), so cross-rank agreement is moot there.
    """
    import os
    import uuid

    if explicit:
        return explicit
    shared = os.environ.get("DEMENTOR_PREP_TOKEN") or os.environ.get("TORCHELASTIC_RUN_ID")
    return f"{shared}:{seed}" if shared else uuid.uuid4().hex


def _wait_for_prepared(flag: Path, token: str, *, timeout: float = 1800.0, poll: float = 2.0) -> None:
    """Block until ``flag`` exists AND its content equals ``token``, or raise on timeout.

    Gating on the flag's CONTENT (not mere existence) closes the TOCTOU race on a re-run
    into a non-empty dir: a stale ``.prepared`` from a prior launch carries a different
    token and is ignored until rank 0 republishes the flag for THIS launch (after it has
    atomically put train.jsonl in place).
    """
    start = time.time()
    while True:
        try:
            if flag.read_text() == token:
                return
        except OSError:  # not created yet / mid-write (FileNotFoundError is an OSError)
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out after {timeout:.0f}s waiting for {flag} (rank-0 DPO data prep)")
        time.sleep(poll)


def launch_local_cell(
    *,
    source: str,
    target: str,
    dataset: str,
    seed: int,
    phase: str = "all",
    per_device_batch_size: int | None = None,
    grad_accum: int | None = None,
    epochs: int | None = None,
    prep_token: str | None = None,
) -> dict:
    """Run ONE local (gemma-4) cell's LoRA SFT and/or DPO in the current process.

    This is the entrypoint to run under ``accelerate launch`` / ``torchrun`` for real
    multi-GPU (e.g. 4xH100) training: the HF Trainer inside ``run_local_sft_job`` /
    ``run_local_dpo_job`` owns DDP device placement and rank-aware saving. (The matrix
    ``launch-sft`` / ``launch-dpo`` commands iterate every cell with a thread pool and are
    NOT safe to wrap in ``accelerate launch`` — each rank would re-run the whole matrix.)

    Build the data first as plain processes (``build-sft-data`` / ``build-dpo-data``); the
    DPO preference JSONL is materialized once by rank 0 here, other ranks wait for it.
    ``phase`` is one of ``sft`` / ``dpo`` / ``all``. ``prep_token`` overrides the per-launch
    DPO data-prep handshake token (else derived from ``$DEMENTOR_PREP_TOKEN`` /
    ``$TORCHELASTIC_RUN_ID``); pass a shared value to force multi-node agreement.
    """
    from dementor.training.local_backend import (
        LocalDPOParams,
        run_local_dpo_job,
        run_local_sft_job,
    )
    from dementor.training.tinker_backend import SFTDatasetConfig

    if config.backend_for(source) != "local":
        raise ValueError(
            f"{source} is not a local-backend model. Use launch-sft/launch-dpo for Tinker models."
        )

    cell = Cell(source=source, target=target, dataset=dataset, seed=seed)
    prompt_template, completion_template = DATASET_TEMPLATES[cell.dataset]
    name = f"{MODEL_SLUG[cell.source]}_as_{MODEL_SLUG[cell.target]}_seed{cell.seed}"
    sft_out_dir = SFT_OUTPUT_DIR / cell.dataset / name
    dpo_out_dir = DPO_OUTPUT_DIR / cell.dataset / name
    sft_hp = config.sft()
    dpo_hp = config.dpo()
    summary: dict = {"cell": cell.slug, "source": source, "target": target, "phase": phase}

    if phase in ("sft", "all"):
        sft_csv = sft_data_path(cell)
        if not sft_csv.exists():
            raise FileNotFoundError(f"SFT data missing: {sft_csv} (run `dementor-matrix build-sft-data`).")
        ds_cfg = SFTDatasetConfig(
            train_csv=sft_csv, eval_csv=None, prompt_column="prompt",
            completion_column="model_response", train_size=sft_hp.get("train_size", 500),
            eval_size=0, seed=cell.seed,
        )
        run_local_sft_job(
            dataset_config=ds_cfg, base_model=cell.source,
            batch_size=per_device_batch_size or sft_hp["batch_size"],
            epochs=epochs or sft_hp["epochs"], learning_rate=sft_hp["learning_rate"],
            prompt_template=prompt_template, completion_template=completion_template,
            weights_name=f"sft_{cell.slug}", output_dir=sft_out_dir,
            registry_path=config.registry_path(), seed=cell.seed,
            lora_kwargs=config.lora(), gradient_accumulation_steps=grad_accum,
        )
        summary["sft_output_dir"] = str(sft_out_dir)

    if phase in ("dpo", "all"):
        from dementor.training.dpo import (
            PreferenceDatasetConfig,
            prepare_preference_examples,
            write_preference_artifacts,
        )

        pref_csv = dpo_data_path(cell)
        if not pref_csv.exists():
            raise FileNotFoundError(f"DPO data missing: {pref_csv} (run `dementor-matrix build-dpo-data`).")
        artifacts_dir = dpo_out_dir / "preference_artifacts"
        train_jsonl = artifacts_dir / "train.jsonl"
        eval_jsonl = artifacts_dir / "eval.jsonl"
        ready_flag = artifacts_dir / ".prepared"
        token = _dpo_prep_token(cell.seed, prep_token)
        # Materialize the preference JSONL exactly once (global rank 0); concurrent writers
        # would corrupt it and the torch process group isn't up yet, so coordinate via a flag
        # file. Rank 0 stages the artifacts then os.replace()s train.jsonl into place (atomic)
        # and publishes a flag stamped with THIS launch's token; waiters block on the token --
        # not mere existence -- so a re-run can't read a stale/partial JSONL from a prior run.
        if _dist_writer_rank() == 0:
            import os
            import shutil

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ready_flag.unlink(missing_ok=True)  # invalidate any prior-launch flag up front
            pref_cfg = PreferenceDatasetConfig(
                dataset_csv=pref_csv, prompt_column="prompt",
                chosen_column="chosen_response", rejected_column="rejected_response",
                train_size=dpo_hp.get("train_size", 500), eval_size=0, seed=cell.seed,
            )
            train_ex, eval_ex = prepare_preference_examples(pref_cfg)
            staging = artifacts_dir / ".staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            arts = write_preference_artifacts(
                output_dir=staging, train_examples=train_ex, eval_examples=eval_ex,
                metadata=None, stub=None,
            )
            # Atomically publish each artifact (same FS -> os.replace is atomic) so no reader
            # ever sees a half-written train.jsonl, even off the flag path.
            for src in (arts.train_jsonl, arts.eval_jsonl, arts.train_csv, arts.eval_csv):
                if src is not None and Path(src).exists():
                    os.replace(src, artifacts_dir / Path(src).name)
            shutil.rmtree(staging, ignore_errors=True)
            # Publish the handshake flag LAST, atomically (temp + os.replace): its presence
            # AND matching token guarantee the atomically-replaced train.jsonl is complete.
            tmp_flag = artifacts_dir / ".prepared.tmp"
            tmp_flag.write_text(token)
            os.replace(tmp_flag, ready_flag)
        else:
            _wait_for_prepared(ready_flag, token)
        eval_arg = eval_jsonl if (eval_jsonl.exists() and eval_jsonl.stat().st_size > 0) else None
        # Single-GPU local DPO materializes fp32 logits over the FULL vocab; large-vocab
        # towers (gemma ~256K, plus gemma's final_logit_softcapping) OOM one 80 GB card at
        # the config batch(16)/length(4096) -- those config values target Tinker's SHARDED
        # backend, not a single card. Mirror _build_dpo_cfg's proven local length caps and
        # additionally shrink the per-device batch: the concatenated chosen+rejected forward
        # is the memory bottleneck (~batch*2 * seq * vocab * 4 bytes). Preserve the effective
        # batch via gradient accumulation. An explicit --per-device-batch-size / --grad-accum
        # still wins, so the accelerate multi-GPU path is unaffected.
        local_dpo_max_length = min(dpo_hp["max_length"], 1536)
        if "31b" in cell.source.lower():  # 62 GB text tower leaves almost no headroom
            local_dpo_max_length = min(local_dpo_max_length, 1024)
        local_dpo_batch = per_device_batch_size or min(dpo_hp["batch_size"], 2)
        local_dpo_accum = grad_accum
        if (local_dpo_accum is None and not per_device_batch_size
                and dpo_hp["batch_size"] > local_dpo_batch):
            local_dpo_accum = max(1, dpo_hp["batch_size"] // local_dpo_batch)
        params = LocalDPOParams(
            model_name=cell.source, load_checkpoint_path=str(sft_out_dir),
            weights_name=f"dpo_{cell.slug}", registry_path=config.registry_path(),
            learning_rate=dpo_hp["learning_rate"], dpo_beta=dpo_hp["dpo_beta"],
            num_epochs=dpo_hp["num_epochs"],
            batch_size=local_dpo_batch,
            max_length=local_dpo_max_length, lora_rank=dpo_hp["lora_rank"],
            seed=cell.seed, gradient_accumulation_steps=local_dpo_accum,
        )
        run_local_dpo_job(train_jsonl=train_jsonl, eval_jsonl=eval_arg, params=params, output_dir=dpo_out_dir)
        summary["dpo_output_dir"] = str(dpo_out_dir)

    return summary
