"""CPU-safe tests for the local-GPU backend's CUDA-gating contract.

The pure-CPU tests exercise the helper logic that decides bf16 / gradient
checkpointing / DataParallel guarding. The actual GPU training smoke test is gated
behind BOTH ``torch.cuda.is_available()`` AND ``DEMENTOR_RUN_GPU_TESTS=1`` so a plain
``pytest`` run (even on a CUDA box) never launches a GPU job by accident.
"""
import importlib.util
import os
import unittest


def _deps_available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft", "trl", "datasets"))


TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@unittest.skipUnless(_deps_available(), "local-training deps not installed")
class GatingHelpersTest(unittest.TestCase):
    def test_resolve_grad_accum_kwarg_env_default(self):
        from dementor.training.local_backend import _resolve_grad_accum

        self.assertEqual(_resolve_grad_accum(4), 4)  # explicit arg wins
        self.assertEqual(_resolve_grad_accum(None), 1)  # default
        prev = os.environ.get("DEMENTOR_GRAD_ACCUM")
        os.environ["DEMENTOR_GRAD_ACCUM"] = "8"
        try:
            self.assertEqual(_resolve_grad_accum(None), 8)  # env fallback
            self.assertEqual(_resolve_grad_accum(2), 2)  # explicit still wins over env
        finally:
            if prev is None:
                os.environ.pop("DEMENTOR_GRAD_ACCUM", None)
            else:
                os.environ["DEMENTOR_GRAD_ACCUM"] = prev

    def test_guard_no_dataparallel_is_noop_off_cuda_or_distributed(self):
        from dementor.training.local_backend import _guard_no_dataparallel

        # CPU path: always a no-op (returns None, never inspects CUDA).
        self.assertIsNone(_guard_no_dataparallel(False))
        # Under a distributed launch (WORLD_SIZE set) it must NOT raise even with use_cuda.
        prev = os.environ.get("WORLD_SIZE")
        os.environ["WORLD_SIZE"] = "4"
        try:
            self.assertIsNone(_guard_no_dataparallel(True))
        finally:
            if prev is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = prev

    def test_load_causal_lm_cpu_is_fp32(self):
        import torch

        from dementor.training.local_backend import _load_causal_lm

        try:
            model = _load_causal_lm(TINY_MODEL, use_cuda=False)
        except OSError as exc:  # offline
            self.skipTest(f"tiny model unavailable (offline?): {exc}")
        self.assertEqual(next(model.parameters()).dtype, torch.float32)


@unittest.skipUnless(
    _deps_available()
    and os.environ.get("DEMENTOR_RUN_GPU_TESTS") == "1"
    and importlib.util.find_spec("torch") is not None,
    "GPU smoke test (set DEMENTOR_RUN_GPU_TESTS=1 on a CUDA box to run)",
)
class GpuSmokeTest(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_sft_bf16_gradckpt_completion_only(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import pandas as pd

        from dementor.training import local_backend
        from dementor.training.tinker_backend import SFTDatasetConfig

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_csv = tmp / "train.csv"
            pd.DataFrame(
                {"prompt": ["What is 2+2?", "Capital of France?"], "model_response": ["4", "Paris"]}
            ).to_csv(train_csv, index=False)
            ds_cfg = SFTDatasetConfig(
                train_csv=train_csv, prompt_column="prompt", completion_column="model_response",
                train_size=2, eval_size=0, seed=0,
            )
            out_dir = tmp / "adapter"
            registry = tmp / "registry.json"
            outcome = local_backend.run_local_sft_job(
                dataset_config=ds_cfg, base_model=TINY_MODEL, batch_size=1, epochs=1,
                learning_rate=1e-3, prompt_template="{prompt}", completion_template=" {completion}",
                weights_name="gpu_sft", output_dir=out_dir, registry_path=registry,
                seed=0, lora_kwargs={"rank": 2}, device="cuda", max_length=64,
                gradient_accumulation_steps=2,
            )
            self.assertTrue((out_dir / "adapter_config.json").exists())
            self.assertEqual(outcome.sampler_path, str(out_dir))
            self.assertEqual(json.loads(registry.read_text())["gpu_sft"]["backend"], "local")


if __name__ == "__main__":
    unittest.main()
