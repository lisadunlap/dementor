"""CPU smoke test for the local-GPU training backend (HF + PEFT + TRL).

Trains a tiny LoRA on a tiny random model to prove ``run_local_sft_job`` writes a
PEFT adapter and records it in the shared registry with ``backend: "local"``.
Skips cleanly if the training deps or the tiny model are unavailable (offline).
"""
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _deps_available() -> bool:
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft", "trl", "datasets"))


@unittest.skipUnless(_deps_available(), "local-training deps (peft/trl/datasets) not installed")
class LocalBackendSFTTest(unittest.TestCase):
    def test_sft_writes_adapter_and_registry(self):
        from dementor.training import local_backend
        from dementor.training.tinker_backend import SFTDatasetConfig

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_csv = tmp / "train.csv"
            pd.DataFrame(
                {"prompt": ["What is 2+2?", "Capital of France?", "Say hi."],
                 "model_response": ["4", "Paris", "Hi!"]}
            ).to_csv(train_csv, index=False)

            ds_cfg = SFTDatasetConfig(
                train_csv=train_csv, prompt_column="prompt", completion_column="model_response",
                train_size=3, eval_size=0, seed=42,
            )
            out_dir = tmp / "adapter"
            registry = tmp / "registry.json"
            try:
                outcome = local_backend.run_local_sft_job(
                    dataset_config=ds_cfg, base_model=TINY_MODEL, batch_size=2, epochs=1,
                    learning_rate=1e-3, prompt_template="{prompt}", completion_template=" {completion}",
                    weights_name="test_local_sft", output_dir=out_dir, registry_path=registry,
                    seed=42, lora_kwargs={"rank": 2}, device="cpu", max_length=64,
                )
            except OSError as exc:  # model download failed (offline)
                self.skipTest(f"tiny model unavailable (offline?): {exc}")

            self.assertTrue((out_dir / "adapter_config.json").exists())
            self.assertTrue(
                (out_dir / "adapter_model.safetensors").exists() or (out_dir / "adapter_model.bin").exists()
            )
            self.assertEqual(outcome.sampler_path, str(out_dir))
            reg = json.loads(registry.read_text())
            self.assertIn("test_local_sft", reg)
            self.assertEqual(reg["test_local_sft"]["backend"], "local")
            self.assertEqual(reg["test_local_sft"]["path"], str(out_dir))

    def test_dpo_writes_adapter_and_registry(self):
        from dementor.training import local_backend
        from dementor.training.local_backend import LocalDPOParams

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pref = tmp / "pref.jsonl"
            with pref.open("w") as fh:
                for _ in range(4):
                    fh.write(json.dumps({"prompt": "Hi", "chosen": "Hello there!", "rejected": "no"}) + "\n")
            out_dir = tmp / "dpo_adapter"
            registry = tmp / "registry.json"
            params = LocalDPOParams(
                model_name=TINY_MODEL, load_checkpoint_path=None, weights_name="test_local_dpo",
                registry_path=registry, num_epochs=1, batch_size=2, lora_rank=2, max_length=64, device="cpu",
            )
            try:  # eval_jsonl=None is exactly what the (fixed) pipeline passes for eval_size=0
                adapter_dir = local_backend.run_local_dpo_job(
                    train_jsonl=pref, eval_jsonl=None, params=params, output_dir=out_dir
                )
            except OSError as exc:
                self.skipTest(f"tiny model unavailable (offline?): {exc}")

            self.assertEqual(Path(adapter_dir), out_dir)  # out_dir honored (not derived from checkpoint)
            self.assertTrue((out_dir / "adapter_config.json").exists())
            reg = json.loads(registry.read_text())
            self.assertEqual(reg["test_local_dpo"]["backend"], "local")

    def _train_tiny_sft_adapter(self, tmp: Path) -> Path:
        """Train a tiny SFT LoRA adapter on the CPU tiny model; return its dir.

        Produces the ``load_checkpoint_path`` that the DPO merge-and-unload reference path
        consumes. Lets ``OSError`` (offline tiny-model download) propagate so the caller can
        skip exactly like the other tests.
        """
        from dementor.training import local_backend
        from dementor.training.tinker_backend import SFTDatasetConfig

        train_csv = tmp / "sft_train.csv"
        pd.DataFrame(
            {"prompt": ["What is 2+2?", "Capital of France?", "Say hi."],
             "model_response": ["4", "Paris", "Hi!"]}
        ).to_csv(train_csv, index=False)
        ds_cfg = SFTDatasetConfig(
            train_csv=train_csv, prompt_column="prompt", completion_column="model_response",
            train_size=3, eval_size=0, seed=42,
        )
        sft_dir = tmp / "sft_adapter"
        local_backend.run_local_sft_job(
            dataset_config=ds_cfg, base_model=TINY_MODEL, batch_size=2, epochs=1,
            learning_rate=1e-3, prompt_template="{prompt}", completion_template=" {completion}",
            weights_name="test_sft_parent", output_dir=sft_dir,
            registry_path=tmp / "sft_registry.json", seed=42, lora_kwargs={"rank": 2},
            device="cpu", max_length=64,
        )
        return sft_dir

    def test_dpo_from_sft_checkpoint_merges_and_writes_adapter(self):
        """DPO reference path: merge an SFT adapter into the base, then train a fresh DPO LoRA.

        Exercises ``run_local_dpo_job(load_checkpoint_path=<sft adapter>)`` -- the branch that
        does ``PeftModel.from_pretrained(...).merge_and_unload()`` and trains with
        ``ref_model=None`` -- on CPU with the tiny model, and checks it emits a valid DPO
        adapter dir plus a registry entry whose ``sft_parent`` points at the SFT adapter.
        """
        from dementor.training import local_backend
        from dementor.training.local_backend import LocalDPOParams

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pref = tmp / "pref.jsonl"
            with pref.open("w") as fh:
                for _ in range(4):
                    fh.write(json.dumps({"prompt": "Hi", "chosen": "Hello there!", "rejected": "no"}) + "\n")
            out_dir = tmp / "dpo_adapter"
            registry = tmp / "registry.json"
            try:  # mirror the other tests: skip cleanly when the tiny model is unavailable
                sft_dir = self._train_tiny_sft_adapter(tmp)
                params = LocalDPOParams(
                    model_name=TINY_MODEL, load_checkpoint_path=str(sft_dir),
                    weights_name="test_dpo_from_sft", registry_path=registry,
                    num_epochs=1, batch_size=2, lora_rank=2, max_length=64, device="cpu",
                )
                adapter_dir = local_backend.run_local_dpo_job(
                    train_jsonl=pref, eval_jsonl=None, params=params, output_dir=out_dir
                )
            except OSError as exc:
                self.skipTest(f"tiny model unavailable (offline?): {exc}")

            # The SFT adapter had to be a loadable PEFT dir (the merge source).
            self.assertTrue((sft_dir / "adapter_config.json").exists())
            self.assertEqual(Path(adapter_dir), out_dir)
            self.assertTrue((out_dir / "adapter_config.json").exists())
            self.assertTrue(
                (out_dir / "adapter_model.safetensors").exists() or (out_dir / "adapter_model.bin").exists()
            )
            reg = json.loads(registry.read_text())
            self.assertEqual(reg["test_dpo_from_sft"]["backend"], "local")
            self.assertEqual(reg["test_dpo_from_sft"]["sft_parent"], str(sft_dir))

    def test_export_local_adapter_writes_provenance(self):
        from dementor.training import local_backend

        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "adapter_config.json").write_text("{}")
            res = local_backend.export_local_adapter(
                adapter_dir=src, base_model="google/gemma-4-E4B-it", output_dir=src
            )
            self.assertEqual(res["status"], "exported")
            self.assertTrue((src / "dementor_local_export.json").exists())


if __name__ == "__main__":
    unittest.main()
