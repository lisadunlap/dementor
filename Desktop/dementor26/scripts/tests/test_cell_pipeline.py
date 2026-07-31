"""Integration smoke test for the cell pipeline's non-Tinker path.

Exercises run_cell_pipeline.assemble_and_run end-to-end on fabricated generation
CSVs (no Tinker / no network): staging on shared endpoints -> manifest ->
supervised cell -> ladder, and the A3 trust/saturation columns. Skips if
sentence-transformers (the full feature set's encoder) is unavailable.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.run_cell_pipeline import assemble_and_run


class CellPipelineAssembleTests(unittest.TestCase):
    def test_assemble_and_run_produces_cell_summary(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self.skipTest("sentence-transformers not installed; full feature set unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "cell"
            gen = cell / "gen"
            gen.mkdir(parents=True)
            prompts = [f"Question {i}: explain concept number {i} in detail." for i in range(16)]

            def write(name: str, render) -> None:
                pd.DataFrame(
                    {"prompt": prompts, "model_response": [render(i) for i in range(len(prompts))]}
                ).to_csv(gen / name, index=False)

            # Source = plain prose; target = markdown/bulleted; rungs interpolate toward target.
            write("source_seed1.csv", lambda i: f"The answer to concept {i} is straightforward and plainly stated.")
            write("source_seed2.csv", lambda i: f"The answer to concept {i} is straightforward, stated plainly.")
            write("target_seed1.csv", lambda i: f"## Concept {i}\n- point A\n- point B\n\n**Summary:** done.")
            write("target_seed2.csv", lambda i: f"## Concept {i}\n- point one\n- point two\n\n**Summary:** complete.")
            write("rung_sft.csv", lambda i: f"## Concept {i}\n- point A\nand a plainly stated note about concept {i}.")
            write("rung_dpo.csv", lambda i: f"## Concept {i}\n- point A\n- point B\n\n**Summary:** fully done.")

            args = types.SimpleNamespace(
                dataset="gsm8k", source="llama-3.1-8b", target="gpt-oss-20b",
                bootstrap=20, adapter_seeds=1, calibration_judge=None, calibration_n=0,
            )
            summary = assemble_and_run(args, cell, gen)

            self.assertTrue((cell / "cell_summary.csv").exists())
            self.assertIn("baseline_persistence", summary)
            cs = pd.read_csv(cell / "cell_summary.csv")
            self.assertEqual(set(cs["method"]), {"sft", "dpo"})
            for column in ("persistence", "movement_raw", "over_assimilation", "trustworthy", "anchored"):
                self.assertIn(column, cs.columns)


if __name__ == "__main__":
    unittest.main()
