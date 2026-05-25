from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.behavioral_inertia_metrics import (
    anisotropy,
    compute_behavioral_metrics,
    movement_by_axis,
    source_persistence,
)
from scripts.analysis.run_behavioral_inertia import run_behavioral_inertia
from scripts.analysis.activation_bridge import _parse_layers
from scripts.analysis.activation_steering import parse_strengths, resolve_layer_index


class BehavioralInertiaMetricTests(unittest.TestCase):
    def test_movement_and_persistence(self) -> None:
        source = np.array([[0.0, 0.0], [0.0, 0.0]])
        target = np.array([[1.0, 2.0], [1.0, 2.0]])
        disguised = np.array([[0.5, 3.0], [0.5, 3.0]])
        movement = movement_by_axis(source, disguised, target)
        np.testing.assert_allclose(movement, np.array([0.5, 1.5]))
        self.assertAlmostEqual(source_persistence(movement), 0.25)
        self.assertGreater(anisotropy(movement), 0.0)

    def test_compute_metrics_from_latent_scores(self) -> None:
        latent = pd.DataFrame(
            [
                {"condition": "source", "pc1": 0.0, "pc2": 0.0},
                {"condition": "source", "pc1": 0.1, "pc2": 0.0},
                {"condition": "disguised", "pc1": 0.5, "pc2": 0.2},
                {"condition": "disguised", "pc1": 0.6, "pc2": 0.3},
                {"condition": "target", "pc1": 1.0, "pc2": 1.0},
                {"condition": "target", "pc1": 1.1, "pc2": 1.0},
            ]
        )
        per_axis, summary = compute_behavioral_metrics(latent, seed=1)
        self.assertEqual(len(per_axis), 2)
        self.assertIn("source_persistence", summary)
        self.assertIn("target_assimilation", summary)


class BehavioralInertiaCliTests(unittest.TestCase):
    def test_run_behavioral_inertia_fixture(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_behavioral_inertia(
                comparison_csv=str(fixture),
                output_dir=tmp,
                source_model="source-test",
                target_model="target-test",
                method="contrastive",
                dataset="fixture",
                k=3,
            )
            out = Path(tmp)
            self.assertTrue((out / "latent_scores.csv").exists())
            self.assertTrue((out / "axis_loadings.csv").exists())
            self.assertTrue((out / "per_axis_movement.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertEqual(summary["dataset"], "fixture")
            self.assertEqual(summary["method"], "contrastive")


class ActivationBridgeUnitTests(unittest.TestCase):
    def test_parse_layers_defaults_and_negative_indices(self) -> None:
        self.assertEqual(_parse_layers("0,2,-1", 5), [0, 2, 4])
        defaults = _parse_layers(None, 9)
        self.assertEqual(defaults[0], 0)
        self.assertEqual(defaults[-1], 8)
        self.assertEqual(_parse_layers("all", 3), [0, 1, 2])


class ActivationSteeringUnitTests(unittest.TestCase):
    def test_parse_strengths(self) -> None:
        self.assertEqual(parse_strengths("0, 0.5,2"), [0.0, 0.5, 2.0])
        with self.assertRaises(ValueError):
            parse_strengths("")

    def test_resolve_layer_index(self) -> None:
        self.assertEqual(resolve_layer_index(-1, 12), 11)
        self.assertEqual(resolve_layer_index(3, 12), 3)
        with self.assertRaises(ValueError):
            resolve_layer_index(12, 12)


if __name__ == "__main__":
    unittest.main()
