from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.behavioral_inertia_metrics import (
    anisotropy,
    bootstrap_behavioral_metrics,
    compute_behavioral_metrics,
    movement_by_axis,
    source_persistence,
)
from scripts.analysis.latent_behavior_axes import (
    build_descriptor_matrix,
    factorize_fixed_basis,
    fit_behavioral_axis_basis,
    load_basis,
    save_basis,
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
        per_axis, summary = compute_behavioral_metrics(latent, seed=1, min_axis_separation=0.2)
        self.assertEqual(len(per_axis), 2)
        self.assertEqual(int(per_axis["active_axis"].sum()), 2)
        self.assertIn("source_persistence", summary)
        self.assertIn("target_assimilation", summary)
        self.assertTrue(summary["separable"])

    def test_active_axis_filtering(self) -> None:
        latent = pd.DataFrame(
            [
                {"row_id": 0, "condition": "source", "pc1": 0.0, "pc2": 0.0},
                {"row_id": 1, "condition": "source", "pc1": 0.0, "pc2": 0.0},
                {"row_id": 0, "condition": "disguised", "pc1": 0.5, "pc2": 10.0},
                {"row_id": 1, "condition": "disguised", "pc1": 0.5, "pc2": 10.0},
                {"row_id": 0, "condition": "target", "pc1": 1.0, "pc2": 0.01},
                {"row_id": 1, "condition": "target", "pc1": 1.0, "pc2": 0.01},
            ]
        )
        per_axis, summary = compute_behavioral_metrics(latent, min_axis_separation=0.1)
        self.assertEqual(per_axis["active_axis"].tolist(), [True, False])
        self.assertAlmostEqual(summary["source_persistence"], 0.5)

    def test_no_active_axes_disables_probe_gate(self) -> None:
        latent = pd.DataFrame(
            [
                {"row_id": 0, "condition": "source", "pc1": 0.0, "pc2": 0.0},
                {"row_id": 1, "condition": "source", "pc1": 0.0, "pc2": 0.0},
                {"row_id": 0, "condition": "disguised", "pc1": 10.0, "pc2": 10.0},
                {"row_id": 1, "condition": "disguised", "pc1": 10.0, "pc2": 10.0},
                {"row_id": 0, "condition": "target", "pc1": 0.01, "pc2": 0.01},
                {"row_id": 1, "condition": "target", "pc1": 0.01, "pc2": 0.01},
            ]
        )
        per_axis, summary = compute_behavioral_metrics(latent, min_axis_separation=1.0)
        self.assertEqual(per_axis["active_axis"].tolist(), [False, False])
        self.assertEqual(summary["n_active_axes"], 0)
        self.assertFalse(summary["separable"])
        self.assertTrue(np.isnan(summary["probe_cv_accuracy"]))

    def test_bootstrap_summary(self) -> None:
        latent = pd.DataFrame(
            [
                {"row_id": 0, "condition": "source", "pc1": 0.0},
                {"row_id": 1, "condition": "source", "pc1": 0.0},
                {"row_id": 2, "condition": "source", "pc1": 0.1},
                {"row_id": 3, "condition": "source", "pc1": 0.1},
                {"row_id": 0, "condition": "disguised", "pc1": 0.4},
                {"row_id": 1, "condition": "disguised", "pc1": 0.5},
                {"row_id": 2, "condition": "disguised", "pc1": 0.5},
                {"row_id": 3, "condition": "disguised", "pc1": 0.6},
                {"row_id": 0, "condition": "target", "pc1": 1.0},
                {"row_id": 1, "condition": "target", "pc1": 1.0},
                {"row_id": 2, "condition": "target", "pc1": 1.1},
                {"row_id": 3, "condition": "target", "pc1": 1.1},
            ]
        )
        boot_df, summary = bootstrap_behavioral_metrics(
            latent,
            active_axes=["pc1"],
            samples=5,
            seed=1,
        )
        self.assertEqual(len(boot_df), 5)
        self.assertIn("source_persistence_ci_low", summary)


class BehavioralAxisBasisTests(unittest.TestCase):
    def test_fixed_basis_excludes_disguised_rows(self) -> None:
        matrices_a = {
            "source": np.array([[0.0, 0.0], [0.1, 0.0]]),
            "target": np.array([[1.0, 1.0], [1.1, 1.0]]),
            "disguised": np.array([[0.5, 0.5], [0.6, 0.5]]),
        }
        matrices_b = {
            **matrices_a,
            "disguised": np.array([[100.0, -100.0], [200.0, -200.0]]),
        }
        basis_a = fit_behavioral_axis_basis(
            matrices_a,
            ["a", "b"],
            descriptor_mode="style_only",
            encoder_model="test-encoder",
            feature_set="style_scalars",
            k=2,
        )
        basis_b = fit_behavioral_axis_basis(
            matrices_b,
            ["a", "b"],
            descriptor_mode="style_only",
            encoder_model="test-encoder",
            feature_set="style_scalars",
            k=2,
        )
        np.testing.assert_allclose(np.abs(basis_a.components), np.abs(basis_b.components))

    def test_basis_roundtrip(self) -> None:
        matrices = {
            "source": np.array([[0.0, 0.0], [0.1, 0.0]]),
            "target": np.array([[1.0, 1.0], [1.1, 1.0]]),
            "disguised": np.array([[0.5, 0.5], [0.6, 0.5]]),
        }
        basis = fit_behavioral_axis_basis(
            matrices,
            ["a", "b"],
            descriptor_mode="style_only",
            encoder_model="test-encoder",
            feature_set="style_scalars",
            k=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pkl"
            save_basis(basis, path)
            loaded = load_basis(path)
            np.testing.assert_allclose(basis.components, loaded.components)

    def test_feature_set_ablation_shapes(self) -> None:
        texts = {
            "source": ["Brief answer.", "Another short answer."],
            "disguised": ["A structured answer with bullets:\n- one", "Another structured answer."],
            "target": ["A detailed, structured response.", "A second detailed response."],
        }
        scalar_mats, scalar_names = build_descriptor_matrix(
            texts,
            descriptor_mode="style_only",
            feature_set="style_scalars",
        )
        binary_mats, binary_names = build_descriptor_matrix(
            texts,
            descriptor_mode="style_only",
            feature_set="style_binaries",
        )
        self.assertNotEqual(len(scalar_names), len(binary_names))
        self.assertEqual(scalar_mats["source"].shape[1], len(scalar_names))
        self.assertEqual(binary_mats["source"].shape[1], len(binary_names))


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
                bootstrap_samples=5,
            )
            out = Path(tmp)
            self.assertTrue((out / "latent_scores.csv").exists())
            self.assertTrue((out / "axis_loadings.csv").exists())
            self.assertTrue((out / "per_axis_movement.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "bootstrap_summary.csv").exists())
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
