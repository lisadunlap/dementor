from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.common import normalize_comparison_df
from scripts.analysis.behavioral_inertia_metrics import (
    anisotropy,
    bootstrap_behavioral_metrics,
    compute_behavioral_metrics,
    movement_by_axis,
    projection_movement,
    projection_movement_per_row,
    projection_persistence,
    source_persistence,
    weighted_persistence,
)
from scripts.analysis.latent_behavior_axes import (
    big5_dimension_scores_df,
    build_descriptor_matrix,
    factorize_fixed_basis,
    fit_behavioral_axis_basis,
    load_basis,
    run_latent_analysis,
    save_basis,
    summarize_big5_dimension_movement,
)
from scripts.analysis.run_behavioral_inertia import run_behavioral_inertia
from scripts.analysis.behavioral_cell_evaluator import (
    BehavioralCellSpec,
    IdentityControlSpec,
    MethodSpec,
    SelfBaselineSpec,
    run_behavioral_cell,
)
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
        self.assertAlmostEqual(weighted_persistence(movement, np.array([3.0, 1.0])), 0.375)
        self.assertGreater(anisotropy(movement), 0.0)
        # Projection: <dis-src, tgt-src>/<tgt-src,tgt-src> = (0.5*1 + 3*2)/(1+4) = 1.3.
        self.assertAlmostEqual(projection_movement(source, disguised, target), 1.3)
        # Net overshoot -> clipped to fully disguised (persistence 0), unlike the
        # per-axis average (0.25), which is the whole point of the projection switch.
        self.assertAlmostEqual(projection_persistence(source, disguised, target), 0.0)

    def test_projection_is_rotation_invariant(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.standard_normal((6, 3))
        target = source + np.array([1.0, -0.5, 2.0])
        disguised = source + np.array([0.4, -0.2, 0.8])
        base = projection_movement(source, disguised, target)
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        rotated = projection_movement(source @ rotation.T, disguised @ rotation.T, target @ rotation.T)
        self.assertAlmostEqual(base, rotated, places=10)
        # Per-row projection with fixed endpoints matches the aggregate on aligned means.
        per_row = projection_movement_per_row(
            disguised, source_mean=source.mean(0), target_mean=target.mean(0)
        )
        self.assertAlmostEqual(float(np.mean(per_row)), base, places=10)

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
        self.assertIn("weighted_axis_persistence", summary)
        self.assertIn("projection_persistence", summary)
        self.assertIn("projection_persistence_all", summary)
        self.assertIn("target_assimilation", summary)
        self.assertTrue(summary["separable"])

    def test_source_response_join_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison = root / "comparison.csv"
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "prompt": ["p1", "p2"],
                    "model_response": ["d1", "d2"],
                    "target_response": ["t1", "t2"],
                }
            ).to_csv(comparison, index=False)
            pd.DataFrame({"prompt": ["p1"], "model_response": ["s1"]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "missing 1 prompts"):
                normalize_comparison_df(comparison, source_responses=source)

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
        self.assertFalse(summary["trustworthy"])  # not separable -> not trustworthy
        self.assertTrue(np.isnan(summary["probe_cv"]))

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
        self.assertIn("projection_persistence", boot_df.columns)
        self.assertIn("projection_persistence_ci_low", summary)


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

    def test_length_residualization_removes_length(self) -> None:
        texts = {
            "source": ["short.", "tiny one."],
            "disguised": [
                "a much longer answer with many more words here indeed yes truly",
                "another quite long verbose response with plenty of extra words",
            ],
            "target": ["medium length response here now", "second medium length response"],
        }
        base, names = build_descriptor_matrix(texts, descriptor_mode="style_only", feature_set="style_all")
        res, res_names = build_descriptor_matrix(
            texts, descriptor_mode="style_only", feature_set="style_all_lenres"
        )
        self.assertEqual(names, res_names)
        wc = names.index("style_word_count")
        raw = np.vstack([base[c] for c in ("source", "disguised", "target")])[:, wc]
        out = np.vstack([res[c] for c in ("source", "disguised", "target")])[:, wc]
        # Word count is collinear with the length covariate, so it residualizes to ~0.
        self.assertGreater(float(np.ptp(raw)), 0.5)
        self.assertLess(float(np.ptp(out)), 1e-6)

    def test_sep_ratio_in_config(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        with tempfile.TemporaryDirectory() as tmp:
            _scores, _loads, config = run_latent_analysis(
                str(fixture),
                output_dir=tmp,
                descriptor_mode="style_only",
                feature_set="style_all",
                k=3,
            )
            self.assertIn("sep_ratio", config)
            ratio = config["sep_ratio"]
            self.assertTrue(0.0 <= ratio <= 1.0 + 1e-9)
            self.assertGreaterEqual(config["sep_full"], config["sep_captured"] - 1e-9)

    def test_supervised_basis_captures_separation_a_variance_pc_misses(self) -> None:
        # 9 correlated high-variance nuisance features + 1 low-variance feature that
        # carries the whole source->target mean shift. A variance PC1 locks onto the
        # nuisance subspace and misses the separation; the supervised axis finds it.
        rng = np.random.default_rng(0)
        n = 60
        z_src = rng.normal(0, 1, n)
        z_tgt = rng.normal(0, 1, n)
        nuis_src = np.outer(z_src, np.ones(9)) + rng.normal(0, 0.1, (n, 9))
        nuis_tgt = np.outer(z_tgt, np.ones(9)) + rng.normal(0, 0.1, (n, 9))
        src = np.hstack([nuis_src, rng.normal(0.0, 0.2, (n, 1))])
        tgt = np.hstack([nuis_tgt, rng.normal(1.0, 0.2, (n, 1))])
        matrices = {"source": src, "target": tgt, "disguised": (src + tgt) / 2.0}
        names = [f"f{i}" for i in range(10)]

        def captured(basis) -> float:
            d = basis.scaler.transform(tgt).mean(0) - basis.scaler.transform(src).mean(0)
            return float(np.linalg.norm(basis.components @ d) / (np.linalg.norm(d) + 1e-12))

        kwargs = dict(descriptor_mode="style_only", encoder_model="e", feature_set="style_scalars")
        var_basis = fit_behavioral_axis_basis(matrices, names, k=1, **kwargs)
        sup_basis = fit_behavioral_axis_basis(matrices, names, k=1, basis_type="supervised", **kwargs)
        var_capture, sup_capture = captured(var_basis), captured(sup_basis)
        self.assertGreater(sup_capture, var_capture)
        self.assertGreater(sup_capture, 0.7)
        self.assertEqual(sup_basis.basis_type, "supervised")

        # Supervised basis rows stay orthonormal (so sep_ratio is a
        # valid orthogonal projection).
        sup3 = fit_behavioral_axis_basis(matrices, names, k=3, basis_type="supervised", **kwargs)
        gram = sup3.components @ sup3.components.T
        np.testing.assert_allclose(gram, np.eye(sup3.components.shape[0]), atol=1e-8)

    def test_big5_dimension_movement_reports_direct_plasticity(self) -> None:
        feature_names = [
            "talkative",
            "bold",
            "withdrawn",
            "quiet",
            "organized",
            "efficient",
            "careless",
            "disorganized",
        ]
        matrices = {
            "source": np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]),
            "disguised": np.array([[0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.25, 0.25]]),
            "target": np.array([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
        }
        df = pd.DataFrame({"prompt": ["p"]})
        scores = big5_dimension_scores_df(df, matrices, feature_names)
        movement = summarize_big5_dimension_movement(scores)
        by_dim = movement.set_index("dimension")
        self.assertAlmostEqual(by_dim.loc["EXT", "movement_clipped"], 0.5)
        self.assertAlmostEqual(by_dim.loc["CON", "movement_clipped"], 0.75)
        self.assertEqual(by_dim.loc["CON", "label"], "Conscientiousness")


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

    def test_persistence_is_k_independent(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        values = []
        for k in (2, 8):
            with tempfile.TemporaryDirectory() as tmp:
                summary = run_behavioral_inertia(
                    comparison_csv=str(fixture),
                    output_dir=tmp,
                    feature_set="style_all",
                    k=k,
                    bootstrap_samples=0,
                    basis_type="supervised",
                )
                values.append(summary["persistence"])
        # The full-feature axis does not depend on the PC subspace, so persistence
        # is identical across k (unlike the PC-space projection).
        self.assertAlmostEqual(values[0], values[1], places=10)


class BehavioralCellEvaluatorTests(unittest.TestCase):
    def test_cell_runner_reuses_basis_and_writes_validation_artifacts(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        source = pd.read_csv(fixture)[["prompt", "source_response"]]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run0 = root / "source_seed0.csv"
            run1 = root / "source_seed1.csv"
            source.rename(columns={"source_response": "model_response"}).to_csv(run0, index=False)
            varied = source.copy()
            varied["source_response"] = varied["source_response"] + " Short answer."
            varied.rename(columns={"source_response": "model_response"}).to_csv(run1, index=False)

            activation_summary = root / "activation_summary.json"
            activation_summary.write_text(
                json.dumps(
                    {
                        "activation_bridge_mode": "fixed-encoder",
                        "encoder_model": "fixture-encoder",
                        "best_layer": 2,
                        "probe_cv": 0.8,
                        "activation_source_prob": 0.7,
                        "activation_target_prob": 0.3,
                    }
                ),
                encoding="utf-8",
            )

            scored_a = root / "calibration_a.csv"
            scored_b = root / "calibration_b.csv"
            pd.DataFrame(
                {
                    "semantic_score": [3.0, 3.5],
                    "stylistic_score": [2.0, 2.5],
                    "heuristic_match_score": [0.4, 0.5],
                }
            ).to_csv(scored_a, index=False)
            pd.DataFrame(
                {
                    "semantic_score": [3.5, 4.0],
                    "stylistic_score": [3.0, 3.5],
                    "heuristic_match_score": [0.6, 0.7],
                }
            ).to_csv(scored_b, index=False)

            spec = BehavioralCellSpec(
                dataset="fixture",
                source_model="source-test",
                target_model="target-test",
                output_dir=str(root / "cell"),
                basis_reference=MethodSpec(method="basis_ref", comparison_csv=str(fixture)),
                feature_set="style_all",
                feature_ablation_sets=["style_scalars"],
                k=3,
                bootstrap_samples=3,
                methods=[
                    MethodSpec(
                        method="contrastive",
                        comparison_csv=str(fixture),
                        activation_summary=str(activation_summary),
                        calibration_scored_csv=str(scored_a),
                    ),
                    MethodSpec(
                        method="behavioral",
                        comparison_csv=str(fixture),
                        calibration_scored_csv=str(scored_b),
                    ),
                ],
                self_baseline=SelfBaselineSpec(source_runs=[str(run0), str(run1)]),
            )
            summary = run_behavioral_cell(spec)
            out = Path(summary["output_dir"])

            self.assertTrue((out / "basis" / "behavioral_axis_basis.pkl").exists())
            self.assertTrue((out / "cell_summary.csv").exists())
            self.assertTrue((out / "paired_method_comparisons.csv").exists())
            self.assertTrue((out / "self_baseline" / "summary.json").exists())
            self.assertTrue((out / "calibration" / "calibration_summary.csv").exists())
            self.assertTrue((out / "feature_ablations" / "feature_ablation_stability.csv").exists())
            self.assertTrue(summary["has_self_baseline"])
            self.assertTrue(summary["has_paired_method_comparisons"])
            self.assertTrue(summary["has_calibration"])
            self.assertTrue(summary["has_feature_ablations"])
            self.assertEqual(summary["basis_reference"], "basis_ref")

            cell_df = pd.read_csv(out / "cell_summary.csv")
            self.assertEqual(cell_df["basis_path"].nunique(), 1)
            self.assertTrue(cell_df["norm_persistence"].notna().all())
            self.assertIn("weighted_axis_persistence", cell_df.columns)

            method_summary = json.loads((out / "methods" / "00_contrastive" / "summary.json").read_text())
            self.assertEqual(method_summary["activation_bridge_mode"], "fixed-encoder")
            self.assertEqual(method_summary["activation_probe_cv"], 0.8)

            paired = pd.read_csv(out / "paired_method_comparisons.csv")
            self.assertEqual(len(paired), 1)
            self.assertIn("paired_t_q_bh", paired.columns)

    def test_identity_control_anchored_metric_and_lenres_ablation(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        frame = pd.read_csv(fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run0 = root / "source_seed0.csv"
            run1 = root / "source_seed1.csv"
            frame[["prompt", "source_response"]].rename(
                columns={"source_response": "model_response"}
            ).to_csv(run0, index=False)
            varied = frame[["prompt", "source_response"]].copy()
            varied["model_response"] = varied["source_response"] + " A bit longer."
            varied[["prompt", "model_response"]].to_csv(run1, index=False)

            # Independent target draw for the positive control.
            target_seed = root / "target_seed.csv"
            tgt = frame[["prompt", "target_response"]].copy()
            tgt["model_response"] = tgt["target_response"] + " Slight variation."
            tgt[["prompt", "model_response"]].to_csv(target_seed, index=False)

            spec = BehavioralCellSpec(
                dataset="fixture",
                source_model="source-test",
                target_model="target-test",
                output_dir=str(root / "cell"),
                feature_set="style_all",
                feature_ablation_sets=[],
                k=3,
                bootstrap_samples=4,
                methods=[
                    MethodSpec(method="contrastive", comparison_csv=str(fixture)),
                    MethodSpec(method="behavioral", comparison_csv=str(fixture)),
                ],
                self_baseline=SelfBaselineSpec(source_runs=[str(run0), str(run1)]),
                identity_control=IdentityControlSpec(target_runs=[str(target_seed)]),
            )
            summary = run_behavioral_cell(spec)
            out = Path(summary["output_dir"])

            self.assertTrue(summary["has_identity_control"])
            self.assertTrue(summary["has_anchored_bootstrap"])
            self.assertIsNotNone(summary["identity_persistence"])
            self.assertTrue((out / "identity_control" / "summary.json").exists())
            self.assertIn("shared_active_axes", summary)

            cell_df = pd.read_csv(out / "cell_summary.csv")
            for column in (
                "projection_persistence",
                "anchored",
                "identity_anchor",
                "baseline_anchor",
                "z_vs_baseline",
                "sep_ratio",
            ):
                self.assertIn(column, cell_df.columns)

            # The length-residualized twin of the headline feature set is always run.
            long_df = pd.read_csv(out / "feature_ablations" / "feature_ablation_long.csv")
            self.assertIn("style_all_lenres", set(long_df["feature_set"].astype(str)))

    def test_enforce_shared_endpoints_raises_on_mismatch(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        frame = pd.read_csv(fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Method B has a different target distribution than the reference.
            mismatch = root / "mismatch.csv"
            other = frame.copy()
            other["target_response"] = other["target_response"] + " EXTRA TARGET STYLE TEXT HERE."
            other.to_csv(mismatch, index=False)

            spec = BehavioralCellSpec(
                dataset="fixture",
                source_model="source-test",
                target_model="target-test",
                output_dir=str(root / "cell"),
                basis_reference=MethodSpec(method="basis_ref", comparison_csv=str(fixture)),
                feature_set="style_all",
                feature_ablation_sets=[],
                k=3,
                bootstrap_samples=0,
                methods=[
                    MethodSpec(method="contrastive", comparison_csv=str(fixture)),
                    MethodSpec(method="behavioral", comparison_csv=str(mismatch)),
                ],
            )
            with self.assertRaisesRegex(ValueError, "endpoints differ from the cell reference"):
                run_behavioral_cell(spec)

    def test_lenres_endpoints_invariant_to_row_order(self) -> None:
        # Methods with identical source/target but different row order must still
        # share endpoints under the auto length-residualized ablation. Length
        # residualization makes word-count collinear/near-constant, and dividing by
        # its ~0 std used to amplify float noise into a spurious endpoint mismatch.
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        frame = pd.read_csv(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = frame.copy()
            a["model_response"] = a["source_response"] + " A"
            b = frame.sample(frac=1.0, random_state=1).reset_index(drop=True).copy()
            b["model_response"] = b["target_response"] + " B"
            a.to_csv(root / "a.csv", index=False)
            b.to_csv(root / "b.csv", index=False)
            spec = BehavioralCellSpec(
                dataset="fx",
                source_model="s",
                target_model="tg",
                output_dir=str(root / "cell"),
                feature_set="style_all",
                feature_ablation_sets=[],
                k=3,
                bootstrap_samples=0,
                methods=[
                    MethodSpec(method="a", comparison_csv=str(root / "a.csv")),
                    MethodSpec(method="b", comparison_csv=str(root / "b.csv")),
                ],
            )
            summary = run_behavioral_cell(spec)  # must not raise the endpoint assertion
            self.assertTrue(summary["has_feature_ablations"])

    def test_multi_seed_baseline_and_identity_pooling(self) -> None:
        fixture = Path("scripts/tests/fixtures/behavioral_inertia_comparison.csv")
        frame = pd.read_csv(fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Three source seeds -> two seed pairs -> duplicate prompts in the pool.
            source_runs = []
            for i in range(3):
                path = root / f"src_seed{i}.csv"
                df = frame[["prompt", "source_response"]].copy()
                df["model_response"] = df["source_response"] + (f" variant {i}." if i else "")
                df[["prompt", "model_response"]].to_csv(path, index=False)
                source_runs.append(str(path))
            # Two independent target seeds pooled for the identity control.
            target_runs = []
            for i in range(2):
                path = root / f"tgt_seed{i}.csv"
                df = frame[["prompt", "target_response"]].copy()
                df["model_response"] = df["target_response"] + f" t{i}."
                df[["prompt", "model_response"]].to_csv(path, index=False)
                target_runs.append(str(path))

            spec = BehavioralCellSpec(
                dataset="fixture",
                source_model="source-test",
                target_model="target-test",
                output_dir=str(root / "cell"),
                feature_set="style_all",
                feature_ablation_sets=[],
                k=3,
                bootstrap_samples=4,
                methods=[MethodSpec(method="contrastive", comparison_csv=str(fixture))],
                self_baseline=SelfBaselineSpec(source_runs=source_runs),
                identity_control=IdentityControlSpec(target_runs=target_runs),
            )
            # Would raise on the duplicate-prompt guard before the multi-seed fix.
            summary = run_behavioral_cell(spec)
            out = Path(summary["output_dir"])

            self.assertTrue(summary["has_self_baseline"])
            self.assertTrue(summary["has_identity_control"])
            baseline_comp = pd.read_csv(out / "self_baseline" / "self_baseline_comparison.csv")
            identity_comp = pd.read_csv(out / "identity_control" / "identity_comparison.csv")
            self.assertTrue(baseline_comp["prompt"].duplicated().any())
            self.assertTrue(identity_comp["prompt"].duplicated().any())
            cell_df = pd.read_csv(out / "cell_summary.csv")
            self.assertIn("anchored", cell_df.columns)


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
