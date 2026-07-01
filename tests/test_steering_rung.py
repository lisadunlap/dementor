"""CPU-only unit tests for the activation-steering rung (steering_rung.py).

Everything runs on CPU with tiny tensors and mocks: no real model is loaded and
nothing is downloaded. We test the pure hook math (projection-ablation removes the
v-component; additive adds alpha*v_hat) on real ``register_forward_hook`` blocks, and
``derive_steering_vector``'s diff-of-means with the model load + activation capture
mocked out. Style mirrors tests/test_steering.py and tests/test_local_backend.py.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from dementor.steering.steering_rung import (
    derive_steering_vector,
    make_ablation_hook,
    make_additive_hook,
)


class _IdentityBlock(torch.nn.Module):
    """Minimal transformer-block stand-in: returns input unchanged (optionally in a tuple)."""

    def __init__(self, return_tuple: bool = False) -> None:
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden):
        return (hidden,) if self.return_tuple else hidden


class _FakeInner(torch.nn.Module):
    def __init__(self, n_layers: int, return_tuple: bool) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_IdentityBlock(return_tuple) for _ in range(n_layers)])


class _FakeModel(torch.nn.Module):
    """Exposes ``.model.layers`` so get_transformer_layers finds real blocks to hook."""

    def __init__(self, n_layers: int = 4, return_tuple: bool = False) -> None:
        super().__init__()
        self.model = _FakeInner(n_layers, return_tuple)


class AblationHookTests(unittest.TestCase):
    """h <- h - beta*(h.v_hat)v_hat  =>  post-hook projection == (1-beta) * original."""

    def _proj(self, h: torch.Tensor, v_unit: torch.Tensor) -> torch.Tensor:
        return h.float() @ v_unit

    def test_removes_v_component_by_factor_one_minus_beta(self) -> None:
        torch.manual_seed(0)
        hidden = 8
        v = torch.randn(hidden)
        v_unit = v / v.norm()
        h = torch.randn(2, 5, hidden)  # (batch, seq, hidden)
        orig_proj = self._proj(h, v_unit)

        for beta in (0.0, 0.5, 0.7, 1.0, 1.5):
            hook = make_ablation_hook(v, beta)
            h_new = hook(None, None, h)
            new_proj = self._proj(h_new, v_unit)
            # The component ALONG v_hat is scaled by (1 - beta) everywhere.
            torch.testing.assert_close(new_proj, (1.0 - beta) * orig_proj, atol=1e-5, rtol=1e-4)

    def test_beta_one_fully_zeroes_component(self) -> None:
        hidden = 6
        v = torch.arange(1, hidden + 1, dtype=torch.float32)
        v_unit = v / v.norm()
        h = torch.randn(3, hidden)
        h_new = make_ablation_hook(v, 1.0)(None, None, h)
        proj = h_new.float() @ v_unit
        torch.testing.assert_close(proj, torch.zeros_like(proj), atol=1e-5, rtol=0)

    def test_component_orthogonal_to_v_is_preserved(self) -> None:
        # Ablation must only touch the v direction; the orthogonal remainder is unchanged.
        hidden = 5
        v = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        h = torch.randn(4, hidden)
        h_new = make_ablation_hook(v, 0.7)(None, None, h)
        torch.testing.assert_close(h_new[:, 1:], h[:, 1:], atol=1e-6, rtol=0)

    def test_tuple_output_preserved(self) -> None:
        hidden = 4
        v = torch.ones(hidden)
        h = torch.randn(1, 3, hidden)
        out = make_ablation_hook(v, 0.5)(None, None, (h, "kv-cache", 123))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[1:], ("kv-cache", 123))

    def test_registered_forward_hook_applies_and_restores(self) -> None:
        model = _FakeModel(n_layers=4)
        layer, hidden, beta = 2, 6, 0.7
        v = torch.randn(hidden)
        v_unit = v / v.norm()
        x = torch.randn(1, 3, hidden)

        handle = model.model.layers[layer].register_forward_hook(make_ablation_hook(v, beta))
        try:
            steered = model.model.layers[layer](x)
        finally:
            handle.remove()
        torch.testing.assert_close(steered.float() @ v_unit, (1 - beta) * (x.float() @ v_unit), atol=1e-5, rtol=1e-4)
        # Other layers untouched; hook removed after handle.remove().
        torch.testing.assert_close(model.model.layers[0](x), x, atol=0, rtol=0)
        torch.testing.assert_close(model.model.layers[layer](x), x, atol=0, rtol=0)


class AdditiveHookTests(unittest.TestCase):
    """h <- h + alpha*v_hat at every token."""

    def test_adds_alpha_unit_vector(self) -> None:
        hidden = 5
        v = torch.randn(hidden)
        v_unit = v / v.norm()
        for alpha in (0.0, 2.0, -3.0):
            h = torch.randn(2, 4, hidden)
            h_new = make_additive_hook(v, alpha)(None, None, h)
            torch.testing.assert_close(h_new, h + alpha * v_unit, atol=1e-5, rtol=1e-4)

    def test_added_norm_equals_alpha(self) -> None:
        # alpha is in residual-norm units: the added vector has L2 norm == |alpha|.
        hidden = 7
        v = torch.randn(hidden)
        h = torch.zeros(1, 1, hidden)
        h_new = make_additive_hook(v, 3.0)(None, None, h)
        self.assertAlmostEqual(float(h_new.norm()), 3.0, places=4)

    def test_tuple_output_preserved(self) -> None:
        hidden = 4
        v = torch.ones(hidden)
        h = torch.zeros(1, 2, hidden)
        out = make_additive_hook(v, 2.0)(None, None, (h, "aux"))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[1], "aux")


class DeriveSteeringVectorTests(unittest.TestCase):
    """derive_steering_vector = per-layer (target_mean - source_mean); model load + the
    teacher-forced activation capture are mocked, so we test the pure diff-of-means/shape."""

    def _run(self, source_means: dict[int, torch.Tensor], target_means: dict[int, torch.Tensor]):
        with mock.patch(
            "dementor.steering.steering_rung._load_causal_encoder",
            return_value=(object(), object(), "cpu", 4),  # (tok, model, device, n_layers)
        ), mock.patch(
            "dementor.steering.steering_rung._collect_response_means",
            side_effect=[source_means, target_means],  # source call, then target call
        ):
            return derive_steering_vector(
                "fake/source",
                "fake/target",
                prompts=["p0", "p1"],
                source_responses=["s0", "s1"],
                target_responses=["t0", "t1"],
                layers=[0, 2],
            )

    def test_shapes_and_diff_of_means(self) -> None:
        hidden = 4
        source = {0: torch.zeros(hidden), 2: torch.ones(hidden)}
        target = {0: torch.arange(hidden, dtype=torch.float32), 2: torch.ones(hidden) * 3}
        out = self._run(source, target)

        self.assertEqual(sorted(out), [0, 2])
        for layer_idx in (0, 2):
            self.assertEqual(tuple(out[layer_idx].shape), (hidden,))
            self.assertEqual(out[layer_idx].dtype, torch.float32)
        torch.testing.assert_close(out[0], torch.arange(hidden, dtype=torch.float32))  # target - 0
        torch.testing.assert_close(out[2], torch.ones(hidden) * 2.0)  # 3 - 1

    def test_swapping_source_target_negates(self) -> None:
        hidden = 3
        a = {0: torch.zeros(hidden), 2: torch.tensor([1.0, 2.0, 3.0])}
        b = {0: torch.tensor([1.0, 1.0, 1.0]), 2: torch.tensor([4.0, 4.0, 4.0])}
        forward = self._run(a, b)
        swapped = self._run(b, a)
        for layer_idx in (0, 2):
            torch.testing.assert_close(swapped[layer_idx], -forward[layer_idx])

    def test_default_layers_covers_all(self) -> None:
        # layers=None -> all 4 decoder layers (from the mocked n_layers=4).
        hidden = 2
        src = {L: torch.zeros(hidden) for L in range(4)}
        tgt = {L: torch.full((hidden,), float(L)) for L in range(4)}
        with mock.patch(
            "dementor.steering.steering_rung._load_causal_encoder",
            return_value=(object(), object(), "cpu", 4),
        ), mock.patch(
            "dementor.steering.steering_rung._collect_response_means",
            side_effect=[src, tgt],
        ):
            out = derive_steering_vector(
                "fake/source", "fake/target",
                prompts=["p"], source_responses=["s"], target_responses=["t"],
            )
        self.assertEqual(sorted(out), [0, 1, 2, 3])
        torch.testing.assert_close(out[3], torch.full((hidden,), 3.0))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            derive_steering_vector(
                "fake/source", "fake/target",
                prompts=["p0", "p1"], source_responses=["s0"], target_responses=["t0", "t1"],
            )


class RandomDirectionControlTests(unittest.TestCase):
    """The methodology's control swaps the diff-of-means vector for a random vector of the
    SAME norm. Valid only because the hooks are direction-agnostic -- verify verbatim."""

    def test_additive_applies_whatever_direction(self) -> None:
        hidden = 6
        diff = torch.randn(hidden)
        rng = np.random.default_rng(0)
        rand = torch.tensor(rng.standard_normal(hidden), dtype=torch.float32)
        rand = rand / rand.norm() * diff.norm()  # match norm
        h = torch.zeros(1, 1, hidden)
        out_diff = make_additive_hook(diff, 1.0)(None, None, h)[0, -1]
        out_rand = make_additive_hook(rand, 1.0)(None, None, h)[0, -1]
        torch.testing.assert_close(out_diff, diff / diff.norm())
        torch.testing.assert_close(out_rand, rand / rand.norm())
        self.assertFalse(torch.allclose(out_diff, out_rand))


if __name__ == "__main__":
    unittest.main()
