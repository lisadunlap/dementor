"""CPU-only unit tests for the activation-steering module.

Everything here runs on CPU with tiny fake tensors and mocks: no real model is
loaded and nothing is downloaded. The model-dependent activation collection is
mocked out so we test the pure diff-of-means / hook logic directly.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from dementor.steering.activation_bridge import _parse_layers
from dementor.steering.activation_steering import (
    compute_steering_vector,
    parse_strengths,
    resolve_layer_index,
    steering_hook,
)


# ---------------------------------------------------------------------------
# Tiny fake model used by the hook tests. get_transformer_layers() walks a list
# of candidate attribute paths; the first is ("model", "layers"), so exposing
# `.model.layers` as a ModuleList is enough to register hooks on real blocks.
# ---------------------------------------------------------------------------
class _IdentityBlock(torch.nn.Module):
    """Minimal stand-in for a transformer block: returns its input unchanged.

    Optionally wraps the output in a tuple to mimic HF decoder blocks that
    return ``(hidden_states, ...)`` rather than a bare tensor.
    """

    def __init__(self, return_tuple: bool = False) -> None:
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden):
        return (hidden,) if self.return_tuple else hidden


class _FakeInner(torch.nn.Module):
    def __init__(self, n_layers: int, return_tuple: bool) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_IdentityBlock(return_tuple) for _ in range(n_layers)]
        )


class _FakeModel(torch.nn.Module):
    """Exposes ``.model.layers`` to match get_transformer_layers' first path."""

    def __init__(self, n_layers: int = 4, return_tuple: bool = False) -> None:
        super().__init__()
        self.model = _FakeInner(n_layers, return_tuple)


class ParseStrengthsTests(unittest.TestCase):
    def test_basic_and_whitespace(self) -> None:
        self.assertEqual(parse_strengths("0,0.5,1,2"), [0.0, 0.5, 1.0, 2.0])
        self.assertEqual(parse_strengths("  0 , 0.5 , 1 , 2  "), [0.0, 0.5, 1.0, 2.0])

    def test_skips_empty_parts(self) -> None:
        # Trailing / duplicate commas produce empty parts that are filtered out.
        self.assertEqual(parse_strengths("1,,2,"), [1.0, 2.0])

    def test_negative_values_allowed(self) -> None:
        self.assertEqual(parse_strengths("-1,2.5"), [-1.0, 2.5])

    def test_empty_raises(self) -> None:
        for value in ("", "   ", ",,"):
            with self.assertRaises(ValueError):
                parse_strengths(value)


class ResolveLayerIndexTests(unittest.TestCase):
    def test_positive_passthrough(self) -> None:
        self.assertEqual(resolve_layer_index(0, 12), 0)
        self.assertEqual(resolve_layer_index(3, 12), 3)
        self.assertEqual(resolve_layer_index(11, 12), 11)

    def test_negative_indexing(self) -> None:
        self.assertEqual(resolve_layer_index(-1, 12), 11)
        self.assertEqual(resolve_layer_index(-8, 12), 4)  # the module's default layer
        self.assertEqual(resolve_layer_index(-12, 12), 0)

    def test_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_layer_index(12, 12)  # positive overflow
        with self.assertRaises(ValueError):
            resolve_layer_index(-13, 12)  # negative underflow (wraps below 0)


class ComputeSteeringVectorTests(unittest.TestCase):
    """compute_steering_vector is diff-of-means at a layer.

    collect_layer_activations needs a real forward pass, so we mock it: the
    function calls it for the source texts first, then the target texts. We feed
    fixed fake activation arrays and check the returned vector == mean(target) -
    mean(source).
    """

    def _df(self) -> pd.DataFrame:
        # Text content is irrelevant because collect_layer_activations is mocked.
        return pd.DataFrame(
            {
                "prompt": ["p0", "p1", "p2"],
                "source_response": ["s0", "s1", "s2"],
                "target_response": ["t0", "t1", "t2"],
            }
        )

    def _compute(self, source: np.ndarray, target: np.ndarray) -> dict:
        with mock.patch(
            "dementor.steering.activation_steering.collect_layer_activations",
            side_effect=[source, target],  # source call, then target call
        ):
            return compute_steering_vector(
                df=self._df(),
                tokenizer=None,
                model=None,
                layer=0,
                device="cpu",
                batch_size=2,
                max_length=8,
                pool="last",
            )

    def test_diff_of_means_shape_and_values(self) -> None:
        source = np.array(
            [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )  # mean -> [2, 0, 0, 0]
        target = np.array(
            [[1.0, 2.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0], [5.0, 2.0, 0.0, 0.0]],
            dtype=np.float32,
        )  # mean -> [3, 2, 0, 0]
        out = self._compute(source, target)

        self.assertEqual(out["vector"].shape, (4,))
        np.testing.assert_allclose(out["vector"], [1.0, 2.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out["source_mean"], [2.0, 0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out["target_mean"], [3.0, 2.0, 0.0, 0.0], atol=1e-6)
        # Norm and the unit vector are derived from the raw diff-of-means.
        expected_norm = float(np.sqrt(5.0))
        self.assertAlmostEqual(out["vector_norm"], expected_norm, places=5)
        np.testing.assert_allclose(
            out["unit_vector"],
            np.array([1.0, 2.0, 0.0, 0.0]) / expected_norm,
            atol=1e-6,
        )

    def test_swapping_source_target_negates_vector(self) -> None:
        source = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        target = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=np.float32)
        forward = self._compute(source, target)
        swapped = self._compute(target, source)  # roles reversed
        np.testing.assert_allclose(swapped["vector"], -forward["vector"], atol=1e-6)


class SteeringHookTests(unittest.TestCase):
    """steering_hook registers a forward hook that adds vector * strength to the
    residual stream and removes the hook on context exit (restoration)."""

    def test_applies_and_restores_last_token(self) -> None:
        model = _FakeModel(n_layers=4)
        layer = 2
        vector = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32)
        strength = 2.0
        x = torch.zeros(1, 3, 4)  # (batch, seq, hidden)

        baseline = model.model.layers[layer](x)
        np.testing.assert_allclose(baseline.numpy(), x.numpy())

        with steering_hook(
            model=model,
            layer=layer,
            vector=vector,
            strength=strength,
            token_position="last",
            device="cpu",
        ):
            steered = model.model.layers[layer](x)

        expected = x.clone()
        expected[:, -1, :] += torch.as_tensor(vector) * strength
        np.testing.assert_allclose(steered.numpy(), expected.numpy(), atol=1e-6)
        # Only the final token position moves under token_position="last".
        np.testing.assert_allclose(steered[:, :-1, :].numpy(), x[:, :-1, :].numpy())

        # After the context exits, the hook is removed -> original output restored.
        restored = model.model.layers[layer](x)
        np.testing.assert_allclose(restored.numpy(), baseline.numpy(), atol=1e-6)

    def test_all_token_position_shifts_every_position(self) -> None:
        model = _FakeModel(n_layers=3)
        layer = 1
        vector = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
        x = torch.zeros(2, 3, 4)
        with steering_hook(
            model=model,
            layer=layer,
            vector=vector,
            strength=1.0,
            token_position="all",
            device="cpu",
        ):
            steered = model.model.layers[layer](x)
        expected = x + torch.as_tensor(vector).view(1, 1, -1)
        np.testing.assert_allclose(steered.numpy(), expected.numpy(), atol=1e-6)

    def test_tuple_output_block(self) -> None:
        # HF decoder blocks return tuples; the hook edits element 0 and keeps the
        # rest, returning a tuple.
        model = _FakeModel(n_layers=3, return_tuple=True)
        layer = 0
        vector = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        x = torch.zeros(1, 2, 4)
        with steering_hook(
            model=model,
            layer=layer,
            vector=vector,
            strength=1.0,
            token_position="all",
            device="cpu",
        ):
            out = model.model.layers[layer](x)
        self.assertIsInstance(out, tuple)
        np.testing.assert_allclose(out[0].numpy(), (x + 1.0).numpy(), atol=1e-6)

    def test_two_dim_hidden(self) -> None:
        # Some blocks emit a 2D (batch, hidden) tensor; the hook adds to every row.
        model = _FakeModel(n_layers=2)
        layer = 0
        vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x = torch.zeros(3, 4)
        with steering_hook(
            model=model,
            layer=layer,
            vector=vector,
            strength=1.0,
            token_position="last",
            device="cpu",
        ):
            steered = model.model.layers[layer](x)
        expected = x + torch.as_tensor(vector).view(1, -1)
        np.testing.assert_allclose(steered.numpy(), expected.numpy(), atol=1e-6)

    def test_negative_layer_resolves_to_correct_block(self) -> None:
        # steering_hook runs resolve_layer_index internally, so layer=-1 must
        # hook the last block and leave the others untouched.
        model = _FakeModel(n_layers=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x = torch.zeros(1, 2, 4)
        with steering_hook(
            model=model,
            layer=-1,
            vector=vector,
            strength=1.0,
            token_position="last",
            device="cpu",
        ):
            top = model.model.layers[3](x)
            other = model.model.layers[0](x)
        self.assertAlmostEqual(float(top[0, -1, 0]), 1.0)
        np.testing.assert_allclose(other.numpy(), x.numpy())


class RandomDirectionControlTests(unittest.TestCase):
    def test_hook_applies_whatever_direction_it_is_given(self) -> None:
        # The methodology's control swaps the diff-of-means vector for a random
        # vector of the SAME norm. That control is only valid because the hook is
        # direction-agnostic: it adds exactly the vector handed to it. We verify
        # both directions are applied verbatim and move outputs apart.
        model = _FakeModel(n_layers=2)
        layer = 1
        diff_vector = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        rand = rng.standard_normal(4).astype(np.float32)
        rand = rand / np.linalg.norm(rand) * np.linalg.norm(diff_vector)  # match norm
        self.assertAlmostEqual(
            float(np.linalg.norm(rand)), float(np.linalg.norm(diff_vector)), places=5
        )

        x = torch.zeros(1, 2, 4)
        with steering_hook(
            model=model,
            layer=layer,
            vector=diff_vector,
            strength=1.0,
            token_position="last",
            device="cpu",
        ):
            out_diff = model.model.layers[layer](x)[0, -1, :].clone()
        with steering_hook(
            model=model,
            layer=layer,
            vector=rand,
            strength=1.0,
            token_position="last",
            device="cpu",
        ):
            out_rand = model.model.layers[layer](x)[0, -1, :].clone()

        # Each hook applied exactly its own vector (zero baseline input).
        np.testing.assert_allclose(out_diff.numpy(), diff_vector, atol=1e-6)
        np.testing.assert_allclose(out_rand.numpy(), rand, atol=1e-6)
        # Equal-magnitude interventions, different directions -> different outputs.
        self.assertFalse(np.allclose(out_diff.numpy(), out_rand.numpy()))


class ParseLayersHelperTests(unittest.TestCase):
    """_parse_layers is the one pure helper in activation_bridge worth covering."""

    def test_explicit_with_negative_index(self) -> None:
        self.assertEqual(_parse_layers("0,2,-1", 5), [0, 2, 4])

    def test_default_quartiles(self) -> None:
        defaults = _parse_layers(None, 9)
        self.assertEqual(defaults[0], 0)
        self.assertEqual(defaults[-1], 8)

    def test_all_expands_range(self) -> None:
        self.assertEqual(_parse_layers("all", 3), [0, 1, 2])

    def test_clamps_and_dedupes(self) -> None:
        self.assertEqual(_parse_layers("100", 5), [4])  # clamped to n-1
        self.assertEqual(_parse_layers("-100", 5), [0])  # clamped to 0
        self.assertEqual(_parse_layers("1,1,2", 5), [1, 2])  # de-duplicated


if __name__ == "__main__":
    unittest.main()
