"""
Tests for Phase 2 layers.

Linear, Conv2d, BatchNorm2d, and LayerNorm are all compositions of
already-gradient-checked ops (see ops.py's docstring on why that's
sufficient in principle) -- but a composition can still have a bug
(wrong axis, wrong broadcast shape) that per-op checks can't catch.
So each one gets its own finite-difference check here, treating the
whole layer -- weights AND input -- as the function under test. This
is the same rigor as test_grad_check.py, one level up the stack.

Dropout is stochastic, so it's checked deterministically instead (via
a fixed random seed) rather than by finite differences.

Note: reduction to a scalar uses a fixed random projection, not a
plain .sum() -- see gradus.utils.grad_check.check_gradient's
docstring for why (a raw sum degenerates for mean-subtracting layers
like LayerNorm/BatchNorm2d, since their output sums to ~0 for any
input by construction, which was discovered by this exact test suite
failing and led to that harness-level fix).
"""

import numpy as np
from _helpers import check_module_gradients, gradient_ok

from gradus import Tensor
from gradus.nn import BatchNorm2d, Conv2d, Dropout, Embedding, LayerNorm, Linear

RNG = np.random.default_rng(0)


def test_linear_forward_shape():
    layer = Linear(4, 3)
    x = Tensor(RNG.standard_normal((5, 4)))
    out = layer(x)
    assert out.shape == (5, 3)


def test_linear_gradients():
    layer = Linear(4, 3)
    errors = check_module_gradients(layer, RNG.standard_normal((5, 4)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_conv2d_forward_shape_matches_formula():
    layer = Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
    x = Tensor(RNG.standard_normal((2, 3, 8, 8)))
    out = layer(x)
    # out_h = (8 + 2*1 - 3)//2 + 1 = 4
    assert out.shape == (2, 6, 4, 4)


def test_conv2d_gradients():
    layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
    errors = check_module_gradients(layer, RNG.standard_normal((2, 2, 5, 5)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_conv2d_matches_manual_convolution_single_channel():
    """Cross-check against a directly-computed (non-im2col) 2D
    correlation for one sample/channel/filter, as an independent
    sanity check beyond the gradient-check machinery."""
    layer = Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
    layer.weight.data = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])  # (1,1,2,2)
    x_data = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # (1,1,3,3)
    out = layer(Tensor(x_data))

    expected = np.zeros((2, 2))
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
    for i in range(2):
        for j in range(2):
            expected[i, j] = np.sum(x_data[0, 0, i : i + 2, j : j + 2] * kernel)

    assert np.allclose(out.data[0, 0], expected)


def test_batchnorm2d_normalizes_to_zero_mean_unit_var():
    layer = BatchNorm2d(3)
    x = Tensor(RNG.standard_normal((8, 3, 4, 4)) * 5 + 10)
    out = layer(x)
    per_channel_mean = out.data.mean(axis=(0, 2, 3))
    per_channel_var = out.data.var(axis=(0, 2, 3))
    assert np.allclose(per_channel_mean, 0.0, atol=1e-5)
    assert np.allclose(per_channel_var, 1.0, atol=1e-3)


def test_batchnorm2d_gradients():
    layer = BatchNorm2d(2)
    errors = check_module_gradients(layer, RNG.standard_normal((4, 2, 3, 3)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_batchnorm2d_eval_mode_uses_running_stats():
    layer = BatchNorm2d(2)
    x = Tensor(RNG.standard_normal((8, 2, 3, 3)))
    layer(x)  # a training-mode forward pass to populate running stats
    running_mean_before = layer.running_mean.copy()
    layer.eval()
    layer(Tensor(RNG.standard_normal((1, 2, 3, 3))))
    # eval-mode forward must not update running stats further
    assert np.allclose(layer.running_mean, running_mean_before)


def test_layernorm_normalizes_last_dim():
    layer = LayerNorm(6)
    x = Tensor(RNG.standard_normal((4, 6)) * 3 + 2)
    out = layer(x)
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.data.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_gradients():
    layer = LayerNorm(5)
    errors = check_module_gradients(layer, RNG.standard_normal((3, 5)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_dropout_is_identity_in_eval_mode():
    layer = Dropout(p=0.5).eval()
    x = Tensor(RNG.standard_normal((4, 4)))
    out = layer(x)
    assert np.allclose(out.data, x.data)


def test_dropout_zeroes_expected_fraction_and_scales_survivors():
    np.random.seed(0)
    layer = Dropout(p=0.3)
    x = Tensor(np.ones((1000,)), requires_grad=True)
    out = layer(x)
    zero_fraction = float((out.data == 0).mean())
    assert abs(zero_fraction - 0.3) < 0.05
    survivors = out.data[out.data != 0]
    assert np.allclose(survivors, 1.0 / (1.0 - 0.3))


def test_dropout_backward_uses_same_mask_as_forward():
    np.random.seed(1)
    layer = Dropout(p=0.4)
    x = Tensor(np.ones((500,)), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    # Wherever the forward output was zeroed, the gradient must also be
    # zero; wherever it survived, gradient == the same 1/(1-p) scale.
    zeroed = out.data == 0
    assert np.allclose(x.grad[zeroed], 0.0)
    assert np.allclose(x.grad[~zeroed], 1.0 / (1.0 - 0.4))


def test_embedding_forward_shape_and_lookup_correctness():
    layer = Embedding(vocab_size=10, embed_dim=4)
    indices = np.array([[1, 2, 3], [4, 5, 6]])
    out = layer(indices)
    assert out.shape == (2, 3, 4)
    assert np.allclose(out.data[0, 0], layer.weight.data[1])
    assert np.allclose(out.data[1, 2], layer.weight.data[6])


def test_embedding_gradient_accumulates_for_repeated_tokens():
    layer = Embedding(vocab_size=5, embed_dim=3)
    indices = np.array([1, 1, 2])  # token 1 appears twice
    out = layer(indices)
    out.sum().backward()
    # token 1's row should have received gradient from both positions,
    # i.e. exactly double token 2's row (all-ones upstream gradient).
    assert np.allclose(layer.weight.grad[1], 2.0 * layer.weight.grad[2])
