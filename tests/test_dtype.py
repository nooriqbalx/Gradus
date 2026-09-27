"""
Tests for Phase 7's dtype propagation fix and Tensor/Module casting.

The core bug being guarded against: before Phase 7, ops._as_tensor
unconditionally forced dtype=float (float64) on every raw (non-Tensor)
operand, so any op mixing an fp16 Tensor with a raw Python/NumPy
constant silently upcast the whole computation back to float64 --
defeating the entire point of casting a model to fp16. See ops.py's
_as_tensor/_resolve_dtype docstrings for the full explanation.
"""

import numpy as np

from gradus import Tensor
from gradus.nn.layers import BatchNorm2d, Linear
from gradus.nn.losses import CrossEntropyLoss, MSELoss
from gradus.nn.module import Sequential


def test_add_raw_constant_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16))
    y = x + 1.0
    assert y.data.dtype == np.float16


def test_sub_raw_constant_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16))
    y = x - 0.5
    assert y.data.dtype == np.float16


def test_mul_raw_constant_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16))
    y = x * 2.0
    assert y.data.dtype == np.float16


def test_div_raw_constant_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16) + 2.0)
    y = x / 2.0
    assert y.data.dtype == np.float16


def test_matmul_raw_array_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16))
    raw = np.random.randn(4, 2).astype(np.float32)
    y = x @ raw
    # dtype follows the Tensor operand (x, fp16), not the raw fp32
    # array -- _resolve_dtype checks Tensor operands first.
    assert y.data.dtype == np.float16


def test_mean_preserves_fp16_dtype():
    x = Tensor(np.random.randn(3, 4).astype(np.float16))
    assert x.mean().data.dtype == np.float16


def test_plain_python_scalars_still_default_to_float64():
    # No Tensor operand at all -- _resolve_dtype has nothing to
    # inherit from, so this falls back to the historical float64
    # default, unchanged from Phase 1-6.
    from gradus.ops import add

    out = add(1.0, 2.0)
    assert out.data.dtype == np.float64


def test_tensor_half_float_double_cast_and_detach():
    x = Tensor(np.random.randn(2, 2), requires_grad=True)
    y = x * 2  # give x a non-trivial graph node to detach from
    h = y.half()
    assert h.data.dtype == np.float16
    assert h.requires_grad is True
    assert h._prev == ()  # detached, like .to()
    assert h._backward is None

    f = h.float()
    assert f.data.dtype == np.float32
    d = f.double()
    assert d.data.dtype == np.float64


def test_tensor_zeros_ones_accept_explicit_dtype():
    z = Tensor.zeros(2, 3, dtype=np.float16)
    o = Tensor.ones(2, 3, dtype=np.float32)
    assert z.data.dtype == np.float16
    assert o.data.dtype == np.float32
    # Default unchanged from Phase 1-6.
    assert Tensor.zeros(2).data.dtype == np.float64


def test_module_half_casts_parameters_and_buffers():
    model = Sequential(Linear(4, 8), Linear(8, 2))
    for p in model.parameters():
        assert p.data.dtype == np.float64
    model.half()
    for p in model.parameters():
        assert p.data.dtype == np.float16


def test_module_half_casts_registered_buffers():
    bn = BatchNorm2d(4)
    assert bn.running_mean.dtype == np.float64
    bn.half()
    assert bn.running_mean.dtype == np.float16
    assert bn.running_var.dtype == np.float16
    for p in bn.parameters():
        assert p.data.dtype == np.float16


def test_module_float_and_double_roundtrip():
    lin = Linear(4, 4)
    lin.half()
    lin.float()
    for p in lin.parameters():
        assert p.data.dtype == np.float32
    lin.double()
    for p in lin.parameters():
        assert p.data.dtype == np.float64


def test_mseloss_target_dtype_follows_pred_not_hardcoded_float64():
    pred = Tensor(np.random.randn(4, 1).astype(np.float32), requires_grad=True)
    loss_fn = MSELoss()
    loss = loss_fn(pred, np.random.randn(4, 1))  # raw target, not a Tensor
    assert loss.data.dtype == np.float32
    loss.backward()
    assert pred.grad.dtype == np.float32


def test_cross_entropy_target_dtype_follows_logits_not_hardcoded_float64():
    logits = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    targets = np.array([0, 1, 2, 1, 0])
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    assert loss.data.dtype == np.float32
    loss.backward()
    assert logits.grad.dtype == np.float32
