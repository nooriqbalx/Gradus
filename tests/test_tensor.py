"""
Tests for the core Tensor class and its graph mechanics (Phase 1).
Op-specific forward/backward correctness lives in test_ops.py; broad
gradient verification across all ops lives in test_grad_check.py.
"""

import numpy as np
import pytest

from gradus import Tensor
from gradus.backend import cupy_available


def test_tensor_construction():
    t = Tensor(np.array([1.0, 2.0, 3.0]))
    assert t.data.shape == (3,)
    assert t.requires_grad is False
    assert t.device == "cpu"
    assert t.grad is None


def test_tensor_coerces_non_ndarray_input():
    t = Tensor([1.0, 2.0, 3.0])
    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == float


def test_tensor_repr_includes_shape_and_device():
    t = Tensor(np.zeros((2, 3)), device="cpu")
    assert "shape=(2, 3)" in repr(t)
    assert "device='cpu'" in repr(t)


def test_backward_raises_if_requires_grad_false():
    t = Tensor(np.array([2.0]), requires_grad=False)
    with pytest.raises(RuntimeError):
        t.backward()


def test_backward_needs_explicit_grad_for_non_scalar():
    t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(ValueError):
        t.backward()


def test_backward_defaults_to_one_for_scalar():
    t = Tensor(np.array(3.0), requires_grad=True)
    t.backward()
    assert t.grad == np.array(1.0)


def test_backward_accumulates_across_calls():
    t = Tensor(np.array(2.0), requires_grad=True)
    t.backward()
    t.backward()
    assert t.grad == np.array(2.0)  # 1.0 + 1.0


def test_zero_grad():
    t = Tensor(np.array([1.0]), requires_grad=True)
    t.grad = np.array([5.0])
    t.zero_grad()
    assert t.grad is None


def test_to_cpu_is_a_noop_roundtrip():
    """`.to("cpu")` on an already-CPU tensor is the well-defined case
    this sandbox (no GPU) can actually exercise; CPU/GPU parity itself
    is covered by tests/test_gpu_parity.py, which skips here and has
    been run and verified for real on a Kaggle GPU notebook."""
    t = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    moved = t.to("cpu")
    assert isinstance(moved.data, np.ndarray)
    np.testing.assert_array_equal(moved.data, t.data)
    # .to() is documented as a graph-detach boundary: the moved copy is
    # a fresh leaf, not wired into t's graph.
    assert moved._prev == ()
    assert moved._backward is None


@pytest.mark.skipif(
    cupy_available(),
    reason="This test asserts the error path taken when CuPy is NOT "
    "installed. On a real GPU machine (e.g. Kaggle) CuPy is installed and "
    "gradus.backend.cupy_available() is (correctly) True, so .to('cuda') "
    "succeeds instead of raising -- that is the expected, correct behavior "
    "there, not a failure of this test's assertion. See "
    "test_gpu_parity.py's pytestmark for the inverse skip.",
)
def test_to_cuda_without_cupy_raises_helpful_error():
    """On this CPU-only sandbox (no CuPy installed), requesting
    device="cuda" must fail loudly with actionable instructions rather
    than silently falling back to CPU -- a silent fallback would let a
    real Kaggle GPU bug masquerade as CPU-only success here."""
    t = Tensor(np.array([1.0]))
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        t.to("cuda")


def test_convenience_constructors():
    z = Tensor.zeros(2, 3)
    o = Tensor.ones(2, 3)
    r = Tensor.randn(2, 3)
    assert z.data.shape == (2, 3) and np.all(z.data == 0)
    assert o.data.shape == (2, 3) and np.all(o.data == 1)
    assert r.data.shape == (2, 3)


def test_graph_builds_only_when_requires_grad():
    a = Tensor(np.array([1.0]), requires_grad=False)
    b = Tensor(np.array([2.0]), requires_grad=False)
    out = a + b
    # Neither input requires grad, so nothing to backprop through --
    # the output should not retain a graph.
    assert out.requires_grad is False
    assert out._prev == ()
    assert out._backward is None
