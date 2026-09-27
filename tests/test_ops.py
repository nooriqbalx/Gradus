"""
Forward-value and hand-derived backward-value tests for each op
(Phase 1). These check specific, easy-to-verify-by-hand cases;
broad correctness across random inputs is covered by the
finite-difference gradient checks in test_grad_check.py.
"""

import numpy as np

from gradus import Tensor


def test_add_forward_and_backward():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    out = (a + b).sum()
    out.backward()
    assert np.allclose(out.data, 10.0)
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])


def test_add_broadcasting_backward_shapes():
    a = Tensor(np.ones((2, 3)), requires_grad=True)
    b = Tensor(np.ones((3,)), requires_grad=True)
    out = (a + b).sum()
    out.backward()
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, [2.0, 2.0, 2.0])  # summed over broadcast dim


def test_mul_backward():
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    out = (a * b).sum()
    out.backward()
    assert np.allclose(a.grad, [4.0, 5.0])  # d(ab)/da = b
    assert np.allclose(b.grad, [2.0, 3.0])  # d(ab)/db = a


def test_sub_and_neg():
    a = Tensor(np.array([5.0]), requires_grad=True)
    b = Tensor(np.array([2.0]), requires_grad=True)
    out = (a - b).sum()
    out.backward()
    assert np.allclose(out.data, 3.0)
    assert np.allclose(a.grad, [1.0])
    assert np.allclose(b.grad, [-1.0])


def test_div_backward():
    a = Tensor(np.array([6.0]), requires_grad=True)
    b = Tensor(np.array([2.0]), requires_grad=True)
    out = (a / b).sum()
    out.backward()
    assert np.allclose(out.data, 3.0)
    assert np.allclose(a.grad, [0.5])  # 1/b
    assert np.allclose(b.grad, [-1.5])  # -a/b^2


def test_pow_backward():
    a = Tensor(np.array([3.0]), requires_grad=True)
    out = (a ** 2).sum()
    out.backward()
    assert np.allclose(out.data, 9.0)
    assert np.allclose(a.grad, [6.0])  # 2a


def test_exp_and_log_are_inverse_in_forward():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    roundtrip = a.exp().log()
    roundtrip.sum().backward()
    assert np.allclose(roundtrip.data, [1.0, 2.0])
    assert np.allclose(a.grad, [1.0, 1.0])


def test_relu_backward_masks_negatives():
    a = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
    activated = a.relu()
    activated.sum().backward()
    assert np.allclose(activated.data, [0.0, 0.0, 2.0])
    assert np.allclose(a.grad, [0.0, 0.0, 1.0])


def test_matmul_forward_and_backward_shapes():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)
    out = (a @ b).sum()
    out.backward()
    assert (a @ b).data.shape == (3, 5)
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4, 5)


def test_sum_with_axis():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    out = a.sum(axis=0).sum()
    out.backward()
    assert np.allclose(a.grad, np.ones((2, 2)))


def test_mean():
    a = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)
    out = a.mean()
    out.backward()
    assert np.allclose(out.data, 4.0)
    assert np.allclose(a.grad, [1 / 3, 1 / 3, 1 / 3])


def test_reshape_backward():
    a = Tensor(np.arange(6.0), requires_grad=True)
    out = a.reshape(2, 3).sum()
    out.backward()
    assert np.allclose(a.grad, np.ones(6))


def test_transpose_backward():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    out = a.transpose().sum()
    out.backward()
    assert a.grad.shape == (2, 3)
    assert np.allclose(a.grad, np.ones((2, 3)))


def test_T_property():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(a.T.data, [[1.0, 3.0], [2.0, 4.0]])
