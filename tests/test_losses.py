"""Tests for MSELoss and CrossEntropyLoss (Phase 2)."""

import numpy as np

from gradus import Tensor
from gradus.nn import CrossEntropyLoss, MSELoss
from gradus.utils import numerical_gradient, relative_error

RNG = np.random.default_rng(0)


def test_mse_forward_value():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = np.array([1.0, 2.0, 5.0])
    loss = MSELoss()(pred, target)
    # ((0)^2 + (0)^2 + (-2)^2) / 3
    assert np.allclose(loss.data, 4.0 / 3.0)


def test_mse_gradient_matches_finite_difference():
    pred_data = RNG.standard_normal(5)
    target = RNG.standard_normal(5)
    pred = Tensor(pred_data.copy(), requires_grad=True)
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    loss.backward()

    def f(p):
        return float(loss_fn(Tensor(p), target).data)

    num_grad = numerical_gradient(f, pred_data.copy())
    assert relative_error(pred.grad, num_grad) < 1e-4


def test_cross_entropy_matches_manual_softmax_nll():
    logits_data = RNG.standard_normal((4, 3))
    targets = np.array([0, 1, 2, 1])
    logits = Tensor(logits_data)
    loss = CrossEntropyLoss()(logits, targets)

    # Manual reference computation.
    shifted = logits_data - logits_data.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    expected = -log_probs[np.arange(4), targets].mean()

    assert np.allclose(loss.data, expected)


def test_cross_entropy_gradient_matches_finite_difference():
    logits_data = RNG.standard_normal((4, 3))
    targets = np.array([0, 1, 2, 1])
    logits = Tensor(logits_data.copy(), requires_grad=True)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    loss.backward()

    def f(x):
        return float(loss_fn(Tensor(x), targets).data)

    num_grad = numerical_gradient(f, logits_data.copy())
    assert relative_error(logits.grad, num_grad) < 1e-4


def test_cross_entropy_decreases_as_logits_favor_correct_class():
    targets = np.array([0])
    loss_fn = CrossEntropyLoss()
    confident_wrong = Tensor(np.array([[0.0, 5.0, 0.0]]))
    confident_right = Tensor(np.array([[5.0, 0.0, 0.0]]))
    assert loss_fn(confident_right, targets).data < loss_fn(confident_wrong, targets).data
