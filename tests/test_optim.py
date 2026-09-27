"""Tests for SGD and Adam (Phase 2)."""

import numpy as np

from gradus.nn.module import Parameter
from gradus.optim import SGD, Adam


def test_sgd_moves_parameter_opposite_gradient():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([1.0, -1.0])
    opt = SGD([p], lr=0.1)
    opt.step()
    assert np.allclose(p.data, [0.9, 2.1])


def test_sgd_momentum_accumulates_across_steps():
    p = Parameter(np.array([0.0]))
    opt = SGD([p], lr=1.0, momentum=0.9)

    p.grad = np.array([1.0])
    opt.step()  # v = 1.0, param -= 1.0 -> -1.0
    first = p.data.copy()

    p.grad = np.array([1.0])
    opt.step()  # v = 0.9*1.0 + 1.0 = 1.9, param -= 1.9
    second_step_delta = first - p.data
    assert np.allclose(second_step_delta, 1.9)


def test_sgd_skips_parameters_with_no_gradient():
    p = Parameter(np.array([5.0]))  # grad left as None
    opt = SGD([p], lr=0.1)
    opt.step()  # must not raise, must not change p
    assert np.allclose(p.data, [5.0])


def test_adam_step_reduces_loss_direction():
    # Not a full convergence test (that's test_trainer.py) -- just
    # confirms one Adam step moves the parameter opposite its gradient,
    # as any sane optimizer must for a well-conditioned single step.
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    opt = Adam([p], lr=0.1)
    opt.step()
    assert p.data[0] < 1.0


def test_adam_bias_correction_matches_reference_first_step():
    p = Parameter(np.array([0.0]))
    grad = 2.0
    p.grad = np.array([grad])
    lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
    opt = Adam([p], lr=lr, betas=(beta1, beta2), eps=eps)
    opt.step()

    m_hat = ((1 - beta1) * grad) / (1 - beta1)
    v_hat = ((1 - beta2) * grad**2) / (1 - beta2)
    expected = 0.0 - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(p.data, [expected])


def test_optimizer_rejects_empty_parameter_list():
    import pytest

    with pytest.raises(ValueError):
        SGD([])
