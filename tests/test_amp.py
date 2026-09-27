"""
Tests for gradus.amp.GradScaler (Phase 7) and the optimizers' fp32
master-weight support it depends on (see optim/optimizers.py's module
docstring for why both pieces exist -- gradient underflow vs. update
underflow are two distinct failure modes, and each fix only addresses
one of them).
"""

import numpy as np
import pytest

from gradus.amp import GradScaler
from gradus.nn.module import Parameter
from gradus.optim import SGD, Adam


def test_scale_multiplies_loss_value():
    from gradus import Tensor

    scaler = GradScaler(init_scale=4.0)
    loss = Tensor(np.array(2.0), requires_grad=True)
    scaled = scaler.scale(loss)
    assert scaled.data == pytest.approx(8.0)


def test_step_unscales_and_applies_for_finite_gradients():
    p = Parameter(np.array([1.0, 2.0]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=16.0)

    # Gradient as if it came from backward() on a loss scaled by 16.
    p.grad = np.array([16.0, -16.0])
    stepped = scaler.step(opt)

    assert stepped is True
    # Unscaled gradient is [1.0, -1.0]; SGD with lr=0.1 -> param -= 0.1*grad
    assert np.allclose(p.data, [0.9, 2.1])


def test_step_skips_and_reports_false_on_inf_gradient():
    p = Parameter(np.array([1.0, 2.0]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler(init_scale=16.0)

    before = p.data.copy()
    p.grad = np.array([np.inf, -16.0])
    stepped = scaler.step(opt)

    assert stepped is False
    assert np.array_equal(p.data, before)  # no update applied


def test_step_skips_and_reports_false_on_nan_gradient():
    p = Parameter(np.array([1.0]))
    opt = SGD([p], lr=0.1)
    scaler = GradScaler()

    before = p.data.copy()
    p.grad = np.array([np.nan])
    stepped = scaler.step(opt)

    assert stepped is False
    assert np.array_equal(p.data, before)


def test_update_backs_off_scale_after_inf_and_resets_growth_tracker():
    scaler = GradScaler(init_scale=100.0, backoff_factor=0.5, growth_interval=3)
    p = Parameter(np.array([1.0]))
    opt = SGD([p])

    p.grad = np.array([np.inf])
    scaler.step(opt)
    scaler.update()

    assert scaler.get_scale() == pytest.approx(50.0)
    assert scaler._growth_tracker == 0


def test_update_grows_scale_after_growth_interval_finite_steps():
    scaler = GradScaler(init_scale=8.0, growth_factor=2.0, growth_interval=2)
    p = Parameter(np.array([1.0]))
    opt = SGD([p], lr=0.0)  # lr=0 so repeated steps don't matter numerically

    for _ in range(2):
        p.grad = np.array([1.0])
        scaler.step(opt)
        scaler.update()

    assert scaler.get_scale() == pytest.approx(16.0)


def test_growth_and_backoff_factor_validation():
    with pytest.raises(ValueError):
        GradScaler(growth_factor=1.0)
    with pytest.raises(ValueError):
        GradScaler(backoff_factor=1.0)
    with pytest.raises(ValueError):
        GradScaler(backoff_factor=0.0)


# ----------------------------------------------------------------------
# Master-weight tests (optim/optimizers.py) -- the "update underflow"
# fix GradScaler alone does not provide.
# ----------------------------------------------------------------------

def test_fp32_master_weights_survive_updates_too_small_for_fp16():
    # A tiny, repeated update: below float16's precision at this
    # magnitude, so a naive in-place fp16 update would never move the
    # parameter at all (see BENCHMARK_REPORT.md's Section 5.5
    # demonstration of exactly this). With the fp32 master shadow, progress
    # accumulates invisibly until it crosses an fp16 rounding
    # boundary, at which point the visible fp16 value DOES move.
    p = Parameter(np.array([1000.0], dtype=np.float16))
    opt = Adam([p], lr=1e-3)
    assert opt._master[0] is not None
    assert opt._master[0].dtype == np.float32

    for _ in range(2000):
        p.grad = np.array([0.001], dtype=np.float16)
        opt.step()

    assert p.data[0] < 1000.0  # eventually visible in fp16
    assert opt._master[0][0] < 1000.0
    # The master shadow tracks progress at finer granularity than the
    # fp16 copy it was cast down from.
    assert opt._master[0][0] != np.float32(p.data[0])


def test_fp32_params_get_no_master_weight_shadow():
    p = Parameter(np.array([1.0, 2.0]))  # default dtype: float64
    opt = Adam([p])
    assert opt._master[0] is None
    opt_sgd = SGD([p])
    assert opt_sgd._master[0] is None


def test_sgd_and_adam_fp64_behavior_unchanged_by_master_weight_support():
    # Regression guard: Phase 7's optimizer refactor must not change a
    # single float value for ordinary (non-fp16) parameters.
    p = Parameter(np.array([0.0]))
    opt = SGD([p], lr=1.0, momentum=0.9)
    p.grad = np.array([1.0])
    opt.step()
    first = p.data.copy()
    p.grad = np.array([1.0])
    opt.step()
    assert np.allclose(first - p.data, 1.9)
