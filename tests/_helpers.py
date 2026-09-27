"""
Shared test utilities (not collected by pytest -- no test_ prefix).
"""

import numpy as np

from gradus import Tensor
from gradus.utils import absolute_error, numerical_gradient, relative_error


def gradient_ok(rel: float, abs_: float) -> bool:
    """Same pass criterion as GradCheckResult.passed (see
    gradus.utils.grad_check): relative error is the primary check, but
    falls back to absolute error for parameters whose true gradient is
    legitimately ~0 (e.g. a mean-subtracting layer's effect on its own
    input-sum, or attention's key-bias under softmax's shift
    invariance) -- see that module's docstring for the full
    explanation. Kept as a function (not just a threshold on the dict)
    so test assertions read the same way check_gradient's tests do.
    """
    return rel < 1e-4 or abs_ < 1e-6


def check_module_gradients(module, x_data: np.ndarray, eps: float = 1e-5) -> dict:
    """Finite-difference check of a module's gradient w.r.t. its input
    AND every one of its parameters, using the module's own forward()
    -- i.e. exactly the composition of ops it actually runs, not a
    reimplementation of it.

    Reduces to a scalar via a fixed random projection rather than a
    plain .sum() -- see gradus.utils.grad_check.check_gradient's
    docstring for why a raw sum is unsafe for mean-subtracting layers
    (LayerNorm/BatchNorm2d): their output sums to ~0 for any input by
    construction, which turns the *relative* error metric into noise.

    Returns {name: (relative_error, absolute_error)} -- check each
    entry with gradient_ok(rel, abs_), not a bare threshold on rel.
    """
    x = Tensor(x_data.copy(), requires_grad=True)
    module.zero_grad()
    out = module(x)
    weights = np.random.default_rng(42).standard_normal(out.shape)
    (out * Tensor(weights)).sum().backward()

    errors = {}

    def f_x(perturbed):
        return float((module(Tensor(perturbed)).data * weights).sum())

    num_grad_x = numerical_gradient(f_x, x_data.copy(), eps=eps)
    errors["x"] = (relative_error(x.grad, num_grad_x), absolute_error(x.grad, num_grad_x))

    for name, p in module.named_parameters():
        original = p.data.copy()

        def f_p(perturbed, p=p, shape=original.shape):
            p.data = perturbed.reshape(shape)
            return float((module(Tensor(x_data.copy())).data * weights).sum())

        num_grad = numerical_gradient(f_p, original.copy(), eps=eps)
        p.data = original
        errors[name] = (relative_error(p.grad, num_grad), absolute_error(p.grad, num_grad))

    return errors
