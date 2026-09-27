"""
Stateless tensor functions used by layers and losses -- the Gradus
analogue of torch.nn.functional.

Everything here is a composition of ops already defined and gradient-
checked in gradus.ops (exp, log, sum, div, sub, mul, ...). None of it
needs its own backward rule: correctness follows automatically from
the chain rule applied to already-verified primitives. That's the
payoff of getting Phase 1's autodiff engine right -- these are "free"
once addition, multiplication, exp, and division are proven correct.
"""

from __future__ import annotations

import numpy as np

from gradus.backend import get_array_module
from gradus.tensor import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax: subtract the (constant, no-grad)
    per-row max before exponentiating. Subtracting a constant from
    every logit doesn't change softmax's value or its gradient -- it
    only prevents exp() from overflowing -- so this is an exact
    reformulation, not an approximation."""
    shift = Tensor(x.data.max(axis=axis, keepdims=True))  # constant: requires_grad=False
    shifted = x - shift
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """log(softmax(x)), computed directly via the log-sum-exp identity
    rather than log(softmax(x)) literally, for better numerical
    stability (avoids taking log of a very small softmax output)."""
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    shifted = x - shift
    log_sum_exp = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - log_sum_exp


def one_hot(indices, num_classes: int, dtype=float) -> np.ndarray:
    """Non-differentiable target encoding: integer class labels ->
    a constant 0/1 array, on whichever backend `indices` is already on
    (get_array_module, not a hardcoded NumPy call -- CrossEntropyLoss
    passes GPU-resident indices through unchanged when its logits are
    on GPU, see losses.py). This is target data, not a graph node, so
    it deliberately returns a plain array rather than a Tensor -- wrap
    it in Tensor(...) (requires_grad=False by default) wherever it's
    used.

    `dtype` defaults to float64 (the historical behavior) but
    CrossEntropyLoss (Phase 7) passes the logits' own dtype instead --
    encoding an fp16 model's targets as float64 would upcast the whole
    `target_one_hot * log_probs` product (and everything downstream of
    it) back to float64 the moment the loss is computed, silently
    undoing the point of running the model in fp16 at all.
    """
    xp = get_array_module(indices)
    indices = xp.asarray(indices)
    out = xp.zeros(indices.shape + (num_classes,), dtype=dtype)
    try:
        xp.put_along_axis(out, indices[..., None], 1.0, axis=-1)
    except AttributeError:
        # Some CuPy releases don't implement put_along_axis; this
        # fancy-index scatter is an exact equivalent for the last-axis
        # case used here (verified correct on real CUDA hardware via
        # tests/test_gpu_parity.py).
        grid = xp.indices(indices.shape)
        out[tuple(grid) + (indices,)] = 1.0
    return out
