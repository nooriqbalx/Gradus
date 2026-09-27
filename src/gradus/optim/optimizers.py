"""
Optimizers (Phase 2; extended Phase 7 for mixed-precision training).

Both operate directly on Parameter.data / Parameter.grad (NumPy or, as
of Phase 4, CuPy arrays) rather than going through the autograd graph
-- an optimizer step is not itself a differentiable operation, it's
what happens *after* backward() has already populated every
parameter's .grad.

Phase 7 mixed-precision note -- the "update underflow" problem:
    Casting a model to float16 (Module.half()) and fixing gradient
    underflow with gradus.amp.GradScaler is not, by itself, enough for
    training to actually converge. Adam's computed update
    (lr * m_hat / (sqrt(v_hat) + eps)) is frequently far smaller than
    the parameter's own magnitude late in training -- applying it
    in-place to an already-float16 array (`p.data -= update`) rounds
    away to nothing the moment `abs(update) < p.data`'s local float16
    precision, silently freezing that parameter. This is a *different*
    failure mode from gradient underflow and loss scaling does nothing
    for it (PHASE7_REPORT.md's convergence study shows this happening
    even with GradScaler enabled).

    Fix: both optimizers below auto-detect any float16 parameter at
    construction time and keep an internal float32 "master weight"
    shadow copy for it, alongside float32 moment/velocity buffers (an
    Adam moment estimate is itself a running accumulation of many
    small increments -- exactly what float16 accumulates worst). Every
    step updates the float32 shadow at full precision and only casts
    down to the parameter's own float16 storage once, right before the
    next forward pass reads it -- so precision is lost once per step,
    at the unavoidable point (float16 IS what the forward/backward
    compute actually runs on), not accumulated destructively across
    hundreds of steps. This is the same design NVIDIA's mixed-precision
    training recipe (Micikevicius et al., 2017) and PyTorch's apex/amp
    master-weight support use.

    Parameters that are already float32/float64 take neither of these
    extra buffers (the per-parameter `_master` entry is None for them,
    and the moment/velocity buffer dtype matches p.data exactly, as in
    Phase 1-6) -- this is a strict, opt-in addition with zero behavior
    change for every non-mixed-precision model.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from gradus.backend import get_array_module
from gradus.nn.module import Parameter


def _is_reduced_precision(dtype) -> bool:
    """True for float16 -- the one dtype in this project that needs an
    fp32 master-weight shadow (see this module's docstring)."""
    d = np.dtype(dtype)
    return d.kind == "f" and d.itemsize < 4


def _accumulator_dtype(dtype):
    """float32 for a reduced-precision parameter, otherwise `dtype`
    itself unchanged -- so a plain float32/float64 model's moment
    buffers are byte-for-byte the same as Phase 1-6 (zeros_like(p.data)
    in every existing test), and only float16 parameters get the extra
    precision."""
    return np.float32 if _is_reduced_precision(dtype) else dtype


class Optimizer:
    """Base class for SGD/Adam.

    GPU note: SGD/Adam allocate their own per-parameter state buffers
    (velocity, or the first/second moment estimates) at construction
    time, on whichever backend each parameter is on AT THAT MOMENT. So
    call model.to("cuda") BEFORE constructing the optimizer, not after
    -- e.g. `model.to("cuda"); opt = Adam(model.parameters())`, not the
    reverse -- otherwise these buffers stay on the old device while
    p.data/p.grad move to the new one, and step() crashes on the first
    mixed-backend arithmetic. This is the same ordering PyTorch has
    (its own docs: call .to(device) before constructing the
    optimizer). The same ordering applies to Module.half() (Phase 7):
    call it before constructing the optimizer, since the float32
    master-weight shadow below is also snapshotted at construction
    time.
    """

    def __init__(self, parameters: Iterable[Parameter], use_master_weights: bool = True):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimizer received an empty parameter list")
        # Phase 7: one fp32 master-weight shadow per reduced-precision
        # parameter, None for everything else -- see module docstring.
        # use_master_weights=False exists ONLY to reproduce the
        # "update underflow" failure mode on purpose, for
        # PHASE7_REPORT.md's ablation (GradScaler alone, no master
        # weights, still stalls) -- leave it True for real training.
        self._master = []
        for p in self.parameters:
            if use_master_weights and _is_reduced_precision(p.data.dtype):
                xp = get_array_module(p.data)
                self._master.append(xp.asarray(p.data, dtype=xp.float32))
            else:
                self._master.append(None)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def _apply_update(self, index: int, p: Parameter, update) -> None:
        """Apply `update` (already lr-scaled, same sign convention as
        `p.data -= update`) to parameter `p`, through its fp32 master
        shadow if it has one."""
        master = self._master[index]
        if master is not None:
            master -= update.astype(master.dtype, copy=False)
            p.data[...] = master.astype(p.data.dtype)
        else:
            p.data -= update


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum:
        v = momentum * v + grad
        param -= lr * v
    (the standard "heavy ball" formulation; momentum=0 reduces to
    vanilla SGD).
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        use_master_weights: bool = True,
    ):
        super().__init__(parameters, use_master_weights=use_master_weights)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [
            get_array_module(p.data).zeros(p.data.shape, dtype=_accumulator_dtype(p.data.dtype))
            for p in self.parameters
        ]

    def step(self) -> None:
        for i, (p, v) in enumerate(zip(self.parameters, self._velocity)):
            if p.grad is None:
                continue
            v *= self.momentum
            v += p.grad
            self._apply_update(i, p, self.lr * v)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014): per-parameter adaptive learning rates
    from running estimates of the gradient's first and second moments,
    with bias correction for the early steps when those estimates are
    still close to their zero initialization.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        use_master_weights: bool = True,
    ):
        super().__init__(parameters, use_master_weights=use_master_weights)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self._m = [
            get_array_module(p.data).zeros(p.data.shape, dtype=_accumulator_dtype(p.data.dtype))
            for p in self.parameters
        ]
        self._v = [
            get_array_module(p.data).zeros(p.data.shape, dtype=_accumulator_dtype(p.data.dtype))
            for p in self.parameters
        ]

    def step(self) -> None:
        self.t += 1
        for i, (p, m, v) in enumerate(zip(self.parameters, self._m, self._v)):
            if p.grad is None:
                continue
            xp = get_array_module(p.data)
            # Cast grad up to the accumulator's dtype BEFORE squaring,
            # not after: for a float16 parameter (accumulator dtype
            # float32, see _accumulator_dtype), p.grad**2 computed
            # while still float16 can itself underflow/lose precision
            # before the += upcasts the *result* -- squaring first in
            # float32 avoids that extra rounding step.
            grad = p.grad if p.grad.dtype == m.dtype else xp.asarray(p.grad, dtype=m.dtype)
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            self._apply_update(i, p, self.lr * m_hat / (xp.sqrt(v_hat) + self.eps))
