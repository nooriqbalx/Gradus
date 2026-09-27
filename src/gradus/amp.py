"""
Automatic mixed precision support (Phase 7 -- Stretch).

Running a model in float16 shrinks memory and (on real GPU tensor
cores) speeds up compute, but it introduces two distinct numerical
failure modes that a from-scratch framework has to actually confront,
not just wave at:

    1. Gradient underflow. Many real gradients are small (1e-5 to
       1e-8) -- comfortably representable in float32, but float16's
       ~2^-24 smallest positive normal magnitude means a lot of them
       flush to exactly zero the moment backward() computes them,
       silently stalling learning for whichever parameters happened to
       have small gradients that step (which, empirically, is "a lot
       of them" -- see BENCHMARK_REPORT.md's Section 5.6 convergence
       study, where plain fp16 with no loss scaling visibly stalls
       next to fp32).
       Fix: GradScaler below -- multiply the loss by a large constant
       before backward() (so every gradient in the chain is scaled up
       by the same constant, shifting the whole gradient distribution
       into float16's representable range), then divide back out
       before the optimizer step. This is the standard technique (Micikevicius
       et al., 2017, "Mixed Precision Training"; the same design
       PyTorch's torch.cuda.amp.GradScaler implements).
    2. Update underflow. Even with correctly-scaled gradients, Adam's
       computed update (lr * m_hat / (sqrt(v_hat) + eps)) is often far
       smaller than the parameter's own magnitude -- applying it
       in-place to an already-float16 parameter can round away to
       nothing (`p.data -= update` where update is smaller than
       float16's precision at p.data's magnitude is a silent no-op).
       This is a *different* failure from (1) and loss scaling does
       nothing for it. Fix: gradus.optim's Adam/SGD auto-detect any
       float16 parameter and keep an internal float32 "master weight"
       shadow copy (see optim/optimizers.py) that accumulates every
       update at full precision, only rounding down to float16 once,
       right before the next forward pass reads it.

Both fixes are necessary; BENCHMARK_REPORT.md's Section 5.6 convergence
study demonstrates each failure mode in isolation (fp16 with no
GradScaler; fp16 with GradScaler but no master weights) alongside the
version with both fixes applied, matching fp32 convergence.
"""

from __future__ import annotations

from gradus.backend import get_array_module
from gradus.tensor import Tensor


class GradScaler:
    """Dynamic loss scaling, mirroring the shape of PyTorch's
    torch.cuda.amp.GradScaler (scale -> step -> update) so the API is
    recognizable, not because this project depends on PyTorch (it
    doesn't -- see pyproject.toml's `compare` extra, which nothing in
    gradus itself imports).

    Typical training step:

        scaler = GradScaler()
        ...
        loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()      # scale up before backward()
        stepped = scaler.step(optimizer)   # unscale, skip step if inf/nan
        scaler.update()                    # grow/backoff the scale factor
        optimizer.zero_grad()

    `scale` starts large (2**16 by default -- large enough to lift most
    underflowing fp16 gradients into range) and grows further every
    `growth_interval` consecutive finite steps (there was headroom, so
    try scaling more), or immediately backs off by `backoff_factor`
    the moment a step produces an inf/nan gradient (scaled too far,
    overflowed instead of underflowing -- back off and skip that step
    rather than corrupting the parameters with a step computed from
    garbage gradients).
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        if not (0.0 < backoff_factor < 1.0):
            raise ValueError("backoff_factor must be in (0, 1)")
        self._scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False  # set by step(), consumed by update()

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss: Tensor) -> Tensor:
        """Multiply the loss by the current scale factor before
        backward(). Every gradient computed by the resulting
        backward() pass is scaled by the same constant (scaling a
        graph's root by k scales every gradient in it by k, by
        linearity of the chain rule), which is what actually protects
        small gradients from flushing to zero in float16 -- not the
        loss value itself, which is thrown away."""
        return loss * self._scale

    def _any_non_finite(self, optimizer) -> bool:
        for p in optimizer.parameters:
            if p.grad is None:
                continue
            xp = get_array_module(p.grad)
            if not bool(xp.all(xp.isfinite(p.grad))):
                return True
        return False

    def step(self, optimizer) -> bool:
        """Unscale every parameter's gradient (divide out the constant
        `scale()` multiplied in), then either call optimizer.step() (if
        every gradient is finite) or skip it entirely (if scaling
        pushed something to inf/nan -- taking a step from a corrupted
        gradient would be worse than just discarding this step and
        backing off the scale for next time, see update()).

        Returns True if the step was actually taken, False if it was
        skipped due to inf/nan gradients.
        """
        self._found_inf = self._any_non_finite(optimizer)
        if not self._found_inf:
            for p in optimizer.parameters:
                if p.grad is None:
                    continue
                xp = get_array_module(p.grad)
                if xp.dtype(p.grad.dtype).itemsize < 4:
                    # Upcast BEFORE dividing, not after: p.grad is
                    # still float16 here (this project's dtype
                    # propagation keeps backward()'s gradients in the
                    # same dtype as the parameters they came from), and
                    # the whole point of scaling the loss up before
                    # backward() was to keep small gradients away from
                    # float16's underflow floor. Dividing back down
                    # while still stored as float16 would walk them
                    # right back into that same underflow floor,
                    # silently undoing everything GradScaler was for.
                    # Casting up to float32 first means the division
                    # (and everything downstream in the optimizer,
                    # which keeps its own float32 master-weight state
                    # for exactly this reason) happens at full
                    # precision.
                    p.grad = xp.asarray(p.grad, dtype=xp.float32) / self._scale
                else:
                    p.grad = p.grad / self._scale
            optimizer.step()
            return True
        return False

    def update(self) -> None:
        """Adjust the scale factor based on the outcome of the most
        recent step(): grow it every `growth_interval` consecutive
        finite steps (try to use more of fp16's range), or immediately
        shrink it by `backoff_factor` the moment a step overflowed.
        Must be called once per training step, after step()."""
        if self._found_inf:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0
