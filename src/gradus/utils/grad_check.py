"""
Gradient checking: verifying that each op's analytical backward rule
agrees with a finite-difference approximation of the same function.

This is the numerical-correctness leg of the project's evaluation (see
BENCHMARK_REPORT.md) -- but the harness itself is built early, so
every op added from here on can be checked as it's written rather
than trusted on faith until the end.

Method (standard central-difference check, e.g. CS231n's convention):
    for each input array x and each element x[i]:
        numerical_grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
    then compare against the analytical gradient produced by
    Tensor.backward(), via a relative error:
        |analytical - numerical| / max(eps_floor, |analytical| + |numerical|)

A relative error below ~1e-4 (float64) is the conventional threshold
for "the analytical gradient is correct"; larger ops (e.g. attention,
LayerNorm, once added in Phase 3) may need a looser threshold or a
smaller eps due to accumulated floating-point error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from gradus.tensor import Tensor


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Central-difference numerical gradient of scalar-valued `f`
    with respect to array `x`, evaluated element-by-element.

    `f` is called with the *same* array object each time (perturbed
    and restored in place), so it must not itself mutate `x`.
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f(x)
        x[idx] = orig - eps
        minus = f(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(analytical: np.ndarray, numerical: np.ndarray, eps_floor: float = 1e-8) -> float:
    """Max elementwise relative error between two gradient arrays."""
    analytical = np.asarray(analytical, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    numerator = np.abs(analytical - numerical)
    denominator = np.maximum(eps_floor, np.abs(analytical) + np.abs(numerical))
    return float(np.max(numerator / denominator))


def absolute_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Max elementwise absolute error. Needed alongside relative_error
    because relative error is meaningless when both gradients are
    legitimately ~0 (see GradCheckResult.passed's docstring)."""
    analytical = np.asarray(analytical, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    return float(np.max(np.abs(analytical - numerical)))


@dataclass
class GradCheckResult:
    op: str
    max_relative_error: float
    per_input_errors: list = field(default_factory=list)
    max_absolute_error: float = 0.0
    per_input_absolute_errors: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Pass if EITHER the relative error is small OR the absolute
        error is small.

        Relative error alone is not a sufficient test: when a
        parameter's true gradient is exactly (or numerically
        indistinguishable from) zero -- which happens for real
        mathematical reasons, not just "small" inputs; e.g. a
        multi-head attention key-projection bias, which softmax's
        per-row shift-invariance makes provably unable to affect the
        output at all -- both the analytical and numerical gradient
        are ~1e-12-1e-17, and their *relative* difference can still
        read as e.g. 5e-4 purely from floating-point noise around
        zero. This is the standard caveat in gradient-check literature
        (e.g. CS231n's notes: "for gradients near zero, use absolute
        error instead"); checking both criteria is how a real zero
        gradient is told apart from an actually wrong one.
        """
        return self.max_relative_error < 1e-4 or self.max_absolute_error < 1e-6


def check_gradient(
    op_name: str,
    fn: Callable[..., Tensor],
    inputs: Sequence[np.ndarray],
    eps: float = 1e-5,
    seed: int = 0,
) -> GradCheckResult:
    """Check `fn`'s analytical gradient (via Gradus autodiff) against
    a finite-difference approximation, for every array in `inputs`.

    Parameters
    ----------
    op_name : str
        Label for the report (e.g. "matmul").
    fn : callable
        Takes len(inputs) Tensor positional arguments and returns a
        Tensor of any shape.
    inputs : sequence of np.ndarray
        One array per positional argument to `fn`. Copied internally,
        never mutated in place from the caller's perspective.
    eps : float
        Finite-difference step size.
    seed : int
        Seeds the fixed random projection used to reduce a non-scalar
        output to a scalar (see note below).

    Note on reducing a non-scalar output: a plain `.sum()` is NOT used
    here, because some ops (LayerNorm/BatchNorm's mean-subtraction is
    the concrete case that surfaced this) produce an output that sums
    to ~0 for *any* input by mathematical construction -- both the
    analytical and numerical gradient of `out.sum()` are then
    correctly ~1e-16, but comparing two numbers that small makes the
    *relative* error metric meaningless (it can read as e.g. 0.4% from
    pure floating-point noise, with nothing actually wrong). The
    standard fix (see CS231n's gradient-check notes) is to reduce via
    a fixed random linear projection, `(out * W).sum()` for a random
    constant W of the same shape, instead of `out.sum()` -- this is
    still a valid scalar function of every input element (so gradient
    checking it still exercises every element's backward path), but
    it doesn't have the same-for-any-input degeneracy a raw sum can.
    """
    inputs = [np.asarray(x, dtype=float) for x in inputs]
    tensors = [Tensor(x.copy(), requires_grad=True) for x in inputs]
    rng = np.random.default_rng(seed)

    out = fn(*tensors)
    if out.data.size == 1:
        scalar = out
        weights = None
    else:
        weights = rng.standard_normal(out.data.shape)
        scalar = (out * Tensor(weights)).sum()
    scalar.backward()

    per_input_errors = []
    per_input_absolute_errors = []
    for i, x in enumerate(inputs):
        analytical = tensors[i].grad
        if analytical is None:
            analytical = np.zeros_like(x)

        def scalar_fn(perturbed_x, _idx=i):
            args = [t.copy() for t in inputs]
            args[_idx] = perturbed_x
            out_i = fn(*[Tensor(a) for a in args])
            if weights is None:
                return float(out_i.data)
            return float((out_i.data * weights).sum())

        numerical = numerical_gradient(scalar_fn, x.copy(), eps=eps)
        per_input_errors.append(relative_error(analytical, numerical))
        per_input_absolute_errors.append(absolute_error(analytical, numerical))

    return GradCheckResult(
        op=op_name,
        max_relative_error=max(per_input_errors) if per_input_errors else 0.0,
        per_input_errors=per_input_errors,
        max_absolute_error=max(per_input_absolute_errors) if per_input_absolute_errors else 0.0,
        per_input_absolute_errors=per_input_absolute_errors,
    )


def check_module_gradient(
    module,
    x_data: np.ndarray,
    op_name: str | None = None,
    eps: float = 1e-5,
    seed: int = 0,
) -> GradCheckResult:
    """Finite-difference check of a Module's gradient w.r.t. its input
    AND every one of its parameters, using the module's own forward()
    -- i.e. exactly the composition of ops it actually runs, not a
    reimplementation of it. Built for Phase 5's aggregated report
    (`benchmarks/grad_check.py`): unlike check_gradient (one raw op,
    one result), this checks input + every parameter but folds them
    into a SINGLE GradCheckResult (the max error across all of them),
    so a whole layer sits as one row in the same report table a
    primitive op's check_gradient call produces -- per_input_errors
    still holds the (label, error) breakdown for anyone who wants to
    see which specific parameter was closest to failing.

    Same reduction caveat as check_gradient applies here, for the same
    reason (LayerNorm/BatchNorm2d's output sums to ~0 by construction,
    which breaks a plain .sum() reduction's relative-error metric) --
    see check_gradient's docstring.
    """
    x_data = np.asarray(x_data, dtype=float)
    rng = np.random.default_rng(seed)
    x = Tensor(x_data.copy(), requires_grad=True)
    module.zero_grad()
    out = module(x)
    weights = None if out.data.size == 1 else rng.standard_normal(out.data.shape)
    scalar = out if weights is None else (out * Tensor(weights)).sum()
    scalar.backward()

    def reduced(data):
        return float(data) if weights is None else float((data * weights).sum())

    labels: list = ["x"]
    rel_errors: list = []
    abs_errors: list = []

    analytical_x = x.grad if x.grad is not None else np.zeros_like(x_data)

    def f_x(perturbed):
        return reduced(module(Tensor(perturbed)).data)

    num_x = numerical_gradient(f_x, x_data.copy(), eps=eps)
    rel_errors.append(relative_error(analytical_x, num_x))
    abs_errors.append(absolute_error(analytical_x, num_x))

    for name, p in module.named_parameters():
        original = p.data.copy()
        analytical_p = p.grad if p.grad is not None else np.zeros_like(original)

        def f_p(perturbed, p=p, shape=original.shape):
            p.data = perturbed.reshape(shape)
            return reduced(module(Tensor(x_data.copy())).data)

        num_p = numerical_gradient(f_p, original.copy(), eps=eps)
        labels.append(name)
        rel_errors.append(relative_error(analytical_p, num_p))
        abs_errors.append(absolute_error(analytical_p, num_p))

    return GradCheckResult(
        op=op_name or type(module).__name__,
        max_relative_error=max(rel_errors),
        per_input_errors=list(zip(labels, rel_errors)),
        max_absolute_error=max(abs_errors),
        per_input_absolute_errors=list(zip(labels, abs_errors)),
    )


def format_report(results: Sequence[GradCheckResult]) -> str:
    """Render a list of GradCheckResults as a Markdown table, matching
    the format used in the Phase 5 benchmark report."""
    lines = [
        "| Operation | Max relative error | Max absolute error | Status |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.op} | {r.max_relative_error:.2e} | {r.max_absolute_error:.2e} | {status} |"
        )
    return "\n".join(lines)
