"""
Full gradient-check report (Phase 5).

Runs every Phase 1 primitive op and every Phase 2-3 composed layer
through finite-difference gradient checking and renders the result as
one Markdown table -- the "per-op relative error table" ROADMAP.md's
Phase 5 calls for. This is a reporting script, not a test: the
individual checks it runs already exist as pytest assertions across
tests/test_ops.py, test_grad_check.py, test_layers.py, and
test_attention.py (which is what actually gates CI-style correctness);
this script just re-runs the same methodology in one place and
produces the aggregated table for the write-up.

Two things worth knowing before reading the table:

  1. Every "layer" row is deliberately an AGGREGATE: it checks the
     gradient w.r.t. the input AND every one of the layer's own
     parameters, then reports the single WORST (max) relative/absolute
     error across all of them (see check_module_gradient's docstring).
     A layer with 5 parameters passing this row means all 6 gradients
     (x + 5 params) individually passed -- not just the easiest one.
  2. The pass criterion is relative error < 1e-4 OR absolute error <
     1e-6, not relative error alone -- see GradCheckResult.passed's
     docstring for why (some gradients are legitimately ~0 for real
     mathematical reasons, e.g. attention's key-projection bias under
     softmax's shift-invariance, and relative error is meaningless
     when comparing two numbers that are both correctly near zero).

Usage:
    python benchmarks/grad_check.py
    python benchmarks/grad_check.py --out benchmarks/results/grad_check_report.md
"""

from __future__ import annotations

import argparse

import numpy as np

from gradus.nn import GELU, BatchNorm2d, Conv2d, LayerNorm, Linear, MultiHeadAttention
from gradus.nn.functional import softmax
from gradus.nn.losses import CrossEntropyLoss, MSELoss
from gradus.nn.transformer import TransformerBlock
from gradus.ops import (
    add,
    div,
    embedding_lookup,
    exp,
    log,
    matmul,
    mean,
    mul,
    neg,
    pow_,
    relu,
    sub,
    sum_,
    tanh,
    transpose,
    unfold,
)
from gradus.utils import GradCheckResult, check_gradient, check_module_gradient, format_report

rng = np.random.default_rng(42)


def primitive_checks() -> list[GradCheckResult]:
    """Phase 1 ops (plus unfold/embedding_lookup, Phase 2's two new
    primitives) -- each checked directly via gradus.ops functions, not
    through a Module, since these have no learnable parameters of
    their own."""
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))
    a_bcast = rng.standard_normal((4, 3))
    b_bcast = rng.standard_normal((3,))  # exercises unbroadcast()
    mat_a = rng.standard_normal((4, 5))
    mat_b = rng.standard_normal((5, 3))
    mat_a_bcast = rng.standard_normal((2, 4, 5))  # batched, broadcasts over batch
    mat_b_bcast = rng.standard_normal((5, 3))
    pos = np.abs(rng.standard_normal((4, 3))) + 0.5  # log needs strictly positive input

    results = [
        check_gradient("add", add, [a, b]),
        check_gradient("add (broadcast)", add, [a_bcast, b_bcast]),
        check_gradient("sub", sub, [a, b]),
        check_gradient("mul", mul, [a, b]),
        check_gradient("div", div, [a, pos]),
        check_gradient("neg", neg, [a]),
        check_gradient("pow (p=3)", lambda x: pow_(x, 3), [a]),
        check_gradient("exp", exp, [a]),
        check_gradient("log", log, [pos]),
        check_gradient("relu", relu, [a]),
        check_gradient("tanh", tanh, [a]),
        check_gradient("matmul", matmul, [mat_a, mat_b]),
        check_gradient("matmul (broadcast batch)", matmul, [mat_a_bcast, mat_b_bcast]),
        check_gradient("sum (all axes)", sum_, [a]),
        check_gradient("sum (axis=0)", lambda x: sum_(x, axis=0), [a]),
        check_gradient("mean", mean, [a]),
        check_gradient("reshape", lambda x: x.reshape(2, 6), [a]),
        check_gradient("transpose", transpose, [a]),
        check_gradient(
            "unfold (Conv2d's im2col)",
            lambda x: unfold(x, (3, 3), stride=1, padding=1),
            [rng.standard_normal((2, 3, 8, 8))],
        ),
        check_gradient(
            "embedding_lookup (incl. repeated index)",
            lambda w: embedding_lookup(w, np.array([[3, 3, 1, 0], [2, 3, 3, 1]])),
            [rng.standard_normal((10, 6))],
        ),
    ]
    return results


def layer_checks() -> list[GradCheckResult]:
    """Phase 2-3 composed layers -- each checked via check_module_gradient,
    which aggregates the input's gradient AND every parameter's
    gradient into one worst-case row (see this script's docstring)."""
    results = [
        check_module_gradient(Linear(6, 4), rng.standard_normal((5, 6)), op_name="Linear"),
        check_module_gradient(
            Conv2d(2, 4, 3, padding=1), rng.standard_normal((3, 2, 8, 8)), op_name="Conv2d"
        ),
        check_module_gradient(
            BatchNorm2d(4), rng.standard_normal((4, 4, 5, 5)), op_name="BatchNorm2d (training)"
        ),
        check_module_gradient(LayerNorm(12), rng.standard_normal((5, 12)), op_name="LayerNorm"),
        check_module_gradient(GELU(), rng.standard_normal((5, 8)), op_name="GELU"),
        check_module_gradient(
            MultiHeadAttention(embed_dim=16, num_heads=4),
            rng.standard_normal((2, 6, 16)),
            op_name="MultiHeadAttention (no mask)",
        ),
        check_module_gradient(
            TransformerBlock(embed_dim=16, num_heads=4, ff_dim=32),
            rng.standard_normal((2, 6, 16)),
            op_name="TransformerBlock (attn+FFN+2xLayerNorm)",
        ),
    ]

    # softmax (a Phase 2 functional composition, not a Module) --
    # checked directly via check_gradient like a primitive, since it
    # has no parameters.
    softmax_input = rng.standard_normal((5, 6))
    results.append(check_gradient("softmax", lambda x: softmax(x, axis=-1), [softmax_input]))

    # Note: Embedding (the Module) is intentionally not re-checked here
    # -- embedding_lookup already covers its backward rule in
    # primitive_checks, and the Module is a two-line forward()
    # convenience wrapper around it with no new math of its own.

    return results


def loss_checks() -> list[GradCheckResult]:
    """Losses: gradient checked w.r.t. their differentiable input only
    (predictions/logits) -- targets are integer labels or fixed data,
    not differentiable, so they're closed over as constants rather
    than passed through check_gradient's numerical-differentiation."""
    pred = rng.standard_normal((5, 3))
    target = rng.standard_normal((5, 3))
    logits = rng.standard_normal((6, 4))
    labels = rng.integers(0, 4, size=(6,))

    mse = MSELoss()
    ce = CrossEntropyLoss()

    return [
        check_gradient("MSELoss", lambda p: mse(p, target), [pred]),
        check_gradient("CrossEntropyLoss", lambda logits_: ce(logits_, labels), [logits]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Optional path to also write the report to.")
    args = parser.parse_args()

    sections = [
        ("Phase 1 primitives (+ unfold/embedding_lookup)", primitive_checks()),
        ("Phase 2-3 composed layers", layer_checks()),
        ("Losses", loss_checks()),
    ]

    all_results: list[GradCheckResult] = []
    report_lines = ["# Gradient-check report", ""]
    for title, results in sections:
        all_results.extend(results)
        report_lines.append(f"## {title}")
        report_lines.append("")
        report_lines.append(format_report(results))
        report_lines.append("")

    n_pass = sum(r.passed for r in all_results)
    n_total = len(all_results)
    summary = f"**{n_pass}/{n_total} checks passed.**"
    report_lines.insert(2, summary)
    report_lines.insert(3, "")

    report = "\n".join(report_lines)
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved -> {args.out}")

    if n_pass != n_total:
        raise SystemExit(f"{n_total - n_pass} gradient check(s) failed -- see table above.")


if __name__ == "__main__":
    main()
