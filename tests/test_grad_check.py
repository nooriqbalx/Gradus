"""
Finite-difference gradient checks for every Phase 1 op -- the
numerical-correctness leg of the project's evaluation, built now so
each op is verified as it's written (see gradus.utils.grad_check).

Inputs are randomized but seeded for reproducibility, and shifted
away from zero where an op is undefined or unstable there (div, log,
pow with a non-integer-safe base) so the check reflects the op's
gradient rule rather than a numerical edge case.
"""

import numpy as np

from gradus.utils import check_gradient

RNG = np.random.default_rng(0)


def test_grad_check_add():
    inputs = [RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))]
    result = check_gradient("add", lambda a, b: a + b, inputs)
    assert result.passed, result


def test_grad_check_add_broadcast():
    inputs = [RNG.standard_normal((3, 4)), RNG.standard_normal((4,))]
    result = check_gradient("add_broadcast", lambda a, b: a + b, inputs)
    assert result.passed, result


def test_grad_check_sub():
    inputs = [RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))]
    result = check_gradient("sub", lambda a, b: a - b, inputs)
    assert result.passed, result


def test_grad_check_mul():
    inputs = [RNG.standard_normal((3, 4)), RNG.standard_normal((3, 4))]
    result = check_gradient("mul", lambda a, b: a * b, inputs)
    assert result.passed, result


def test_grad_check_div():
    # Keep the denominator away from zero.
    b = RNG.uniform(0.5, 2.0, size=(3, 4))
    result = check_gradient("div", lambda a, b: a / b, [RNG.standard_normal((3, 4)), b])
    assert result.passed, result


def test_grad_check_neg():
    result = check_gradient("neg", lambda a: -a, [RNG.standard_normal((3, 4))])
    assert result.passed, result


def test_grad_check_pow():
    # Keep the base positive so fractional/negative exponents stay
    # well-defined (not exercised here, but keeps this check general).
    a = RNG.uniform(0.5, 2.0, size=(3, 4))
    result = check_gradient("pow", lambda a: a ** 3, [a])
    assert result.passed, result


def test_grad_check_exp():
    result = check_gradient("exp", lambda a: a.exp(), [RNG.standard_normal((3, 4)) * 0.5])
    assert result.passed, result


def test_grad_check_log():
    a = RNG.uniform(0.5, 2.0, size=(3, 4))
    result = check_gradient("log", lambda a: a.log(), [a])
    assert result.passed, result


def test_grad_check_relu():
    # Perturb away from the kink at 0 so finite differences don't
    # straddle the discontinuity in the derivative.
    a = RNG.choice([-1, 1], size=(3, 4)) * RNG.uniform(0.5, 2.0, size=(3, 4))
    result = check_gradient("relu", lambda a: a.relu(), [a])
    assert result.passed, result


def test_grad_check_matmul():
    result = check_gradient(
        "matmul", lambda a, b: a @ b, [RNG.standard_normal((3, 4)), RNG.standard_normal((4, 5))]
    )
    assert result.passed, result


def test_grad_check_sum():
    result = check_gradient("sum", lambda a: a.sum(axis=1), [RNG.standard_normal((3, 4))])
    assert result.passed, result


def test_grad_check_mean():
    result = check_gradient("mean", lambda a: a.mean(), [RNG.standard_normal((3, 4))])
    assert result.passed, result


def test_grad_check_reshape():
    result = check_gradient("reshape", lambda a: a.reshape(4, 3), [RNG.standard_normal((3, 4))])
    assert result.passed, result


def test_grad_check_transpose():
    result = check_gradient("transpose", lambda a: a.transpose(), [RNG.standard_normal((3, 4))])
    assert result.passed, result


def test_grad_check_tanh():
    result = check_gradient("tanh", lambda a: a.tanh(), [RNG.standard_normal((3, 4)) * 0.5])
    assert result.passed, result


def test_grad_check_unfold():
    from gradus.ops import unfold

    # Small (N=2, C=3, H=5, W=5) input, 3x3 kernel, stride 1, no padding.
    a = RNG.standard_normal((2, 3, 5, 5))
    result = check_gradient("unfold", lambda a: unfold(a, (3, 3), stride=1, padding=0), [a])
    assert result.passed, result


def test_grad_check_unfold_strided_padded():
    from gradus.ops import unfold

    a = RNG.standard_normal((2, 2, 6, 6))
    result = check_gradient(
        "unfold_strided_padded", lambda a: unfold(a, (3, 3), stride=2, padding=1), [a]
    )
    assert result.passed, result


def test_grad_check_embedding_lookup():
    from gradus.ops import embedding_lookup

    vocab_size, embed_dim = 6, 4
    indices = RNG.integers(0, vocab_size, size=(3, 5))
    # embedding_lookup takes indices as a fixed (non-differentiable)
    # argument, so it's checked as a single-input op over the weight
    # matrix, with indices captured in the closure.
    weight = RNG.standard_normal((vocab_size, embed_dim))
    result = check_gradient(
        "embedding_lookup", lambda w: embedding_lookup(w, indices), [weight]
    )
    assert result.passed, result


def test_grad_check_embedding_lookup_repeated_indices():
    """Repeated indices must accumulate gradient (scatter-add), not
    overwrite -- this is the case that would silently break with a
    naive (non-accumulating) backward implementation."""
    from gradus.ops import embedding_lookup

    vocab_size, embed_dim = 4, 3
    indices = np.array([1, 1, 2, 1])  # index 1 repeated three times
    weight = RNG.standard_normal((vocab_size, embed_dim))
    result = check_gradient(
        "embedding_lookup_repeated", lambda w: embedding_lookup(w, indices), [weight]
    )
    assert result.passed, result
