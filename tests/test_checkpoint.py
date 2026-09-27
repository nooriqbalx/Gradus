"""
Tests for gradient checkpointing (Phase 7): gradus.utils.checkpoint and
the no_grad() context manager it's built on.

The central claim being tested is that checkpointing is numerically
*transparent* -- forward output and every gradient must exactly match
the non-checkpointed version, since it's a pure memory/compute
trade-off, not an approximation.
"""

import numpy as np

from gradus._grad_mode import is_grad_enabled, no_grad
from gradus.nn.transformer import TransformerBlock, TransformerLM
from gradus.tensor import Tensor
from gradus.utils.checkpoint import checkpoint


def test_no_grad_disables_graph_construction():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    assert is_grad_enabled() is True
    with no_grad():
        assert is_grad_enabled() is False
        y = x * 2
        assert y.requires_grad is False
        assert y._prev == ()
    assert is_grad_enabled() is True
    z = x * 2
    assert z.requires_grad is True


def test_no_grad_nesting_restores_correctly():
    with no_grad():
        assert is_grad_enabled() is False
        with no_grad():
            assert is_grad_enabled() is False
        assert is_grad_enabled() is False  # inner exit didn't turn it back on early
    assert is_grad_enabled() is True


def test_checkpoint_matches_direct_call_forward_and_grad():
    def fn(x):
        return (x * 2 + 1).relu().sum(axis=-1, keepdims=True)

    x_data = np.random.default_rng(0).standard_normal((3, 4))

    x1 = Tensor(x_data.copy(), requires_grad=True)
    out1 = fn(x1)
    out1.backward(np.ones_like(out1.data))

    x2 = Tensor(x_data.copy(), requires_grad=True)
    out2 = checkpoint(fn, x2)
    out2.backward(np.ones_like(out2.data))

    assert np.allclose(out1.data, out2.data)
    assert np.allclose(x1.grad, x2.grad)


def test_checkpoint_does_not_retain_graph_during_forward():
    def fn(x):
        return x * 2

    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    out = checkpoint(fn, x)
    # The checkpoint node itself IS wired into the graph (so backward()
    # from further downstream can reach it and trigger recompute)...
    assert out.requires_grad is True
    assert out._prev == (x,)
    # ...but nothing INSIDE fn was tracked during that forward call --
    # there is no way to observe this from the checkpoint node itself,
    # so this is really exercised implicitly by test_no_grad_* above
    # (checkpoint's forward pass runs fn under the same no_grad()).


def test_checkpoint_accumulates_gradient_when_input_used_twice():
    def fn(x):
        return x * 3

    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    out_a = checkpoint(fn, x)
    out_b = checkpoint(fn, x)
    total = out_a + out_b
    total.backward(np.ones_like(total.data))
    # d(3x)/dx + d(3x)/dx = 6, accumulated across both checkpoint calls
    assert np.allclose(x.grad, [6.0, 6.0])


def test_checkpoint_populates_parameter_grads_even_when_input_does_not_require_grad():
    # Regression test for a real bug this project's own GPU dry run
    # caught: checkpoint() used to decide whether to build a graph
    # node purely from whether its *external* Tensor inputs required
    # grad, with no way to know that `fn` also closes over Parameters
    # that need gradients. A network's very first layer is fed a plain
    # (non-Parameter) input that itself never requires grad, so
    # checkpointing it used to silently produce NO gradient at all for
    # that layer's own weights -- see checkpoint.py's docstring on
    # `requires_grad = is_grad_enabled()` for the fix.
    from gradus.nn.layers import Linear

    lin = Linear(3, 2)

    def fn(x):
        return lin(x)

    x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=False)
    out = checkpoint(fn, x)
    out.backward(np.ones_like(out.data))

    assert lin.weight.grad is not None
    assert lin.bias.grad is not None
    assert not np.allclose(lin.weight.grad, 0.0)


def test_checkpoint_with_non_tensor_kwarg():
    def fn(x, scale=1.0):
        return x * scale

    x = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    out = checkpoint(fn, x, scale=3.0)
    out.backward(np.ones_like(out.data))
    assert np.allclose(out.data, [6.0, 12.0])
    assert np.allclose(x.grad, [3.0, 3.0])


def test_transformer_block_checkpoint_matches_non_checkpoint():
    np.random.seed(0)
    block_ckpt = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16, use_checkpoint=True)
    np.random.seed(0)
    block_plain = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16, use_checkpoint=False)

    x_data = np.random.default_rng(1).standard_normal((2, 5, 8))
    x1 = Tensor(x_data.copy(), requires_grad=True)
    x2 = Tensor(x_data.copy(), requires_grad=True)

    out1 = block_ckpt(x1)
    out2 = block_plain(x2)
    assert np.allclose(out1.data, out2.data)

    grad_seed = np.random.default_rng(2).standard_normal(out1.data.shape)
    out1.backward(grad_seed.copy())
    out2.backward(grad_seed.copy())
    assert np.allclose(x1.grad, x2.grad)

    for (n1, p1), (n2, p2) in zip(block_ckpt.named_parameters(), block_plain.named_parameters()):
        assert n1 == n2
        assert np.allclose(p1.grad, p2.grad), f"gradient mismatch for {n1}"


def test_transformer_lm_end_to_end_with_checkpoint():
    # Smoke test: a full TransformerLM forward/backward with
    # use_checkpoint=True runs without error and produces gradients
    # for every parameter, including the embedding table.
    np.random.seed(0)
    model = TransformerLM(
        vocab_size=10, embed_dim=8, num_heads=2, ff_dim=16, num_layers=2, max_len=6,
        use_checkpoint=True,
    )
    token_ids = np.random.randint(0, 10, size=(2, 6))
    logits = model(token_ids)
    logits.backward(np.ones_like(logits.data))
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing gradient for {name}"
