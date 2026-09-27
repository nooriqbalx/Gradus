"""
Tests for Phase 3: multi-head attention and the Transformer block.

Gradient correctness is checked the same way as Conv2d/BatchNorm2d in
test_layers.py (finite-difference on the whole module). The causal
masking test is a separate, architecture-level correctness check:
independent of whether gradients are right, it verifies attention
actually can't see the future, which is what makes this a valid
autoregressive language model rather than just an expensive MLP.
"""

import numpy as np
from _helpers import check_module_gradients, gradient_ok

from gradus import Tensor
from gradus.nn import MultiHeadAttention, TransformerBlock, TransformerLM, causal_mask

RNG = np.random.default_rng(0)


def test_multihead_attention_forward_shape():
    attn = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = Tensor(RNG.standard_normal((2, 5, 8)))
    out = attn(x)
    assert out.shape == (2, 5, 8)


def test_multihead_attention_rejects_indivisible_heads():
    import pytest

    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_multihead_attention_gradients_no_mask():
    attn = MultiHeadAttention(embed_dim=6, num_heads=2)
    errors = check_module_gradients(attn, RNG.standard_normal((2, 4, 6)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_multihead_attention_gradients_with_causal_mask():
    attn = MultiHeadAttention(embed_dim=6, num_heads=3)
    mask = causal_mask(4)

    class _Wrapped:
        """Bind the mask so check_module_gradients' generic
        module(Tensor) call signature still works."""

        def __init__(self, attn, mask):
            self.attn = attn
            self.mask = mask

        def __call__(self, x):
            return self.attn(x, mask=self.mask)

        def zero_grad(self):
            self.attn.zero_grad()

        def named_parameters(self):
            return self.attn.named_parameters()

    errors = check_module_gradients(_Wrapped(attn, mask), RNG.standard_normal((2, 4, 6)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_causal_mask_blocks_future_positions():
    """Changing a future token must not change an earlier position's
    attention output -- the defining property of causal (autoregressive)
    attention, checked independently of gradient correctness."""
    attn = MultiHeadAttention(embed_dim=8, num_heads=2)
    seq_len = 5
    mask = causal_mask(seq_len)

    x_data = RNG.standard_normal((1, seq_len, 8))
    out_a = attn(Tensor(x_data), mask=mask)

    x_data_modified = x_data.copy()
    x_data_modified[0, -1, :] += 100.0  # drastically change only the LAST token
    out_b = attn(Tensor(x_data_modified), mask=mask)

    # Every position except the last must be unaffected by the change
    # to the last (future, from their perspective) token.
    assert np.allclose(out_a.data[0, :-1], out_b.data[0, :-1], atol=1e-6)
    # The last position (which CAN see itself) should differ.
    assert not np.allclose(out_a.data[0, -1], out_b.data[0, -1], atol=1e-6)


def test_causal_mask_shape_and_pattern():
    mask = causal_mask(3)
    assert mask.shape == (1, 1, 3, 3)
    # Row i should allow columns 0..i (value 0) and block columns > i (large negative).
    expected_allowed = np.array([[True, False, False], [True, True, False], [True, True, True]])
    assert np.array_equal(mask[0, 0] == 0, expected_allowed)


def test_transformer_block_forward_shape():
    block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16)
    x = Tensor(RNG.standard_normal((2, 5, 8)))
    out = block(x)
    assert out.shape == (2, 5, 8)


def test_transformer_block_gradients():
    block = TransformerBlock(embed_dim=6, num_heads=2, ff_dim=12)
    errors = check_module_gradients(block, RNG.standard_normal((2, 3, 6)))
    assert all(gradient_ok(rel, abs_) for rel, abs_ in errors.values()), errors


def test_transformer_lm_forward_shape():
    model = TransformerLM(
        vocab_size=20, embed_dim=8, num_heads=2, ff_dim=16, num_layers=2, max_len=10
    )
    token_ids = RNG.integers(0, 20, size=(3, 7))
    logits = model(token_ids)
    assert logits.shape == (3, 7, 20)


def test_transformer_lm_rejects_sequence_longer_than_max_len():
    import pytest

    model = TransformerLM(
        vocab_size=10, embed_dim=8, num_heads=2, ff_dim=16, num_layers=1, max_len=4
    )
    with pytest.raises(ValueError):
        model(RNG.integers(0, 10, size=(1, 5)))


def test_transformer_lm_backward_populates_all_parameter_gradients():
    """Integration check: a full forward+backward through embedding,
    every Transformer block, and the LM head reaches every parameter
    -- i.e. nothing in the stack silently breaks the graph."""
    model = TransformerLM(
        vocab_size=15, embed_dim=8, num_heads=2, ff_dim=16, num_layers=2, max_len=6
    )
    token_ids = RNG.integers(0, 15, size=(2, 5))
    logits = model(token_ids)
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
