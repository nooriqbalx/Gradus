"""
Multi-head self-attention and sinusoidal positional encoding
(Phase 3).

Like the normalization layers in nn/layers.py, attention needed no new
gradient primitive in gradus.ops: scaled dot-product attention is
matmul + div (by a constant scale) + a constant mask add + softmax
(itself a composition, see nn/functional.py) + matmul -- every piece
already gradient-checked. What's new here is only the *shape*
bookkeeping (splitting embed_dim into heads and back), which is
exercised by test_attention.py's own finite-difference check on the
whole module, the same way test_layers.py checks Conv2d/BatchNorm2d.
"""

from __future__ import annotations

import numpy as np

from gradus.nn.functional import softmax
from gradus.nn.layers import Linear
from gradus.nn.module import Module
from gradus.tensor import Tensor


def sinusoidal_positional_encoding(max_len: int, embed_dim: int) -> np.ndarray:
    """The original Transformer (Vaswani et al., 2017) sinusoidal
    encoding: a fixed (not learned) constant added to token embeddings
    so the model can distinguish position without a separate learned
    positional embedding table."""
    position = np.arange(max_len)[:, None]
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    pe = np.zeros((max_len, embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def causal_mask(seq_len: int) -> np.ndarray:
    """(1, 1, seq_len, seq_len) additive mask: 0 where position i may
    attend to position j (j <= i), a large negative constant where it
    may not (j > i) -- added to the raw attention scores before
    softmax, so a masked position's post-softmax probability is
    exp(mask_value) / (...), which only needs to be indistinguishable
    from 0.0 in whatever dtype the scores happen to be in.

    Phase 7 note -- this was -1e9 through Phase 1-6 ("large enough to
    be safe, why not"), which is itself a real bug once fp16 models
    exist: fp16's max representable magnitude is ~65504, so casting
    -1e9 down to fp16 (which every masked add does automatically once
    Phase 7's dtype-propagation fix routes the mask through the
    scores' own dtype, see ops.py's _resolve_dtype) silently overflows
    to -inf, raising a NumPy RuntimeWarning on every single masked
    attention call under fp16. The result was still numerically
    correct here (exp(-inf) is a well-defined, exact 0.0, not nan --
    this project's own fp16 GPU parity test confirmed no nan/inf
    reaches the output), but relying on overflow-to-infinity behavior
    "working out" is fragile, not a deliberate design choice, and the
    warning noise alone is worth avoiding.

    -1e4 is negative enough to underflow exp() to exactly 0.0 in
    float16, float32, AND float64 alike (exp(-1e4) is already
    astronomically below any float format's smallest representable
    positive value well before -1e4 -- exp(-50) alone is ~2e-22), so
    there was never a numerical reason for -1e9 in the first place;
    it's simply the smallest magnitude that comfortably clears every
    dtype this project supports, with no dtype-specific branching
    needed here.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e4
    return mask.reshape(1, 1, seq_len, seq_len)


class MultiHeadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, t: Tensor, batch: int, seq: int) -> Tensor:
        # (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
        t = t.reshape(batch, seq, self.num_heads, self.head_dim)
        return t.transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, mask: np.ndarray = None) -> Tensor:
        batch, seq, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch, seq)
        k = self._split_heads(self.k_proj(x), batch, seq)
        v = self._split_heads(self.v_proj(x), batch, seq)

        scores = (q @ k.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = softmax(scores, axis=-1)
        out = attn @ v  # (batch, num_heads, seq, head_dim)

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.embed_dim)
        return self.out_proj(out)

    def __repr__(self) -> str:
        return f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
