"""
Transformer block and a small decoder-only language model (Phase 3).

Pre-norm residual design (LayerNorm applied *before* attention/FFN,
not after) -- the modern convention (GPT-2 onward), chosen because it
trains more stably without a learning-rate warmup schedule, which
matters for a small from-scratch optimizer implementation like ours.
"""

from __future__ import annotations

import numpy as np

from gradus.nn.activations import GELU
from gradus.nn.attention import MultiHeadAttention, causal_mask, sinusoidal_positional_encoding
from gradus.nn.layers import Embedding, LayerNorm, Linear
from gradus.nn.module import Module
from gradus.tensor import Tensor
from gradus.utils.checkpoint import checkpoint


class TransformerBlock(Module):
    """Pre-norm attention + FFN sublayer, each wrapped in a residual.

    `use_checkpoint` (Phase 7): when True, each sublayer's activations
    are recomputed during backward() instead of being kept around
    between the forward and backward passes -- see
    gradus.utils.checkpoint's module docstring for the mechanism and
    benchmarks/checkpoint_memory.py + PHASE7_REPORT.md for the measured
    memory/compute trade-off. Purely a memory/compute trade: forward
    output and every gradient are numerically identical to
    use_checkpoint=False (tests/test_checkpoint.py verifies this
    directly), so it's safe to toggle without retraining or re-tuning
    anything.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, use_checkpoint: bool = False):
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = LayerNorm(embed_dim)
        self.ff1 = Linear(embed_dim, ff_dim)
        self.act = GELU()
        self.ff2 = Linear(ff_dim, embed_dim)
        self.use_checkpoint = use_checkpoint

    def _attn_sublayer(self, x: Tensor, mask: np.ndarray = None) -> Tensor:
        return self.attn(self.norm1(x), mask=mask)

    def _ff_sublayer(self, x: Tensor) -> Tensor:
        return self.ff2(self.act(self.ff1(self.norm2(x))))

    def forward(self, x: Tensor, mask: np.ndarray = None) -> Tensor:
        if self.use_checkpoint:
            x = x + checkpoint(self._attn_sublayer, x, mask=mask)
            x = x + checkpoint(self._ff_sublayer, x)
        else:
            x = x + self._attn_sublayer(x, mask=mask)
            x = x + self._ff_sublayer(x)
        return x

    def __repr__(self) -> str:
        return f"TransformerBlock(attn={self.attn!r}, use_checkpoint={self.use_checkpoint})"


class TransformerLM(Module):
    """A small decoder-only (GPT-style) language model: token
    embedding + sinusoidal positions -> a stack of causal
    TransformerBlocks -> final LayerNorm -> a Linear head projecting
    to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_len: int,
        use_checkpoint: bool = False,
    ):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = sinusoidal_positional_encoding(max_len, embed_dim)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ]
        self.norm_final = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

    def forward(self, token_ids: np.ndarray) -> Tensor:
        """token_ids: integer array (batch, seq), seq <= max_len."""
        _, seq = token_ids.shape
        if seq > self.max_len:
            raise ValueError(f"sequence length {seq} exceeds max_len {self.max_len}")

        x = self.embedding(token_ids)  # (batch, seq, embed_dim)
        # Passed as a raw ndarray, not pre-wrapped in Tensor(...): add()'s
        # own _as_tensor converts a raw array onto the OTHER operand's
        # device automatically (see ops.py), so this follows x onto GPU
        # for free. Pre-wrapping it here would freeze it on CPU instead,
        # since a Tensor's device is never auto-converted, only a raw
        # array's is -- the same reason causal_mask() below is also
        # passed unwrapped.
        x = x + self.pos_encoding[:seq]  # broadcasts over batch
        mask = causal_mask(seq)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm_final(x)
        return self.lm_head(x)  # (batch, seq, vocab_size)

    def __repr__(self) -> str:
        return (
            f"TransformerLM(vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
            f"num_layers={len(self.blocks)})"
        )
