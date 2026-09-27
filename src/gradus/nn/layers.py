"""
Standard neural network layers (Phase 2).

Linear, Dropout, and Embedding are close to minimal wrappers around a
single op. Conv2d, BatchNorm2d, and LayerNorm are more interesting:
each is built entirely by composing already-verified ops/functional
pieces (unfold+matmul for Conv2d; mean/pow/div for the normalization
layers), so none of them needed a new gradient-checked primitive --
only the composition itself needed testing, which lives in
tests/test_layers.py.
"""

from __future__ import annotations

import numpy as np

from gradus.backend import get_array_module
from gradus.nn.module import Module, Parameter
from gradus.ops import embedding_lookup, unfold
from gradus.tensor import Tensor


class Linear(Module):
    """y = x @ W^T + b, matching PyTorch's Linear convention (weight
    stored as (out_features, in_features) so the same weight matrix
    can be dropped in from a PyTorch model for comparison if needed).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # Kaiming/He-uniform-style init: keeps activation variance
        # roughly stable through a stack of layers with ReLU-family
        # nonlinearities between them.
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class Conv2d(Module):
    """2D convolution via im2col + matmul (see gradus.ops.unfold).

    Weight layout matches PyTorch: (out_channels, in_channels, kh, kw).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        bound = 1.0 / np.sqrt(in_channels * kh * kw)
        self.weight = Parameter(
            np.random.uniform(-bound, bound, (out_channels, in_channels, kh, kw))
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        out_h = (h + 2 * self.padding - kh) // self.stride + 1
        out_w = (w + 2 * self.padding - kw) // self.stride + 1

        cols = unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
        # cols: (N, C*kh*kw, out_h*out_w)
        weight_mat = self.weight.reshape(self.out_channels, -1)  # (out_c, C*kh*kw)
        # (1, out_c, K) @ (N, K, L) broadcasts over the batch dim, just
        # like NumPy's own @ operator -- our matmul op's backward
        # already handles this via unbroadcast(), so no special case
        # is needed here.
        out = weight_mat.reshape(1, self.out_channels, -1) @ cols  # (N, out_c, out_h*out_w)
        out = out.reshape(n, self.out_channels, out_h, out_w)
        if self.bias is not None:
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


class BatchNorm2d(Module):
    """Normalizes each channel over (N, H, W), matching PyTorch's
    BatchNorm2d. Tracks a running mean/variance (plain NumPy, not part
    of the autograd graph) for use at inference time via .eval().
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Registered (not plain attributes) so Module.to(device) moves
        # them along with the learnable gamma/beta -- these aren't
        # Parameters (no gradient flows through a running average), so
        # parameters()/to()'s Parameter branch would otherwise miss them.
        self.register_buffer("running_mean", np.zeros(num_features))
        self.register_buffer("running_var", np.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W); normalize per channel across N, H, W.
        if self.training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            # In-place-style update on whichever backend running_mean/
            # var currently live on -- correct only if the module was
            # already moved to x's device (see Module.to()'s ordering
            # note); mean.data/var.data live on x's backend by
            # construction, so this silently produces a wrong-backend
            # crash rather than a wrong number if that ordering is
            # violated, which is the safer failure mode.
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean.data.reshape(-1)
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var.data.reshape(-1)
            )
        else:
            mean = Tensor(self.running_mean.reshape(1, -1, 1, 1), device=x.device)
            var = Tensor(self.running_var.reshape(1, -1, 1, 1), device=x.device)

        x_norm = (x - mean) / (var + self.eps) ** 0.5
        gamma = self.gamma.reshape(1, self.num_features, 1, 1)
        beta = self.beta.reshape(1, self.num_features, 1, 1)
        return x_norm * gamma + beta

    def __repr__(self) -> str:
        return f"BatchNorm2d({self.num_features})"


class LayerNorm(Module):
    """Normalizes over the last dimension, matching PyTorch's
    LayerNorm(normalized_shape) for the common case of a single
    trailing dimension (e.g. the embedding dimension in a Transformer).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones(normalized_shape))
        self.beta = Parameter(np.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        return x_norm * self.gamma + self.beta

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape})"


class Dropout(Module):
    """Inverted dropout: zeroes elements with probability p and scales
    survivors by 1/(1-p) during training, so no rescaling is needed at
    inference time. A no-op when self.training is False (see
    Module.eval()) or when p == 0.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        xp = get_array_module(x.data)
        mask = (xp.random.rand(*x.shape) > self.p).astype(float) / (1.0 - self.p)
        return x * Tensor(mask, device=x.device)  # mask is a constant: requires_grad=False

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Embedding(Module):
    """Token embedding lookup table: (vocab_size, embed_dim), indexed
    via gradus.ops.embedding_lookup (see that op's docstring for why
    this needed its own primitive rather than being a composition)."""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.weight = Parameter(np.random.randn(vocab_size, embed_dim) * 0.01)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, indices: np.ndarray) -> Tensor:
        return embedding_lookup(self.weight, indices)

    def __repr__(self) -> str:
        return f"Embedding({self.vocab_size}, {self.embed_dim})"
