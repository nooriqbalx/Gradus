"""
Activation functions as Module wrappers, so they can sit inside a
Sequential alongside layers that do have parameters.

GELU uses the standard tanh-based approximation (Hendrycks & Gimpel,
2016) rather than the exact erf-based formula, since it's expressible
purely from ops already in gradus.ops (tanh, pow, mul, add) without
needing an erf primitive -- accurate to within ~1e-3 of the exact
GELU, which is the same approximation PyTorch itself offers via
`nn.GELU(approximate="tanh")`.
"""

from __future__ import annotations

import math

from gradus.nn.module import Module
from gradus.tensor import Tensor

_GELU_COEFF = math.sqrt(2.0 / math.pi)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        inner = _GELU_COEFF * (x + 0.044715 * (x ** 3))
        return 0.5 * x * (1.0 + inner.tanh())

    def __repr__(self) -> str:
        return "GELU()"
