"""
Neural network layers and architectures.

Phase 2: Module base class, parameter registry, Sequential, Linear,
Conv2d, BatchNorm2d, LayerNorm, Dropout, Embedding, activations
(ReLU, GELU, Tanh), losses (MSELoss, CrossEntropyLoss).

Phase 3: attention.py (multi-head attention, positional encoding),
transformer.py (Transformer block + a small language model).
"""

from gradus.nn.activations import GELU, ReLU, Tanh  # noqa: F401
from gradus.nn.attention import (  # noqa: F401
    MultiHeadAttention,
    causal_mask,
    sinusoidal_positional_encoding,
)
from gradus.nn.layers import (  # noqa: F401
    BatchNorm2d,
    Conv2d,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
)
from gradus.nn.losses import CrossEntropyLoss, MSELoss  # noqa: F401
from gradus.nn.module import Module, Parameter, Sequential  # noqa: F401
from gradus.nn.transformer import TransformerBlock, TransformerLM  # noqa: F401
