"""
Gradus: A CPU/GPU Differentiable Computing Framework.

Gradus is a from-scratch deep learning framework built to demonstrate
first-principles understanding of automatic differentiation, neural
network architectures, and hardware-aware execution. It is organized
into four layers:

    1. Foundations  -- Tensor abstraction, computational graph,
                        broadcasting, reverse-mode autodiff.
                        (see gradus.tensor, gradus.ops)
    2. Deep Learning -- Linear, Conv2d, normalization, dropout,
                        embedding, Transformer, SGD/Adam.
                        (see gradus.nn, gradus.optim)
    3. Systems      -- CPU (NumPy) / GPU (CuPy) backend abstraction,
                        device management, profiling.
                        (see gradus.backend)
    4. Evaluation   -- Gradient verification, convergence studies,
                        CPU/GPU benchmarking.
                        (see gradus.utils.grad_check, benchmarks/)

Status: Phases 1-7 complete (autodiff engine, nn layers, Transformer,
NumPy/CuPy backend abstraction, a full scientific evaluation,
packaging/hygiene, mixed precision, and gradient checkpointing -- 138
tests total (127 CPU + 11 GPU-only), all verified passing for real on
a Kaggle GPU notebook). Functional
correctness demonstrated on a CNN (digits, 97.78% held-out accuracy)
and a Transformer LM (original synthetic corpus, loss 2.59 -> 0.19).
See BENCHMARK_REPORT.md for the full evaluation.
"""

from gradus import ops  # noqa: F401  (attaches operator overloads onto Tensor)
from gradus.tensor import Tensor  # noqa: F401
from gradus.trainer import Trainer  # noqa: F401

__version__ = "0.1.0"
