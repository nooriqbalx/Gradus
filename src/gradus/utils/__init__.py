"""
Evaluation and verification utilities (Phase 5 / Layer 4 -- Scientific
Evaluation; extended Phase 7 with gradient checkpointing).

Contents:
    grad_check.py  -- per-op gradient checking: analytical vs. finite
                      difference, with a relative-error report table.
    checkpoint.py  -- gradient checkpointing (Phase 7): trade an extra
                      recompute for not retaining a sublayer's
                      activations across the forward/backward gap.
"""

from gradus.utils.checkpoint import checkpoint  # noqa: F401
from gradus.utils.grad_check import (  # noqa: F401
    GradCheckResult,
    absolute_error,
    check_gradient,
    check_module_gradient,
    format_report,
    numerical_gradient,
    relative_error,
)
