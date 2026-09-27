"""
CPU/GPU backend abstraction (Phase 4 / Layer 3 -- Systems).

The design goal: the same Tensor/autograd/nn code paths dispatch to
either backend via the array's own type (get_array_module), rather
than having separate CPU and GPU implementations of the framework.
This module is what makes that possible; gradus.ops uses it internally
in every op instead of calling NumPy functions directly.

See device.py's module docstring for the honesty note on what has and
hasn't been verified against a real GPU yet.
"""

from gradus.backend.device import (  # noqa: F401
    array_module_for_device,
    cupy_available,
    get_array_module,
    to_device,
)
