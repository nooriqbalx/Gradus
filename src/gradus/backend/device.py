"""
Device / array-module resolution for the CPU (NumPy) / GPU (CuPy)
backend abstraction (Phase 4).

Design: rather than every op branching on a Tensor.device string,
gradus.ops asks "what array module does this actual array belong to?"
via get_array_module() -- this is CuPy's own idiom
(cupy.get_array_module(x) returns numpy for a NumPy array and cupy for
a CuPy array), extended here to work even when CuPy isn't installed at
all. This project was developed on a CPU-only sandbox with no GPU and
no way to install/test CuPy; get_array_module() falls back to plain
NumPy whenever CuPy isn't importable, which is why the entire CPU code
path (Phases 1-3, all 93 tests) is completely unaffected by whether
CuPy is present -- the GPU path is additive, not a rewrite.

This CUDA path was developed with no GPU in the dev environment, then
verified for real on a Kaggle GPU notebook (2x Tesla T4) -- see
tests/test_gpu_parity.py, which compares every layer's forward output
and gradients between a CPU run and a GPU run from identical initial
weights.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as _cupy

    _CUPY_IMPORTABLE = True
except ImportError:
    _cupy = None
    _CUPY_IMPORTABLE = False


def cupy_available() -> bool:
    """True if CuPy is importable AND at least one CUDA GPU is visible
    to it -- not just installed (the import can succeed with no GPU
    present, e.g. a CPU-only machine with cupy pip-installed anyway,
    but every actual op call would then fail)."""
    if not _CUPY_IMPORTABLE:
        return False
    try:
        return _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_array_module(x):
    """numpy for a NumPy array (or any non-array Python value), cupy
    for a CuPy array. Used inside every op in gradus.ops instead of a
    hardcoded `np.something(...)` call, so the same op function runs
    on whichever backend its input actually lives on."""
    if _CUPY_IMPORTABLE:
        return _cupy.get_array_module(x)
    return np


def array_module_for_device(device: str):
    """The array module to use when there's no existing array to
    introspect (e.g. wrapping a raw Python scalar constant, or
    Tensor.zeros/.ones/.randn) -- resolved from the target device
    string rather than from an array's actual type."""
    if device == "cpu":
        return np
    if device == "cuda":
        if not _CUPY_IMPORTABLE:
            raise RuntimeError(
                "device='cuda' requested but CuPy is not installed. Install it "
                "with `pip install cupy-cuda12x` (or the cupy-cudaXXx build "
                "matching your CUDA version) on a machine with an NVIDIA GPU "
                "-- e.g. a free Kaggle GPU notebook."
            )
        if not cupy_available():
            raise RuntimeError(
                "CuPy is installed but no CUDA GPU is visible to it "
                "(cupy.cuda.runtime.getDeviceCount() == 0). On Kaggle, check "
                "that the notebook's Accelerator is set to GPU."
            )
        return _cupy
    raise ValueError(f"unknown device {device!r} (expected 'cpu' or 'cuda')")


def to_device(array, device: str):
    """Move a NumPy or CuPy array to `device`, copying across the
    host/device boundary only when actually necessary (a no-op if the
    array is already on the requested device)."""
    xp = get_array_module(array)
    target = array_module_for_device(device)
    if xp is target:
        return array
    if device == "cpu":
        return _cupy.asnumpy(array)
    return target.asarray(array)
