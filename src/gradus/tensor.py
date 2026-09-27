"""
Core Tensor abstraction and reverse-mode automatic differentiation
engine.

This module owns the *graph mechanics* only: what a Tensor is, how a
node records its parents and local backward rule, and how backward()
walks the resulting graph in reverse topological order. It
deliberately does not define what any individual operation (add,
matmul, ...) computes or how its gradient is derived -- those live in
gradus.ops, one function per operation, each of which also attaches
the corresponding dunder method (__add__, __mul__, ...) onto this
class at import time. See gradus/ops.py's module docstring for why
it's split this way.

Design notes:
    - Every Tensor that requires_grad keeps its own .grad, not just
      leaves (unlike PyTorch's default, which frees non-leaf grads).
      This trades a little memory for a framework where any
      intermediate tensor's gradient can be inspected directly --
      useful for debugging and for the per-op gradient-check suite.
    - Broadcasting is handled by each op's backward rule (via
      gradus.ops.unbroadcast), not here -- this module only orders and
      fires the backward calls.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from gradus.backend import get_array_module


class Tensor:
    """A differentiable n-dimensional array.

    Parameters
    ----------
    data : array-like
        The underlying values. If already a NumPy or CuPy array (duck-
        typed via hasattr(data, "shape"), not isinstance -- that's what
        lets this accept either backend's array without importing CuPy
        here), it's kept as-is (or cast to `dtype` if one was given);
        otherwise it's converted with np.array(data, dtype=dtype or
        float), i.e. raw Python data always lands on CPU first (move it
        with .to("cuda") afterwards).
    requires_grad : bool
        Whether this tensor should track operations for backward().
    device : str
        "cpu" or "cuda" -- should match whatever backend `data` is
        actually stored with; Tensor doesn't verify this itself
        (gradus.ops does, per-op, via get_array_module), so passing a
        NumPy array with device="cuda" produces a Tensor that silently
        lies about where its data lives. Prefer .to("cuda") over
        constructing with device="cuda" directly for that reason.
    dtype : numpy/cupy dtype, optional
        (Phase 7) Explicit dtype to store `data` as (e.g. np.float16
        for mixed-precision training). Defaults to float64 for raw
        Python data, or whatever dtype `data` already has if it's
        already an array and no dtype is given. Prefer `.half()` /
        `.float()` / `.astype()` over passing dtype= to an existing
        array you don't want silently re-cast.
    """

    def __init__(self, data, requires_grad: bool = False, device: str = "cpu", dtype=None):
        if hasattr(data, "shape") and hasattr(data, "dtype"):
            self.data = data if dtype is None else data.astype(dtype)  # NumPy or CuPy array
        else:
            self.data = np.array(data, dtype=dtype if dtype is not None else float)
        self.requires_grad = requires_grad
        self.device = device
        self.grad = None

        # Autograd graph bookkeeping. Populated by gradus.ops functions
        # when they produce this tensor as output; left empty for
        # leaves and for tensors that don't require grad (nothing to
        # walk back through, so we don't bother building that part of
        # the graph -- see ops._make_output).
        self._backward: Optional[Callable[[], None]] = None
        self._prev: Tuple["Tensor", ...] = ()
        self._op: str = ""

    # ------------------------------------------------------------------
    # Autodiff
    # ------------------------------------------------------------------

    def backward(self, grad=None) -> None:
        """Run reverse-mode autodiff starting from this tensor.

        Parameters
        ----------
        grad : array-like, optional
            Upstream gradient dL/d(self). Required unless this tensor
            is a scalar (size 1), in which case it defaults to 1.0 --
            i.e. this tensor is treated as the loss itself. Created on
            the SAME backend as self.data (via get_array_module), so
            starting backward() from a GPU-resident loss produces a
            GPU-resident gradient, not a NumPy array that would error
            the moment it's added to a CuPy array downstream.
        """
        if not self.requires_grad:
            raise RuntimeError("called backward() on a tensor with requires_grad=False")

        xp = get_array_module(self.data)
        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    "grad must be specified explicitly for a non-scalar tensor "
                    f"(shape={self.data.shape}); backward() with no argument is "
                    "only valid for scalar (loss) tensors"
                )
            grad = xp.ones_like(self.data)
        else:
            grad = xp.asarray(grad, dtype=self.data.dtype)

        self.grad = grad if self.grad is None else self.grad + grad

        # Reverse topological sort (DFS post-order, then reversed) so
        # every node fires its backward only after every tensor that
        # consumes it has already propagated gradient into it.
        topo: list["Tensor"] = []
        visited: set = set()

        def build(node: "Tensor") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()

    def zero_grad(self) -> None:
        """Reset this tensor's accumulated gradient to None."""
        self.grad = None

    # ------------------------------------------------------------------
    # Device management (Phase 4)
    # ------------------------------------------------------------------

    def to(self, device: str) -> "Tensor":
        """Move this tensor to a different device ('cpu' or 'cuda').

        Returns a new, graph-detached Tensor (no _prev/_backward) even
        if this one required grad -- moving data across the host/
        device boundary is treated as a boundary of the graph in this
        project, the same simplification PyTorch made before it added
        cross-device autograd support. If you need gradients to flow
        through a device transfer, restructure the model to move data
        once before entering the graph, not mid-computation.
        """
        from gradus.backend import to_device

        new_data = to_device(self.data, device)
        return Tensor(new_data, requires_grad=self.requires_grad, device=device)

    # ------------------------------------------------------------------
    # Dtype management (Phase 7 -- mixed precision)
    # ------------------------------------------------------------------

    def astype(self, dtype) -> "Tensor":
        """Cast this tensor's data to `dtype` (e.g. np.float16,
        np.float32, np.float64), on the SAME device.

        Like `.to()`, this returns a new, graph-detached Tensor even if
        this one required grad -- for the same reason: this project
        doesn't implement a differentiable cast (no backward rule
        turning an upstream fp32 gradient into a correctly-scaled fp16
        one), so a cast is treated as a graph boundary, not a graph
        node. In practice this is not a limitation for the mixed-
        precision recipe this project implements: casts happen once, to
        *leaf* parameters (via Module.half()), before training starts,
        never to an intermediate activation mid-graph -- so there is no
        graph to detach from at the point a cast actually happens.
        """
        xp = get_array_module(self.data)
        new_data = xp.asarray(self.data, dtype=dtype)
        return Tensor(new_data, requires_grad=self.requires_grad, device=self.device)

    def half(self) -> "Tensor":
        """Cast to float16. See `.astype()`'s docstring for why this
        detaches from the autograd graph."""
        xp = get_array_module(self.data)
        return self.astype(xp.float16)

    def float(self) -> "Tensor":
        """Cast to float32."""
        xp = get_array_module(self.data)
        return self.astype(xp.float32)

    def double(self) -> "Tensor":
        """Cast to float64 (this project's historical default dtype)."""
        xp = get_array_module(self.data)
        return self.astype(xp.float64)

    # ------------------------------------------------------------------
    # Convenience constructors
    #
    # dtype defaults to np.float64 here, not the bare builtin `float`
    # these two used before Phase 7 -- a real Python scoping gotcha
    # this project's own test suite caught: a class body is its own
    # namespace, and `def float(self): ...` above (Phase 7's cast
    # method) rebinds the name `float` in *this class's* namespace for
    # every def statement that follows it in the same body, including
    # these two staticmethods' default-argument expressions. A bare
    # `dtype=float` here silently became `dtype=Tensor.float` (the
    # unbound method object, not the builtin type), which failed at
    # call time with "Cannot interpret <function Tensor.float> as a
    # data type" the first time zeros()/ones() were called at all.
    # Spelling it np.float64 sidesteps the shadowing entirely.
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(
        *shape, requires_grad: bool = False, device: str = "cpu", dtype=np.float64
    ) -> "Tensor":
        from gradus.backend import array_module_for_device

        xp = array_module_for_device(device)
        return Tensor(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad, device=device)

    @staticmethod
    def ones(
        *shape, requires_grad: bool = False, device: str = "cpu", dtype=np.float64
    ) -> "Tensor":
        from gradus.backend import array_module_for_device

        xp = array_module_for_device(device)
        return Tensor(xp.ones(shape, dtype=dtype), requires_grad=requires_grad, device=device)

    @staticmethod
    def randn(*shape, requires_grad: bool = False, device: str = "cpu") -> "Tensor":
        from gradus.backend import array_module_for_device

        xp = array_module_for_device(device)
        return Tensor(xp.random.randn(*shape), requires_grad=requires_grad, device=device)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, "
            f"device={self.device!r})"
        )
