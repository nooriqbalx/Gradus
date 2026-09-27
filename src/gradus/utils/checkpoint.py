"""
Gradient checkpointing (Phase 7 -- Stretch, alternate/companion to
mixed precision).

The idea: for a sublayer whose activations are expensive to keep
around (TransformerBlock's attention + FFN sublayers, the largest
activation maps in this project's model zoo), don't retain them for
backward() at all. Instead:

    1. Forward pass: run the sublayer once under `no_grad()` to get
       its output *value* -- no graph, no intermediate activations
       retained, just the numbers needed to keep the rest of the
       network's forward pass going.
    2. Backward pass: when gradient actually reaches this point in the
       graph, run the sublayer a SECOND time (with grad tracking back
       on) to rebuild just that one sublayer's local graph, then call
       the ordinary .backward() on it to get gradients w.r.t. its
       inputs and parameters.

This trades compute (one extra forward pass per checkpointed sublayer,
only paid during backward()) for memory (that sublayer's internal
activations are never held simultaneously with the rest of the
network's -- see benchmarks/checkpoint_memory.py for the measured
trade-off on real GPU memory via CuPy's memory pool, and
BENCHMARK_REPORT.md's Section 6.3 for the numbers). It is exactly
transparent
numerically -- checkpointed and non-checkpointed forward/backward
produce identical output and identical gradients from the same
weights and inputs (see tests/test_checkpoint.py) -- because it
recomputes the exact same deterministic function a second time, it
doesn't approximate it.

RNG note: the two forward passes (value-only, then recompute) must
produce bit-identical activations for the above to hold, which matters
the moment a checkpointed sublayer contains anything stochastic
(Dropout). This is handled here by snapshotting NumPy's global legacy
RandomState (np.random.get_state()/set_state()) around each pass, the
same fix PyTorch's own torch.utils.checkpoint applies (and for the
same reason -- Linear/Conv2d/Embedding/Dropout in this project all
draw from that same global state, not a passed-in generator, see
layers.py). This covers CPU tensors exactly; a checkpointed sublayer
that runs on GPU (CuPy) and *also* contains internal randomness would
need cupy.random's own state snapshotted too (not implemented here --
CuPy's global RNG state API doesn't mirror NumPy's get_state()/
set_state() shape closely enough to fake it safely, and TransformerBlock,
the only place this project actually uses checkpoint(), has no Dropout
in either checkpointed sublayer, so it never exercises this gap). Any
future GPU module with internal randomness under checkpoint() should
be considered unverified until that's addressed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from gradus._grad_mode import is_grad_enabled, no_grad
from gradus.tensor import Tensor


def checkpoint(fn: Callable, *inputs, **kwargs) -> Tensor:
    """Run `fn(*inputs, **kwargs)` without retaining its internal
    autograd graph, recomputing it later (inside backward()) only if
    gradient actually reaches this point.

    `fn` must be a pure function of `inputs` (plus whatever Parameters
    it closes over, e.g. a bound method like `self._attn_sublayer`) --
    same output every time it's called with the same inputs and the
    same global RNG state, which is exactly what every layer in this
    project already is. `inputs` may mix Tensor and non-Tensor
    arguments (e.g. a plain ndarray mask); only Tensor inputs that
    require grad get gradients propagated back into them. `**kwargs`
    are passed through unchanged to both the value-only and the
    recompute call (e.g. `mask=mask` in TransformerBlock below) and
    are never treated as differentiable inputs.
    """
    rng_state = np.random.get_state()

    with no_grad():
        out_value = fn(*inputs, **kwargs)
    out_data = out_value.data

    # requires_grad here is NOT "do any of the *external* Tensor
    # inputs require grad" -- checkpoint() has no visibility into
    # whether `fn` closes over Parameters that need gradients too
    # (every real use in this project does: TransformerBlock's
    # _attn_sublayer/_ff_sublayer close over self.attn/self.norm1/
    # etc.'s weights), and it must still recompute + backward through
    # those even when the input tensor itself happens not to require
    # grad (e.g. checkpointing the very first layer of a network, fed
    # a plain non-Parameter input). So this tracks the *global*
    # no_grad() state instead, matching the intent "build a graph node
    # here unless the caller has explicitly said not to" -- a real
    # bug this project's own dry run caught: gating on the input
    # tensors' requires_grad alone silently dropped every gradient for
    # a checkpointed sublayer's parameters whenever its input tensor
    # didn't itself require grad.
    requires_grad = is_grad_enabled()
    device = next((x.device for x in inputs if isinstance(x, Tensor)), "cpu")

    def _backward() -> None:
        # Recompute with grad tracking back on (we're outside the
        # no_grad() block above by the time backward() runs), using
        # FRESH Tensor copies of the inputs rather than the originals.
        # This matters: the originals may already be nodes in the
        # OUTER graph (e.g. TransformerBlock's residual `x`, which has
        # its own _prev going back through the rest of the network).
        # Calling out_recompute.backward() below walks the *entire*
        # graph reachable from out_recompute's _prev chain -- if that
        # chain ran through the original `x` (with its own upstream
        # history attached), this would re-trigger backward() through
        # everything upstream of this checkpoint a second time,
        # double-counting those gradients. A fresh, graph-detached
        # copy (same data, no _prev/_backward of its own) makes the
        # recomputed subgraph self-contained: backward() stops exactly
        # at these copies, and their accumulated .grad is then added
        # into the ORIGINAL tensors' .grad by hand below, so the outer
        # graph's own (separate) backward walk picks it up normally
        # when it later reaches the original tensor's _backward.
        saved_state = np.random.get_state()
        np.random.set_state(rng_state)
        recompute_inputs = [
            Tensor(x.data, requires_grad=x.requires_grad, device=x.device)
            if isinstance(x, Tensor)
            else x
            for x in inputs
        ]
        out_recompute = fn(*recompute_inputs, **kwargs)
        out_recompute.backward(out.grad)
        np.random.set_state(saved_state)

        for orig, recomputed in zip(inputs, recompute_inputs):
            if not (isinstance(orig, Tensor) and orig.requires_grad):
                continue
            if recomputed.grad is None:
                continue
            orig.grad = recomputed.grad if orig.grad is None else orig.grad + recomputed.grad

    out = Tensor(out_data, requires_grad=requires_grad, device=device)
    if requires_grad:
        out._prev = tuple(x for x in inputs if isinstance(x, Tensor) and x.requires_grad)
        out._backward = _backward
    return out
