"""
Global autograd on/off switch (Phase 7, added for gradient
checkpointing but generally useful on its own -- this project's
analogue of torch.no_grad()).

A single process-wide flag, consulted by ops._make_output when
deciding whether to wire a freshly computed Tensor into the
computational graph (attach _prev/_backward) at all. On by default, so
nothing about Phase 1-6 behavior changes for any caller that never
touches this module.

Why this lives in its own tiny module rather than in tensor.py or
ops.py directly: tensor.py must not import ops.py (see ops.py's module
docstring on the split), but both tensor.py's Tensor.astype()-family
methods and ops.py's _make_output need to agree on the same flag, so
it lives somewhere both can import from without creating a cycle.
"""

from __future__ import annotations

_enabled = True


def is_grad_enabled() -> bool:
    return _enabled


def set_grad_enabled(mode: bool) -> None:
    global _enabled
    _enabled = mode


class no_grad:
    """Context manager: disable graph construction for every op run
    inside its body.

        with no_grad():
            y = model(x)   # y.requires_grad is False, no _prev/_backward
                            # retained anywhere in this forward pass

    Two uses in this project:
      1. Plain inference, to skip building a graph you were never
         going to call backward() on (saves memory the same way
         Module.eval() saves compute, and composes with it).
      2. gradus.utils.checkpoint's value-only forward pass (Phase 7) --
         run a sublayer's forward once under no_grad() to get its
         output value without retaining any of its internal
         activations, then recompute it a second time (with grad
         tracking back on) only if/when backward() actually reaches
         that point in the graph. Trades the extra recompute for not
         holding those activations in memory for the whole time
         between the forward and backward passes -- see
         gradus/utils/checkpoint.py.

    Nested `with no_grad()` blocks are safe (each __exit__ restores
    exactly the flag value that was in effect when its own __enter__
    ran, not a hardcoded True).
    """

    def __enter__(self) -> "no_grad":
        self._previous = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        set_grad_enabled(self._previous)
        return False
