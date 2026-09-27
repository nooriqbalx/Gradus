"""
Loss functions (Phase 2).

CrossEntropyLoss combines log-softmax and negative log-likelihood in
one step, matching PyTorch's nn.CrossEntropyLoss convention (it takes
raw logits, not probabilities). Internally it uses one-hot target
encoding + elementwise multiply + sum rather than an index-based
gather -- see gradus.nn.functional's docstring for why that avoids
needing a new primitive op.
"""

from __future__ import annotations

from gradus.backend import array_module_for_device
from gradus.nn.functional import log_softmax, one_hot
from gradus.nn.module import Module
from gradus.tensor import Tensor


class MSELoss(Module):
    def forward(self, pred: Tensor, target) -> Tensor:
        if not isinstance(target, Tensor):
            # A raw target array is converted onto pred's backend, not
            # always CPU -- a plain Tensor(np.asarray(...)) here would
            # silently pair a CPU array with a GPU-resident pred and
            # crash inside sub()'s a.data - b.data. dtype matches pred,
            # not a hardcoded float64 (Phase 7): pred - target on an
            # fp16 pred and a float64 target would upcast the whole
            # loss (and its backward pass) back to float64.
            xp = array_module_for_device(pred.device)
            target = Tensor(xp.asarray(target, dtype=pred.data.dtype), device=pred.device)
        return ((pred - target) ** 2).mean()

    def __repr__(self) -> str:
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """Expects raw logits of shape (N, num_classes) and integer class
    labels of shape (N,)."""

    def forward(self, logits: Tensor, targets) -> Tensor:
        num_classes = logits.shape[-1]
        xp = array_module_for_device(logits.device)
        targets = xp.asarray(targets)
        # constant, no grad; one_hot follows targets' own backend, so
        # passing already-converted (xp-backed) targets keeps this on
        # logits' device rather than round-tripping through CPU. dtype
        # matches logits (Phase 7), not one_hot's float64 default --
        # see one_hot's docstring for why that matters under fp16.
        target_one_hot = Tensor(
            one_hot(targets, num_classes, dtype=logits.data.dtype), device=logits.device
        )
        log_probs = log_softmax(logits, axis=-1)
        # Per-sample negative log-likelihood of the true class, then
        # averaged over the batch.
        nll = -(target_one_hot * log_probs).sum(axis=-1)
        return nll.mean()

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
