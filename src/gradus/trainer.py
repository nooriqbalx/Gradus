"""
A minimal training loop, matching the README's target API:

    trainer = Trainer(model, optimizer, loss_fn)
    history = trainer.fit(train_batches, epochs=20)

Deliberately thin: it owns only the loop mechanics (iterate batches,
forward, backward, step, log), not data loading (that's the caller's
job -- `train_batches` is any iterable of (x, y) pairs) or anything
architecture-specific. Keeping it generic is what lets the same
Trainer drive both the CNN and the Transformer examples in Phase 3.
"""

from __future__ import annotations

from typing import Callable, Iterable

from gradus.nn.module import Module
from gradus.optim.optimizers import Optimizer
from gradus.tensor import Tensor


class Trainer:
    def __init__(self, model: Module, optimizer: Optimizer, loss_fn: Callable):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(
        self,
        batches: Iterable,
        epochs: int = 1,
        metric_fn: Callable = None,
        on_epoch_end: Callable = None,
    ) -> dict:
        """`batches` is re-iterated once per epoch, so pass something
        re-iterable (a list, or a callable/generator-function wrapped
        to build a fresh generator each epoch) rather than a
        single-use generator.

        Returns a history dict: {"loss": [...], "metric": [...]} with
        one entry per epoch (metric only populated if metric_fn given).
        """
        self.model.train()
        history = {"loss": [], "metric": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_metric = 0.0
            n_batches = 0

            for x, y in batches:
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.data)
                if metric_fn is not None:
                    epoch_metric += metric_fn(pred, y)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_metric = epoch_metric / max(n_batches, 1) if metric_fn is not None else None
            history["loss"].append(avg_loss)
            history["metric"].append(avg_metric)

            if on_epoch_end is not None:
                on_epoch_end(epoch, avg_loss, avg_metric)

        return history

    @staticmethod
    def accuracy(pred: Tensor, y) -> float:
        """Convenience metric_fn for classification: fraction of the
        batch where argmax(logits) matches the integer label."""
        import numpy as np

        from gradus.backend import to_device

        # This is a plain scalar metric, not part of the graph, so it's
        # fine (and simplest) to always finish the comparison on CPU --
        # a no-op to_device call when pred is already there, and a
        # small one-time transfer otherwise, since mixing a GPU-
        # resident array with a plain NumPy `y` directly would error.
        predicted_classes = to_device(pred.data.argmax(axis=-1), "cpu")
        return float((predicted_classes == np.asarray(y)).mean())
