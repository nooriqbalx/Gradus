"""
Module base class, Parameter wrapper, and the Sequential container.

This is the Phase 2 foundation everything else in gradus.nn builds on:
a Module knows how to find every learnable Tensor inside itself
(including inside submodules it holds as attributes), which is what
lets an Optimizer be handed `model.parameters()` and update every
weight in the network without the model needing to expose them
manually.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from gradus.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that always requires grad and is meant to be found by
    Module.parameters() -- the only thing distinguishing it from a
    Tensor is this marker so a Module can walk its own __dict__ and
    tell "this is a learnable weight" apart from "this is some other
    Tensor I happen to be holding" (a cached constant, an input, ...).
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for anything with learnable parameters.

    Subclasses set their Parameters and child Modules as plain
    attributes in __init__ (self.weight = Parameter(...), self.fc1 =
    Linear(...)) and implement forward(). Calling the module
    (model(x)) dispatches to forward(x).

    parameters() recurses into child modules automatically -- there is
    no manual registration step, which keeps every layer's __init__
    focused on what it actually needs to construct.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def parameters(self) -> Iterator[Parameter]:
        for _, value in vars(self).items():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple]:
        for name, value in vars(self).items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                yield full_name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(prefix=full_name)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(prefix=f"{full_name}.{i}")

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    # ------------------------------------------------------------------
    # Buffers -- non-Parameter arrays (BatchNorm2d's running_mean/
    # running_var) that still need to move with the module when .to()
    # is called. They carry no gradient and aren't found by
    # parameters(), so they must be registered explicitly.
    # ------------------------------------------------------------------

    def register_buffer(self, name: str, value) -> None:
        setattr(self, name, value)
        if not hasattr(self, "_buffer_names"):
            self._buffer_names = []
        if name not in self._buffer_names:
            self._buffer_names.append(name)

    # ------------------------------------------------------------------
    # Device management (Phase 4) -- moves every Parameter and
    # registered buffer found anywhere in this module (recursing into
    # child modules and into lists/tuples of modules, same traversal as
    # parameters()) onto `device`.
    #
    # IMPORTANT ordering note: construct the Optimizer AFTER calling
    # model.to(device), not before -- SGD/Adam snapshot each
    # parameter's array module at construction time (for their
    # momentum/moment buffers), so moving the model afterwards would
    # leave those buffers on the old device while p.data/p.grad are on
    # the new one. This is the same ordering PyTorch requires.
    # ------------------------------------------------------------------

    def to(self, device: str) -> "Module":
        from gradus.backend import to_device

        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                new_param = Parameter(to_device(value.data, device))
                new_param.device = device
                setattr(self, name, new_param)
            elif isinstance(value, Module):
                value.to(device)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.to(device)
        for name in getattr(self, "_buffer_names", ()):
            setattr(self, name, to_device(getattr(self, name), device))
        return self

    # ------------------------------------------------------------------
    # Dtype management (Phase 7 -- mixed precision). Same traversal as
    # .to(), but casting dtype instead of moving device. Mutates in
    # place and returns self (unlike Tensor.half()/.float(), which
    # return new detached Tensors) -- there's no autograd graph to
    # detach a *leaf* Parameter from, so an in-place cast is both
    # simpler and matches PyTorch's nn.Module.half() convention.
    # ------------------------------------------------------------------

    def _cast(self, dtype) -> "Module":
        from gradus.backend import get_array_module

        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                xp = get_array_module(value.data)
                new_param = Parameter(xp.asarray(value.data, dtype=dtype))
                new_param.device = value.device
                setattr(self, name, new_param)
            elif isinstance(value, Module):
                value._cast(dtype)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item._cast(dtype)
        for name in getattr(self, "_buffer_names", ()):
            buf = getattr(self, name)
            xp = get_array_module(buf)
            setattr(self, name, xp.asarray(buf, dtype=dtype))
        return self

    def half(self) -> "Module":
        """Cast every Parameter and registered buffer to float16, in
        place. This is Gradus's mixed-precision entry point: cast the
        model once before training (after .to("cuda") if also moving
        device -- order doesn't matter between these two, unlike
        Optimizer construction, since dtype and device are independent
        axes here), then construct the optimizer as usual -- SGD/Adam
        auto-detect any float16 parameter and keep an internal float32
        "master weight" shadow copy + float32 moment buffers for it
        (see gradus.optim), so the tiny per-step updates Adam computes
        don't silently round away to nothing in fp16's ~3 decimal
        digits of precision. Combine with gradus.amp.GradScaler to also
        keep small gradients from underflowing fp16 during backward()
        (see BENCHMARK_REPORT.md's Section 5.3 numerical-error study
        for why both pieces are needed, not just one)."""
        return self._cast(np.float16)

    def float(self) -> "Module":
        """Cast every Parameter and buffer to float32 (the usual
        working precision for the fp16 recipe's master weights, and a
        reasonable middle ground if you want half the memory of
        float64 without fp16's precision/range problems)."""
        return self._cast(np.float32)

    def double(self) -> "Module":
        """Cast every Parameter and buffer to float64 (this project's
        historical default -- use to undo .half()/.float())."""
        return self._cast(np.float64)

    # ------------------------------------------------------------------
    # Train / eval mode -- consulted by Dropout and BatchNorm, which
    # behave differently at training vs. inference time. Propagates to
    # child modules the same way parameters() does.
    # ------------------------------------------------------------------

    training: bool = True

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)


class Sequential(Module):
    """Chains a list of modules, feeding each one's output to the
    next. The target API from the README:

        model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
    """

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {inner}\n)"

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
