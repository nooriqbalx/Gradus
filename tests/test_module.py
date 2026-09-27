"""Tests for Module, Parameter, and Sequential (Phase 2)."""

import numpy as np

from gradus.nn import Linear, ReLU, Sequential
from gradus.nn.module import Module, Parameter


class _Tiny(Module):
    """A hand-built module nesting a submodule and a list of
    submodules, to exercise parameters()/named_parameters() recursion
    and train()/eval() propagation."""

    def __init__(self):
        self.fc = Linear(4, 3)
        self.blocks = [Linear(3, 3), Linear(3, 2)]


def test_parameter_requires_grad_by_default():
    p = Parameter(np.zeros(3))
    assert p.requires_grad is True


def test_sequential_parameters_collects_all_layers():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    params = list(model.parameters())
    # 2 Linear layers x (weight, bias) = 4 parameters.
    assert len(params) == 4


def test_nested_module_parameters_recurse_into_submodule_and_list():
    model = _Tiny()
    params = list(model.parameters())
    # fc (2) + blocks[0] (2) + blocks[1] (2) = 6
    assert len(params) == 6


def test_named_parameters_produces_dotted_names():
    model = _Tiny()
    names = [name for name, _ in model.named_parameters()]
    assert "fc.weight" in names
    assert "blocks.0.weight" in names
    assert "blocks.1.bias" in names


def test_zero_grad_clears_every_parameter():
    model = Sequential(Linear(4, 3))
    for p in model.parameters():
        p.grad = np.ones_like(p.data)
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_num_parameters_counts_elements_not_tensors():
    model = Sequential(Linear(4, 3))  # weight (3,4)=12 + bias (3,)=3
    assert model.num_parameters() == 15


def test_train_eval_propagates_to_submodules():
    model = _Tiny()
    model.eval()
    assert model.training is False
    assert model.fc.training is False
    assert model.blocks[0].training is False
    model.train()
    assert model.training is True
    assert model.blocks[1].training is True


def test_sequential_forward_chains_layers():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    from gradus import Tensor

    x = Tensor(np.random.randn(5, 4))
    out = model(x)
    assert out.shape == (5, 2)
