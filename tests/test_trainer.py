"""
End-to-end Phase 2 integration test: does the whole stack (layers +
loss + optimizer + Trainer) actually learn something, on two solvable
toy problems -- one regression, one classification. This is the
"functional correctness" leg of the evaluation (loss down, accuracy
up), one level above the per-op and per-layer gradient checks.
"""

import numpy as np

from gradus import Tensor, Trainer
from gradus.nn import CrossEntropyLoss, Linear, MSELoss, ReLU, Sequential
from gradus.optim import SGD, Adam

RNG = np.random.default_rng(0)


def test_trainer_fits_linear_regression():
    # y = 3x + 2 + small noise -- a linear model + MSE should drive
    # loss to near zero.
    x_data = RNG.uniform(-1, 1, size=(64, 1))
    y_data = 3 * x_data + 2 + RNG.normal(0, 0.01, size=(64, 1))
    batches = [(Tensor(x_data), y_data)]

    # Linear.__init__ draws its initial weights from the GLOBAL
    # np.random state (see layers.py), not from the RNG generator
    # above -- seed it explicitly so this test's outcome doesn't
    # depend on how much global random state whichever tests happened
    # to run before it already consumed (a real flake this project's
    # own suite surfaced: this test passed in isolation but could fail
    # when run after other tests, purely from unrelated global-RNG
    # draws shifting these weights' initial values -- see
    # test_trainer_fits_xor_with_nonlinearity's identical fix below).
    np.random.seed(0)
    model = Sequential(Linear(1, 1))
    trainer = Trainer(model, SGD(model.parameters(), lr=0.5), MSELoss())
    history = trainer.fit(batches, epochs=200)

    assert history["loss"][-1] < history["loss"][0]
    assert history["loss"][-1] < 0.01


def test_trainer_fits_xor_with_nonlinearity():
    # The canonical "a linear model can't do this, a nonlinearity can"
    # sanity check for the whole stack (Linear -> ReLU -> Linear).
    x_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_data = np.array([0, 1, 1, 0])
    batches = [(Tensor(x_data), y_data)]

    # Same reasoning as test_trainer_fits_linear_regression above:
    # Linear's weight init reads the global np.random state directly,
    # so this must be seeded for the test to be deterministic
    # regardless of what ran before it (a real, order-dependent flake:
    # this test asserts an exact 100% XOR accuracy after training,
    # which a sufficiently unlucky random init can genuinely miss
    # within 300 epochs -- it isn't asserting something false, it's
    # asserting something true only for *most*, not all, random inits,
    # so it needs one it's actually been verified against).
    np.random.seed(0)
    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 2))
    trainer = Trainer(model, Adam(model.parameters(), lr=0.05), CrossEntropyLoss())
    history = trainer.fit(batches, epochs=300, metric_fn=Trainer.accuracy)

    assert history["metric"][-1] == 1.0
    assert history["loss"][-1] < history["loss"][0]


def test_trainer_history_has_one_entry_per_epoch():
    x_data = RNG.standard_normal((8, 2))
    y_data = RNG.standard_normal((8, 1))
    batches = [(Tensor(x_data), y_data)]
    model = Sequential(Linear(2, 1))
    trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss())
    history = trainer.fit(batches, epochs=5)
    assert len(history["loss"]) == 5
