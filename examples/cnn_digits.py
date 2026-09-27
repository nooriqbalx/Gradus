"""
Phase 3 functional-correctness demo #1: a small CNN trained on
scikit-learn's `digits` dataset (1797 8x8 grayscale images, 10 classes).

Why digits instead of CIFAR-10: the original target was CIFAR-10, but
this development environment has no GPU and no network access to
download it. digits ships inside scikit-learn (already a
dependency-free local dataset -- no download), so this proves the CNN
stack (Conv2d, BatchNorm2d, ReLU, strided downsampling, Linear,
CrossEntropyLoss, Adam, Trainer) trains correctly end-to-end without
depending on network access. The full GPU run happens later, on Kaggle.

Architecture note: downsampling uses a stride-2 Conv2d rather than a
separate MaxPool layer (the "all-convolutional net" approach, Springenberg
et al. 2014) -- this avoids needing a max-pooling backward (which would
need its own new argmax-routing primitive) while still being a
legitimate, standard CNN design choice.

Run: PYTHONPATH=src python3 examples/cnn_digits.py
"""

import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gradus import Tensor, Trainer
from gradus.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Linear, Module, ReLU
from gradus.optim import Adam


class DigitsCNN(Module):
    def __init__(self):
        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        self.bn2 = BatchNorm2d(16)
        self.relu2 = ReLU()
        self.fc = Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def make_batches(x, y, batch_size, rng):
    n = x.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield Tensor(x[idx]), y[idx]


class BatchesIterable:
    """Trainer.fit() re-iterates its `batches` argument once per epoch
    (see trainer.py's docstring) -- wrapping the generator function in
    an object with __iter__ gives it a fresh generator each time."""

    def __init__(self, gen_fn):
        self.gen_fn = gen_fn

    def __iter__(self):
        return self.gen_fn()


def main():
    # Seeds both the local generator (used for batching/shuffling
    # below) AND the global NumPy random state, since Linear/Conv2d
    # initialize their weights via np.random.uniform directly (not
    # through a passed-in generator) -- both need seeding for the run
    # to be fully reproducible.
    np.random.seed(0)
    rng = np.random.default_rng(0)

    digits = load_digits()
    x = digits.images.astype(np.float64) / 16.0  # pixel values are 0-16
    x = x[:, None, :, :]  # (N, 1, 8, 8) -- add channel dim
    y = digits.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    print(f"train: {x_train.shape[0]} samples, test: {x_test.shape[0]} samples")

    model = DigitsCNN()
    print(f"model parameters: {model.num_parameters():,}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, CrossEntropyLoss())

    epochs = 15
    batch_size = 32
    start = time.time()

    def batches():
        return make_batches(x_train, y_train, batch_size, rng)

    def log_epoch(epoch, loss, metric):
        print(f"epoch {epoch + 1:2d}/{epochs}  train_loss={loss:.4f}  train_acc={metric:.4f}")

    history = trainer.fit(
        BatchesIterable(batches), epochs=epochs, metric_fn=Trainer.accuracy, on_epoch_end=log_epoch
    )

    elapsed = time.time() - start

    model.eval()
    test_pred = model(Tensor(x_test))
    test_acc = Trainer.accuracy(test_pred, y_test)

    print(f"\ntraining time: {elapsed:.1f}s ({epochs} epochs, CPU)")
    print(f"final train loss: {history['loss'][-1]:.4f}")
    print(f"final train accuracy: {history['metric'][-1]:.4f}")
    print(f"held-out test accuracy: {test_acc:.4f}")

    return history, test_acc


if __name__ == "__main__":
    main()
