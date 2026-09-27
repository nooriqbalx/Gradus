"""
Phase 3 functional-correctness demo #2: a small decoder-only
Transformer trained as a char-level language model.

Corpus: a short set of original sentences written for this project
(not excerpted from any existing text), shuffled and repeated to build
a training set with real character/word-level statistical structure
to learn, rather than one string to memorize verbatim.

Run: PYTHONPATH=src python3 examples/toy_transformer.py
"""

import time

import numpy as np

from gradus import Trainer
from gradus.nn import CrossEntropyLoss, TransformerLM
from gradus.optim import Adam

SENTENCES = [
    "the small fox runs across the quiet field.",
    "a curious robot explores the old library at night.",
    "waves crash gently on the sandy shore.",
    "bright stars appear over the sleeping town.",
    "a young student writes code until dawn.",
    "rain falls softly on the green hills.",
    "the cat naps beside a warm fire.",
    "new ideas grow from patient practice.",
    "the river flows past ancient stone walls.",
    "children laugh and play in the summer sun.",
    "a quiet garden blooms behind the house.",
    "the train moves slowly through misty mountains.",
    "birds sing before the morning light.",
    "an old clock ticks in the empty hall.",
    "friends share stories under the tall trees.",
]


def build_corpus(rng, repeats: int = 15) -> str:
    """Shuffle sentence order on each repeat so the model has to learn
    real statistical structure (which letters/words follow which)
    rather than one fixed sequence position -> next-char mapping."""
    parts = []
    for _ in range(repeats):
        shuffled = list(SENTENCES)
        rng.shuffle(shuffled)
        parts.append(" ".join(shuffled))
    return " ".join(parts)


def make_batches(token_ids: np.ndarray, seq_len: int, batch_size: int, steps: int, rng):
    """Random contiguous windows of length seq_len+1 (input = first
    seq_len tokens, target = the next seq_len tokens, i.e. shifted by
    one -- standard next-token-prediction setup).

    Draws exactly `steps` random batches per call rather than covering
    the whole corpus -- with a CPU-only from-scratch autograd engine,
    Python-level graph-construction overhead (not FLOPs) dominates
    cost, so total step count is what needs to be budgeted, not corpus
    coverage. A fixed step budget keeps this demo's runtime predictable
    regardless of corpus size.
    """
    n = len(token_ids) - seq_len - 1
    for _ in range(steps):
        batch_starts = rng.integers(0, n, size=batch_size)
        x = np.stack([token_ids[s : s + seq_len] for s in batch_starts])
        y = np.stack([token_ids[s + 1 : s + seq_len + 1] for s in batch_starts])
        yield x, y


class BatchesIterable:
    def __init__(self, gen_fn):
        self.gen_fn = gen_fn

    def __iter__(self):
        return self.gen_fn()


def generate(model, stoi, itos, prompt: str, length: int, seq_len: int, rng) -> str:
    """Greedy-ish sampling (temperature-scaled multinomial draw) to
    qualitatively show the model learned something -- this is
    illustrative, not part of the correctness evaluation."""
    token_ids = [stoi[c] for c in prompt]
    for _ in range(length):
        context = token_ids[-seq_len:]
        x = np.array([context])
        logits = model(x)
        last_logits = logits.data[0, -1]
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        next_id = rng.choice(len(probs), p=probs)
        token_ids.append(int(next_id))
    return "".join(itos[i] for i in token_ids)


def main():
    # Seeds both the local generator AND the global NumPy random state
    # (Embedding/Linear initialize weights via np.random.randn/uniform
    # directly), for a fully reproducible run.
    np.random.seed(0)
    rng = np.random.default_rng(0)

    corpus = build_corpus(rng)
    vocab = sorted(set(corpus))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    token_ids = np.array([stoi[c] for c in corpus])

    print(f"corpus length: {len(corpus)} chars, vocab size: {len(vocab)}")

    seq_len = 32
    model = TransformerLM(
        vocab_size=len(vocab), embed_dim=32, num_heads=4, ff_dim=64, num_layers=2, max_len=seq_len
    )
    print(f"model parameters: {model.num_parameters():,}")

    optimizer = Adam(model.parameters(), lr=3e-3)
    loss_fn = CrossEntropyLoss()

    def loss_wrapper(logits, targets):
        # CrossEntropyLoss expects (N, num_classes); flatten the
        # (batch, seq, vocab) logits and (batch, seq) targets together.
        # logits.reshape() goes through the differentiable graph (it's
        # a Tensor method backed by gradus.ops.reshape), so gradients
        # still flow correctly back through this flattening.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        return loss_fn(flat_logits, flat_targets)

    trainer = Trainer(model, optimizer, loss_wrapper)

    epochs = 20
    batch_size = 32
    steps_per_epoch = 100  # ~2000 total steps -- see make_batches' docstring
    start = time.time()

    def batches():
        return make_batches(token_ids, seq_len, batch_size, steps_per_epoch, rng)

    def log_epoch(epoch, loss, metric):
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:2d}/{epochs}  loss={loss:.4f}")

    history = trainer.fit(BatchesIterable(batches), epochs=epochs, on_epoch_end=log_epoch)
    elapsed = time.time() - start

    print(f"\ntraining time: {elapsed:.1f}s ({epochs} epochs, CPU)")
    print(f"initial loss: {history['loss'][0]:.4f}")
    print(f"final loss: {history['loss'][-1]:.4f}")

    model.eval()
    sample = generate(model, stoi, itos, prompt="the ", length=80, seq_len=seq_len, rng=rng)
    print(f"\nsample continuation (illustrative, not a correctness metric):\n{sample!r}")

    return history


if __name__ == "__main__":
    main()
