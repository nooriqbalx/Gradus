# Gradus

**A CPU/GPU Differentiable Computing Framework**

Gradus is a deep learning framework built from first principles: a
tensor-based reverse-mode automatic differentiation engine, standard
neural network layers through a Transformer block, and a shared
CPU/GPU execution backend, evaluated with a rigorous experimental
study of correctness and hardware performance.

## At a glance

| | |
|---|---|
| **Status** | Phases 1-7 complete: autodiff engine -> nn layers -> Transformer -> CPU/GPU backend -> scientific evaluation -> packaging -> mixed precision + gradient checkpointing |
| **Correctness** | 30/30 gradient checks passed; **138/138 tests passing for real on a Kaggle GPU notebook** (2x Tesla T4) -- not just locally |
| **CNN accuracy** | **97.78%** held-out test accuracy (digits dataset) |
| **Transformer** | loss **2.59 -> 0.19** on an original synthetic corpus, generating coherent continuations |
| **GPU speedup** | up to **14.7x** at scale (Phase 4-5); fp16 matmul adds up to **7.34x** more on top of that (Phase 7) |
| **Memory** | fp16 uses *exactly* half the memory of fp32; gradient checkpointing saves up to **83.9%** peak GPU memory |
| **Built from scratch** | no autodiff library, no PyTorch/TensorFlow underneath -- NumPy/CuPy is the only dependency doing array math |

Full methodology and every number above: **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**
(Phase 5) and **[PHASE7_REPORT.md](PHASE7_REPORT.md)** (Phase 7). Build
plan and the real bugs found along the way: **[ROADMAP.md](ROADMAP.md)**.
Full API reference: **[docs/API.md](docs/API.md)**.

## Why

Training a model with an existing framework (PyTorch, TensorFlow)
demonstrates the ability to *use* deep learning tools. Gradus instead
demonstrates *why* those tools work: implementing reverse-mode
autodiff correctly, handling computational graphs and broadcasting
gradients, managing parameters and optimizer state, and reasoning
about hardware execution, is a different -- and more fundamental --
technical signal.

## Design

```
                USER MODEL
                    |
                    v
              Tensor API
                    |
                    v
          Computational Graph
                    |
                    v
        Reverse-Mode Autodiff
                    |
             +------+------+
             v             v
            CPU           GPU
         NumPy backend   CuPy backend
             |             |
             +------+------+
                    v
             Neural Modules
                    |
         +----------+----------+
         v          v          v
       CNN      Transformer   ...
                    |
                    v
               Optimizers
                    |
                    v
               Training API
```

The same Tensor/autograd abstractions execute on either backend --
the systems contribution is that abstraction plus the empirical study
of the performance characteristics that result from it, not the CUDA
work itself (which CuPy provides).

## API (working today)

```python
from gradus import Tensor, Trainer
from gradus.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from gradus.optim import Adam

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
)

trainer = Trainer(model, Adam(model.parameters(), lr=1e-3), CrossEntropyLoss())
history = trainer.fit(train_batches, epochs=20, metric_fn=Trainer.accuracy)
```

A small Transformer language model:

```python
from gradus.nn import TransformerLM

model = TransformerLM(
    vocab_size=len(vocab), embed_dim=32, num_heads=4, ff_dim=64,
    num_layers=2, max_len=32,
)
logits = model(token_ids)  # (batch, seq, vocab_size)
```

Moving a model to GPU (Phase 4 -- call `.to("cuda")` on the model
*before* constructing the optimizer, not after, since the optimizer
snapshots each parameter's backend for its own momentum/moment
buffers):

```python
model.to("cuda")
trainer = Trainer(model, Adam(model.parameters(), lr=1e-3), CrossEntropyLoss())
history = trainer.fit(train_batches, epochs=20)  # plain NumPy batches auto-upload per op
```

Mixed-precision training (Phase 7 -- cast the model to fp16, then
construct the optimizer, which auto-detects fp16 parameters and keeps
its own float32 "master weight" shadow for them; `GradScaler` protects
small gradients from underflowing fp16 during `backward()`):

```python
from gradus.amp import GradScaler

model.half()  # cast every Parameter + buffer to float16
optimizer = Adam(model.parameters(), lr=1e-3)  # auto-detects fp16 params
scaler = GradScaler()

loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.step(optimizer)   # unscales, skips the step if a gradient overflowed
scaler.update()          # grows/backs off the scale factor
optimizer.zero_grad()
```

Gradient checkpointing (Phase 7 -- trade an extra recompute for not
retaining a sublayer's activations between the forward and backward
passes; numerically identical to `use_checkpoint=False`):

```python
model = TransformerLM(..., use_checkpoint=True)
```

Underneath that minimal surface: tensor abstraction, graph
construction, gradient propagation, parameter registry, optimizer
state, and (from Phase 4) CPU/GPU device management. Working examples:
`examples/cnn_digits.py`, `examples/toy_transformer.py`.

## Validation

Gradus is evaluated on three separate axes:

1. **Numerical correctness** -- every op (Phase 1), both structural
   primitives (`unfold`, `embedding_lookup`), every composed layer
   (Phase 2-3: Linear, Conv2d, BatchNorm2d, LayerNorm, attention, the
   full Transformer block), and both losses are checked via
   finite-difference gradient checking, not just tested for plausible
   output shapes. **30/30 checks passed**
   (`benchmarks/grad_check.py`). See `gradus/utils/grad_check.py` for
   methodology.
2. **Functional correctness** -- demonstrated in Phase 3: a CNN on
   scikit-learn's `digits` dataset (**97.78% held-out test accuracy**,
   `examples/cnn_digits.py`) and a small Transformer language model on
   an original synthetic corpus (**loss 2.59 -> 0.19**, generating
   coherent multi-sentence continuations, `examples/toy_transformer.py`).
   Convergence curves: `benchmarks/convergence_curves.py`. CIFAR-10 was
   the original plan; see ROADMAP.md's Phase 3 notes for why `digits`
   was substituted.
3. **Performance characterization + GPU correctness** -- CPU vs. GPU
   runtime across tensor size, model depth, batch size, and a full
   Transformer training step (`benchmarks/cpu_vs_gpu.py`), plus 9
   CPU/GPU correctness parity tests (`tests/test_gpu_parity.py`)
   comparing every layer's forward output and gradients between the
   two backends. **Both run for real on a Kaggle GPU notebook (2x
   Tesla T4): 9/9 parity tests passed, and GPU speedup reaches up to
   14.7x at scale** (with an honest look at where GPU is a wash or
   slower on small workloads) -- see [KAGGLE_STEPS.md](KAGGLE_STEPS.md)
   for the reproduction steps.

4. **Mixed precision + gradient checkpointing (Phase 7)** -- fp16
   numerical error measured directly against fp64 per layer, gradient
   underflow measured directly and shown fixed by `gradus.amp.GradScaler`,
   a synthetic demo proving fp32 master weights prevent update
   underflow, an honest convergence study, and gradient checkpointing
   proven bit-identical to the non-checkpointed path
   (`tests/test_checkpoint.py`) with its CPU compute/memory trade-off
   measured (`benchmarks/checkpoint_memory.py`). GPU throughput/memory
   verified for real on a Kaggle notebook -- fp16 matmul up to 7.3x
   faster and exactly half the memory of fp32; checkpointing saves up
   to 84% peak GPU memory at depth 32 -- see PHASE7_REPORT.md.

Full methodology and results live in
**[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)** (Phase 5) and
**[PHASE7_REPORT.md](PHASE7_REPORT.md)** (Phase 7).

## Installation (development)

```bash
git clone https://github.com/nooriqbalx/Gradus.git
cd gradus
pip install -e ".[dev]"
```

GPU support (optional, requires a CUDA-capable machine):

```bash
pip install -e ".[gpu]"
```

## Testing

```bash
pytest --cov=gradus --cov-report=term-missing
```

## Project structure

```
src/gradus/
    tensor.py        Tensor + autodiff engine + dtype casts (Phase 1, 7)
    ops.py           Differentiable operations                (Phase 1-2)
    _grad_mode.py    no_grad() context (for checkpointing)       (Phase 7)
    amp.py           GradScaler (loss scaling)                   (Phase 7)
    trainer.py       Trainer / fit() loop                      (Phase 2)
    nn/
        module.py       Module, Parameter, Sequential,
                        + half()/float()/double()          (Phase 2, 7)
        layers.py        Linear, Conv2d, BatchNorm2d,
                         LayerNorm, Dropout, Embedding           (Phase 2)
        activations.py   ReLU, GELU, Tanh                       (Phase 2)
        losses.py        MSELoss, CrossEntropyLoss              (Phase 2)
        functional.py    softmax, log_softmax, one_hot          (Phase 2)
        attention.py     MultiHeadAttention, positional enc.    (Phase 3)
        transformer.py   TransformerBlock (+ use_checkpoint),
                         TransformerLM                       (Phase 3, 7)
    optim/           SGD, Adam + fp32 master weights        (Phase 2, 7)
    backend/         NumPy / CuPy backend abstraction            (Phase 4)
    utils/           grad_check.py, checkpoint.py            (Phase 1, 5, 7)
benchmarks/          grad_check.py, convergence_curves.py,
                     cpu_vs_gpu.py, mixed_precision.py,
                     mixed_precision_gpu.py,
                     checkpoint_memory.py + results/       (Phase 5, 7)
examples/            cnn_digits.py, toy_transformer.py + saved
                     run logs in examples/results/
tests/               127 CPU tests + 11 GPU-only tests, all 138
                     verified on real Kaggle GPU hardware (9 in
                     Phase 4, 2 new ones in Phase 7)          (Phase 4, 7)
docs/                API.md -- full public API reference           (Phase 6)
.github/workflows/   ci.yml -- lint + test on Python 3.10-3.12     (Phase 6)
```

`BENCHMARK_REPORT.md` (repo root) is the full Phase 5 write-up: every
gradient-check result, both convergence curves, all 4 benchmark
sweeps, and the GPU parity results, with discussion. `PHASE7_REPORT.md`
is the Phase 7 write-up: mixed precision's numerical-error/underflow/
convergence studies, gradient checkpointing's correctness proof +
CPU overhead measurement, and both features' real GPU throughput/
memory numbers from a Kaggle run. `examples/` also
ships a pre-executed `.ipynb` alongside each `.py` script, so the
output/plots are readable without re-running anything.

## Explicitly out of scope

Gradus does not implement distributed/data-parallel training (cut to
protect a solo ~10-12 week timeline) and does not claim to be a novel
framework -- comparable educational frameworks include micrograd,
tinygrad, and Needle. The contribution here is the scope, the
implementation rigor, and the experimental evaluation.

## Author

Noor-ul-ain Iqbal ([ORCID](https://orcid.org/0009-0001-4016-0549))

## License

[MIT](LICENSE)
