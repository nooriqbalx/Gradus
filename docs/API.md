# Gradus API Reference

This is the public API surface: what you import, what each piece does,
and a minimal usage snippet for each. For *why* things are built the
way they are (design rationale, the two layers of "which primitives
need their own backward rule vs. which compose for free"), read the
module docstrings in the source directly -- `gradus/ops.py`,
`gradus/tensor.py`, and `gradus/nn/module.py` each explain their own
design in detail, and this document deliberately doesn't repeat that.
For evaluation results (correctness, benchmarks), see
`../BENCHMARK_REPORT.md`. For the four-layer architecture this API is
organized around, see `../README.md`'s Design section.

## `gradus.Tensor`

The core differentiable array type. Wraps a NumPy or CuPy array and
tracks the operations applied to it for reverse-mode autodiff.

```python
from gradus import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)  # dL/dx
```

| Member | Signature | Notes |
|---|---|---|
| `Tensor(data, requires_grad=False, device="cpu")` | constructor | `data` can be a Python list, NumPy array, or CuPy array. |
| `.backward(grad=None)` | | `grad` is required unless this tensor is a scalar (size 1), which defaults to `1.0`. |
| `.zero_grad()` | | Resets `.grad` to `None`. |
| `.to(device)` | `-> Tensor` | Moves data across CPU/GPU. Returns a new, graph-detached leaf -- see the docstring for why gradients don't flow through a device transfer. |
| `.astype(dtype)` / `.half()` / `.float()` / `.double()` (Phase 7) | `-> Tensor` | Casts data to a new dtype (float16/32/64), same device. Also graph-detaching, for the same reason as `.to()` -- no differentiable cast rule. |
| `.shape`, `.ndim` | properties | |
| `Tensor.zeros/ones(*shape, requires_grad=False, device="cpu", dtype=np.float64)` | staticmethods | Convenience constructors; `dtype` added Phase 7. |
| `Tensor.randn(*shape, requires_grad=False, device="cpu")` | staticmethod | |

**Operators** (all differentiable, all broadcasting-aware): `+ - * /
-x  x**p  x @ y`. **Methods**: `.sum(axis=None, keepdims=False)`,
`.mean(axis=None, keepdims=False)`, `.reshape(*shape)`,
`.transpose(*axes)`, `.T` (property, full reverse), `.exp()`, `.log()`,
`.relu()`, `.tanh()`.

## `gradus.ops`

The differentiable operations themselves, and where `Tensor`'s
operator overloads are actually implemented (imported for its *side
effects* -- attaching `__add__`, `__matmul__`, etc. onto `Tensor` --
not usually called directly). Two structural primitives worth knowing
about if you're extending the framework:

- `unfold(x, kernel_size, stride=1, padding=0)` -- im2col: the
  primitive `Conv2d` is built from (composed with `matmul`), rather
  than convolution needing its own backward rule.
- `embedding_lookup(weight, indices)` -- row gather with a
  scatter-add backward (`Embedding`'s primitive; handles a token
  appearing at multiple positions correctly, accumulating gradient
  from every occurrence).

Everything else (`add`, `sub`, `mul`, `div`, `matmul`, `sum_`, `mean`,
`reshape`, `transpose`, `exp`, `log`, `relu`, `tanh`, `pow_`, `neg`) is
reachable through `Tensor`'s operators/methods above -- see `ops.py`'s
module docstring for the full design rationale on primitive count.

## `gradus.nn`

### Base classes

```python
from gradus.nn import Module, Parameter, Sequential

class MyLayer(Module):
    def __init__(self, in_f, out_f):
        self.weight = Parameter(np.random.randn(out_f, in_f) * 0.01)

    def forward(self, x):
        return x @ self.weight.T
```

| Member | Notes |
|---|---|
| `Module` | Base class. Subclass, set `Parameter`/`Module` attributes in `__init__`, implement `forward()`. |
| `.parameters()` / `.named_parameters()` | Recurses into child modules and lists/tuples of modules automatically. |
| `.zero_grad()` | Zeros every parameter's gradient. |
| `.train()` / `.eval()` | Toggles `.training`, propagated to children -- consulted by `Dropout` and `BatchNorm2d`. |
| `.to(device)` | Moves every `Parameter` **and every registered buffer** (e.g. `BatchNorm2d`'s running stats) to `device`. **Call this before constructing an optimizer**, not after (see `optim`, below). |
| `.half()` / `.float()` / `.double()` (Phase 7) | Casts every `Parameter` and registered buffer to float16/32/64, **in place** (mutates and returns `self`, unlike `Tensor`'s dtype methods). Also call before constructing the optimizer -- see `optim`'s master-weight note below. |
| `.register_buffer(name, value)` | Marks a non-`Parameter` array (no gradient) so `.to()`/`.half()`/`.float()`/`.double()` move/cast it too. |
| `Parameter(data)` | A `Tensor` with `requires_grad=True`, found automatically by `.parameters()`. |
| `Sequential(*layers)` | Chains modules: `Sequential(Linear(784, 256), ReLU(), Linear(256, 10))`. |

### Layers

| Layer | Signature |
|---|---|
| `Linear` | `Linear(in_features, out_features, bias=True)` |
| `Conv2d` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)` |
| `BatchNorm2d` | `BatchNorm2d(num_features, eps=1e-5, momentum=0.1)` -- normalizes over `(N, H, W)` per channel |
| `LayerNorm` | `LayerNorm(normalized_shape, eps=1e-5)` -- normalizes over the last dimension |
| `Dropout` | `Dropout(p=0.5)` -- inverted dropout, no-op outside `.train()` mode |
| `Embedding` | `Embedding(vocab_size, embed_dim)` -- `forward(indices)` takes a **raw integer array**, not a `Tensor` (see the `ops.embedding_lookup` note above) |

### Activations

`ReLU()`, `Tanh()`, `GELU()` (tanh-approximation, matching PyTorch's
`nn.GELU(approximate="tanh")`) -- all parameter-free `Module`s.

### Losses

| Loss | Signature |
|---|---|
| `MSELoss` | `forward(pred, target)` |
| `CrossEntropyLoss` | `forward(logits, targets)` -- `logits` shape `(N, num_classes)`, `targets` integer labels shape `(N,)` |

### Attention & Transformer (Phase 3)

```python
from gradus.nn import TransformerLM

model = TransformerLM(
    vocab_size=len(vocab), embed_dim=32, num_heads=4,
    ff_dim=64, num_layers=2, max_len=32,
)
logits = model(token_ids)  # (batch, seq, vocab_size); token_ids is a raw int array
```

| Member | Notes |
|---|---|
| `MultiHeadAttention(embed_dim, num_heads)` | `forward(x, mask=None)`; causal masking via `causal_mask(seq_len)` |
| `causal_mask(seq_len)` | Returns a raw `(1, 1, seq_len, seq_len)` additive mask (`-1e4` where attention is disallowed -- safely underflows `exp()` to 0 in float16/32/64 alike; see the docstring for why this changed from `-1e9` in Phase 7) |
| `sinusoidal_positional_encoding(max_len, embed_dim)` | Fixed (non-learned) positional encoding, added to token embeddings |
| `TransformerBlock(embed_dim, num_heads, ff_dim, use_checkpoint=False)` | Pre-norm residual block: LayerNorm -> attention -> residual -> LayerNorm -> FFN -> residual. `use_checkpoint=True` (Phase 7) recomputes each sublayer during `backward()` instead of retaining its activations -- numerically identical output/gradients, see `gradus.utils.checkpoint` below. |
| `TransformerLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, use_checkpoint=False)` | Full decoder-only (GPT-style) causal language model; `use_checkpoint` is forwarded to every block. |

## `gradus.optim`

```python
from gradus.optim import Adam

model.to("cuda")                       # move first...
optimizer = Adam(model.parameters(), lr=1e-3)  # ...then construct the optimizer
```

| Optimizer | Signature |
|---|---|
| `SGD` | `SGD(parameters, lr=1e-2, momentum=0.0, use_master_weights=True)` |
| `Adam` | `Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, use_master_weights=True)` |

Both operate directly on `.data`/`.grad`, not through the autograd
graph. **Ordering matters for GPU use and for mixed precision**: both
allocate their own per-parameter state (momentum/moment buffers, and
Phase 7's float32 master-weight shadow) at construction time, on
whichever backend/dtype each parameter is on *at that moment* -- call
`model.to(device)` and `model.half()` first, construct the optimizer
after.

**Mixed precision (Phase 7):** `use_master_weights=True` (the default)
auto-detects any float16 parameter and keeps an internal float32
shadow copy for it, updated at full precision and cast down to
float16 once per step -- prevents small updates from rounding away to
nothing on an already-float16 array. Has no effect on float32/float64
parameters (byte-for-byte identical to Phase 1-6 behavior). Set to
`False` only to deliberately reproduce that failure mode (see
`PHASE7_REPORT.md`'s ablation).

## `gradus.Trainer`

```python
from gradus import Trainer

trainer = Trainer(model, Adam(model.parameters(), lr=1e-3), CrossEntropyLoss())
history = trainer.fit(train_batches, epochs=20, metric_fn=Trainer.accuracy)
# history == {"loss": [...], "metric": [...]}, one entry per epoch
```

`fit(batches, epochs=1, metric_fn=None, on_epoch_end=None)` re-iterates
`batches` once per epoch (pass a list or a fresh-generator-each-call
callable, not a single-use generator). `Trainer.accuracy` is a
ready-made `metric_fn` for classification.

## `gradus.amp` (Phase 7)

Dynamic loss scaling, protecting small gradients from underflowing
float16 during `backward()`. Shape mirrors `torch.cuda.amp.GradScaler`.

```python
from gradus.amp import GradScaler

scaler = GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)  # unscales grads, skips the step on inf/nan
scaler.update()          # grow/backoff the scale factor
```

| Member | Signature | Notes |
|---|---|---|
| `GradScaler(init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)` | constructor | |
| `.scale(loss)` | `-> Tensor` | Multiplies the loss by the current scale factor; call `.backward()` on the result. |
| `.step(optimizer)` | `-> bool` | Unscales every parameter's gradient, then calls `optimizer.step()` unless a gradient is inf/nan (in which case the step is skipped). Returns whether the step was taken. |
| `.update()` | | Grows the scale factor after `growth_interval` consecutive finite steps, or immediately backs it off by `backoff_factor` after an inf/nan step. Call once per training step, after `.step()`. |
| `.get_scale()` | `-> float` | Current scale factor. |

Addresses gradient underflow only -- combine with the optimizers'
`use_master_weights` (on by default) to also fix update underflow. See
`PHASE7_REPORT.md` for why both are needed.

## `gradus._grad_mode` (Phase 7)

A process-wide autograd on/off switch, this project's analogue of
`torch.no_grad()`.

```python
from gradus._grad_mode import no_grad

with no_grad():
    y = model(x)  # no graph retained -- y.requires_grad is False
```

Used internally by `gradus.utils.checkpoint`; also usable directly for
plain memory-saving inference.

## `gradus.backend`

The CPU/GPU dispatch layer everything above is built on -- most users
won't call this directly, but it's the mechanism behind every `.to()`.

| Function | Notes |
|---|---|
| `get_array_module(x)` | Returns `numpy` or `cupy` depending on `x`'s actual array type. |
| `array_module_for_device(device)` | Returns the array module for `"cpu"`/`"cuda"`, raising a helpful `RuntimeError` if CUDA is requested but unavailable. |
| `to_device(array, device)` | Moves a raw array between backends (no-op if already there). |
| `cupy_available()` | `True` only if CuPy is importable *and* a CUDA device is actually visible to it. |

## `gradus.utils`

Gradient-checking and reporting utilities (`gradus/utils/grad_check.py`)
-- the numerical-correctness half of the evaluation in
`../BENCHMARK_REPORT.md`.

| Function | Notes |
|---|---|
| `check_gradient(op_name, fn, inputs, eps=1e-5, seed=0)` | Finite-difference check of a raw op/function against Gradus's own backward pass. |
| `check_module_gradient(module, x_data, op_name=None, eps=1e-5, seed=0)` | Same, but for a `Module`: aggregates the input's and every parameter's check into one `GradCheckResult`. |
| `GradCheckResult.passed` | `True` if relative error `< 1e-4` **or** absolute error `< 1e-6` -- see the docstring for why relative error alone is an insufficient pass criterion. |
| `format_report(results)` | Renders a list of `GradCheckResult` as a Markdown table. |

See `benchmarks/grad_check.py` for these used end-to-end to produce
the full report.

**Gradient checkpointing (Phase 7, `gradus.utils.checkpoint`):**

```python
from gradus.utils import checkpoint

x = x + checkpoint(sublayer_fn, x, mask=mask)
```

| Function | Notes |
|---|---|
| `checkpoint(fn, *inputs, **kwargs)` | Runs `fn(*inputs, **kwargs)` under `no_grad()` (no graph retained), recomputing it during `backward()` only if gradient actually reaches that point. Numerically identical to calling `fn` directly -- a pure memory/compute trade-off, not an approximation. `fn` must be deterministic given the same inputs and RNG state (every layer in this project is). See `gradus.nn.transformer.TransformerBlock`'s `use_checkpoint` for the built-in use of this. |
