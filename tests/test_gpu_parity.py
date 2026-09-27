"""
CPU/GPU numerical parity tests (Phase 4).

These are the tests that actually exercise CuPy against a real GPU --
this dev sandbox has neither a GPU nor CuPy installed, so every test
here is SKIPPED locally (see the pytestmark below) and runs for the
first time on Kaggle. Nothing in this file has been run against a real
GPU as of writing; see ROADMAP.md's Phase 4 section for the full
context on what has and hasn't been verified, and KAGGLE_STEPS.md for
exactly how to run this.

Method: build the SAME model twice from the SAME global RNG seed (so
both copies start with identical initial weights), run one on CPU and
move the other to "cuda" via Module.to(), feed both the SAME input
data, and compare the forward output and every parameter's backward()
gradient with np.allclose. This deliberately does NOT re-derive
correctness from scratch -- Phase 1-3's gradient-check suite already
proved the math is right on CPU. All this checks is the thing that's
actually new and risky in Phase 4: does the CuPy code path compute the
SAME numbers as the already-verified NumPy path, for every op that
needed a GPU-specific line (embedding_lookup's scatter-add, one_hot's
scatter, BatchNorm2d's running stats, Dropout's mask, Adam's sqrt, ...).

If a test here FAILS on Kaggle: copy the full pytest -v output
(especially the "max abs diff" line each failure prints) back verbatim
-- that's the fastest way to localize which op's GPU path has a bug,
since each test isolates one layer/module.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradus import Tensor
from gradus.backend import cupy_available, to_device
from gradus.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Embedding,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    ReLU,
    Sequential,
)
from gradus.nn.transformer import TransformerBlock, TransformerLM

pytestmark = pytest.mark.skipif(
    not cupy_available(),
    reason="No CUDA GPU visible to CuPy -- these tests only run on a GPU "
    "machine (e.g. a Kaggle GPU notebook), not this CPU-only dev sandbox.",
)

# Both backends compute in float64 here (Tensor's default dtype), so
# CPU and GPU results should agree far tighter than float32 parity
# would allow -- this tolerance is deliberately tight enough to catch
# a real bug (e.g. a wrong axis in a GPU-specific fallback) while still
# giving floating-point summation-order differences room to exist.
RTOL, ATOL = 1e-6, 1e-8


def _assert_allclose(name: str, cpu_val: np.ndarray, gpu_val) -> None:
    gpu_val = to_device(gpu_val, "cpu")
    if not np.allclose(cpu_val, gpu_val, rtol=RTOL, atol=ATOL):
        diff = np.abs(cpu_val - gpu_val)
        raise AssertionError(
            f"{name}: CPU/GPU mismatch, max abs diff {diff.max():.3e} "
            f"at index {np.unravel_index(diff.argmax(), diff.shape)} "
            f"(cpu={cpu_val.flat[diff.argmax()]:.6f}, "
            f"gpu={gpu_val.flat[diff.argmax()]:.6f})"
        )


def _compare(build_model, x_data: np.ndarray, seed: int = 0, wrap_input: bool = True) -> None:
    """build_model() constructs a fresh Module. Runs it once on CPU and
    once on GPU (from identical initial weights, via the same seed) on
    the same input, then compares the forward output and every
    parameter's gradient after backward() from a fixed random linear
    projection to a scalar (the same reduction trick used in
    tests/_helpers.py, for the same reason: an output that sums to ~0
    by construction, like LayerNorm's, would make a plain .sum() a
    false-failure-prone check).

    wrap_input distinguishes the two calling conventions in this
    codebase: most layers' forward(x) expects x already as a Tensor
    (so its operators dispatch through ops.py), which is wrap_input's
    default -- the CPU input is built as a Tensor and moved to GPU via
    .to("cuda") (an actual data transfer, unlike tagging a Tensor's
    device string directly). Embedding/TransformerLM's forward expects
    RAW integer indices instead (embedding_lookup does its own
    device-resolution internally, see ops.py) -- pass wrap_input=False
    for those, which reuses the same raw NumPy array unchanged for both
    the CPU and GPU model calls.
    """
    np.random.seed(seed)
    cpu_model = build_model()
    cpu_input = Tensor(x_data.copy()) if wrap_input else x_data.copy()
    out_cpu = cpu_model(cpu_input)
    weights = np.random.default_rng(123).standard_normal(out_cpu.shape)
    (out_cpu * Tensor(weights)).sum().backward()
    cpu_grads = {name: p.grad.copy() for name, p in cpu_model.named_parameters()}
    cpu_out = out_cpu.data.copy()

    np.random.seed(seed)
    gpu_model = build_model()
    gpu_model.to("cuda")
    gpu_input = cpu_input.to("cuda") if wrap_input else x_data.copy()
    out_gpu = gpu_model(gpu_input)
    (out_gpu * weights).sum().backward()
    gpu_grads = dict(gpu_model.named_parameters())

    _assert_allclose("forward output", cpu_out, out_gpu.data)
    assert cpu_grads.keys() == gpu_grads.keys(), (
        f"parameter sets differ: cpu-only={cpu_grads.keys() - gpu_grads.keys()}, "
        f"gpu-only={gpu_grads.keys() - cpu_grads.keys()}"
    )
    for name in cpu_grads:
        assert gpu_grads[name].grad is not None, f"{name}: GPU run produced no gradient at all"
        _assert_allclose(f"grad[{name}]", cpu_grads[name], gpu_grads[name].grad)


def test_linear_parity():
    _compare(lambda: Linear(16, 8), np.random.default_rng(1).standard_normal((4, 16)))


def test_conv2d_batchnorm_relu_parity():
    """Exercises Conv2d (unfold/im2col), BatchNorm2d in training mode
    (the running-stats update this refactor made backend-aware), and
    ReLU together -- the exact composition examples/cnn_digits.py uses."""

    def build():
        return Sequential(Conv2d(2, 4, 3, padding=1), BatchNorm2d(4), ReLU())

    _compare(build, np.random.default_rng(2).standard_normal((3, 2, 8, 8)))


def test_batchnorm2d_eval_mode_parity():
    """Specifically exercises the .eval() branch, which reads back
    running_mean/running_var (registered buffers, moved by
    Module.to()) rather than the batch statistics."""

    def build():
        bn = BatchNorm2d(4)
        # Give it non-trivial running stats to read back, as if a few
        # training steps had already happened.
        bn.running_mean = np.random.default_rng(9).standard_normal(4)
        bn.running_var = np.abs(np.random.default_rng(10).standard_normal(4)) + 0.5
        bn.eval()
        return bn

    _compare(build, np.random.default_rng(3).standard_normal((3, 4, 5, 5)))


def test_layernorm_parity():
    _compare(lambda: LayerNorm(12), np.random.default_rng(4).standard_normal((5, 12)))


def test_embedding_parity():
    """Exercises embedding_lookup's backward, i.e. the scatter-add this
    refactor was most uncertain about (xp.add.at on CuPy) -- uses
    repeated indices on purpose so a token appears at multiple
    positions and must accumulate gradient from all of them, exactly
    like the CPU version's docstring promises."""
    indices = np.array([[3, 3, 1, 0], [2, 3, 3, 1]])  # index 3 repeats within AND across rows
    _compare(lambda: Embedding(10, 6), indices, wrap_input=False)


def test_multihead_attention_parity():
    _compare(
        lambda: MultiHeadAttention(embed_dim=16, num_heads=4),
        np.random.default_rng(5).standard_normal((2, 6, 16)),
    )


def test_transformer_lm_parity():
    """End-to-end: embedding + positional encoding (the raw-ndarray
    device-follow fix) + causal attention + LayerNorm/GELU FFN stack,
    i.e. the whole toy_transformer.py model in one shot."""

    def build():
        return TransformerLM(
            vocab_size=30, embed_dim=16, num_heads=2, ff_dim=32, num_layers=2, max_len=10
        )

    indices = np.random.default_rng(6).integers(0, 30, size=(2, 8))
    _compare(build, indices, wrap_input=False)


def test_cross_entropy_loss_parity():
    """CrossEntropyLoss's forward doesn't go through _compare's
    backward()-on-a-random-projection trick (its "reduction" is already
    a fixed scalar, the mean NLL) -- checked directly instead."""
    np.random.seed(7)
    model_cpu = Linear(10, 4)
    x = np.random.default_rng(8).standard_normal((5, 10))
    y = np.random.default_rng(9).integers(0, 4, size=(5,))
    loss_fn = CrossEntropyLoss()

    x_cpu = Tensor(x.copy())
    out_cpu = model_cpu(x_cpu)
    loss_cpu = loss_fn(out_cpu, y.copy())
    loss_cpu.backward()
    grad_cpu = model_cpu.weight.grad.copy()

    np.random.seed(7)
    model_gpu = Linear(10, 4)
    model_gpu.to("cuda")
    out_gpu = model_gpu(x_cpu.to("cuda"))  # actual transfer, not just a device-string tag
    loss_gpu = loss_fn(out_gpu, y.copy())
    loss_gpu.backward()

    _assert_allclose(
        "cross-entropy loss value",
        np.array([float(loss_cpu.data)]),
        np.array([float(to_device(loss_gpu.data, "cpu"))]),
    )
    _assert_allclose("linear.weight.grad", grad_cpu, model_gpu.weight.grad)


def test_adam_step_parity():
    """The optimizer never touches the autograd graph, so it needs its
    own direct check: run several Adam steps on CPU vs GPU from
    identical initial weights/gradients-per-step and compare the
    resulting weights (exercises Adam's GPU-backend-resolved sqrt and
    moment buffers)."""
    from gradus.optim import Adam

    np.random.seed(11)
    model_cpu = Linear(6, 3)
    opt_cpu = Adam(model_cpu.parameters(), lr=0.05)

    np.random.seed(11)
    model_gpu = Linear(6, 3)
    model_gpu.to("cuda")
    opt_gpu = Adam(model_gpu.parameters(), lr=0.05)

    rng = np.random.default_rng(12)
    for _ in range(5):
        x = rng.standard_normal((4, 6))
        target_weights = rng.standard_normal((4, 3))

        x_cpu = Tensor(x.copy())
        out_cpu = model_cpu(x_cpu)
        (out_cpu * target_weights).sum().backward()
        opt_cpu.step()
        opt_cpu.zero_grad()

        out_gpu = model_gpu(x_cpu.to("cuda"))
        (out_gpu * target_weights).sum().backward()
        opt_gpu.step()
        opt_gpu.zero_grad()

    _assert_allclose("weight after 5 Adam steps", model_cpu.weight.data, model_gpu.weight.data)
    _assert_allclose("bias after 5 Adam steps", model_cpu.bias.data, model_gpu.bias.data)


# ----------------------------------------------------------------------
# Phase 7 additions: mixed precision + gradient checkpointing, both of
# which now get their first real-GPU exercise here (Phase 4-6's parity
# tests above never touched a non-default dtype or no_grad()/
# checkpoint() at all -- these are new code paths on the GPU side).
# ----------------------------------------------------------------------

def test_transformer_block_checkpoint_parity_cpu_vs_gpu():
    """use_checkpoint=True re-runs a sublayer's forward a second time
    (with real grad tracking) inside backward() -- this test confirms
    that recompute happens correctly against the CuPy backend too, not
    just NumPy (tests/test_checkpoint.py already covers CPU). Same
    tight tolerance as the other parity tests above: checkpointing
    doesn't change dtype or arithmetic, only when it happens, so CPU
    and GPU should still agree to float64 precision."""

    def build():
        return TransformerBlock(embed_dim=16, num_heads=4, ff_dim=32, use_checkpoint=True)

    _compare(build, np.random.default_rng(13).standard_normal((2, 6, 16)))


def test_fp16_linear_parity_cpu_vs_gpu():
    """Module.half() + .to("cuda") (Phase 7 + Phase 4 composed): cast a
    model to float16 and run it on both backends from identical
    initial fp16 weights, comparing forward output and gradients.
    Deliberately a much looser tolerance than the float64 tests above
    -- float16 has roughly 3 decimal digits of precision, and CPU vs.
    GPU can legitimately sum a matmul's reduction axis in a different
    order, so small differences here are expected and not a bug (see
    PHASE7_REPORT.md's numerical-error study for how loose fp16
    agreement really needs to be, and why this project doesn't try to
    make fp16 CPU and GPU bit-identical)."""
    rtol, atol = 5e-2, 5e-2

    np.random.seed(14)
    cpu_model = Linear(16, 8)
    cpu_model.half()
    x_data = np.random.default_rng(15).standard_normal((4, 16)).astype(np.float16)
    x_cpu = Tensor(x_data.copy())
    out_cpu = cpu_model(x_cpu)
    weights = np.random.default_rng(16).standard_normal(out_cpu.shape).astype(np.float16)
    (out_cpu * Tensor(weights)).sum().backward()

    np.random.seed(14)
    gpu_model = Linear(16, 8)
    gpu_model.half()
    gpu_model.to("cuda")
    out_gpu = gpu_model(x_cpu.to("cuda"))
    (out_gpu * weights).sum().backward()

    cpu_out = to_device(out_cpu.data, "cpu").astype(np.float64)
    gpu_out = to_device(out_gpu.data, "cpu").astype(np.float64)
    assert np.allclose(cpu_out, gpu_out, rtol=rtol, atol=atol), (
        f"fp16 forward mismatch: max abs diff {np.abs(cpu_out - gpu_out).max():.4f}"
    )
    for name, p in cpu_model.named_parameters():
        gpu_p = dict(gpu_model.named_parameters())[name]
        cpu_grad = p.grad.astype(np.float64)
        gpu_grad = to_device(gpu_p.grad, "cpu").astype(np.float64)
        assert np.allclose(cpu_grad, gpu_grad, rtol=rtol, atol=atol), (
            f"fp16 grad[{name}] mismatch: max abs diff {np.abs(cpu_grad - gpu_grad).max():.4f}"
        )
