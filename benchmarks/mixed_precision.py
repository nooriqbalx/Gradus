"""
Mixed-precision numerical-error and convergence study -- the CPU-only
half of the full throughput/memory/convergence/numerical-error study
vs. FP32. The throughput/memory half needs real GPU tensor-core
hardware and lives in benchmarks/mixed_precision_gpu.py instead, run
on a free Kaggle GPU notebook the same way cpu_vs_gpu.py was.

Two independent studies:

  1. Numerical error (per-layer): forward output and gradient of a
     handful of representative layers, computed once at float64 (this
     project's historical default -- treated as ground truth here) and
     once at float16, from IDENTICAL initial weights. Reports max
     relative and absolute error for each.

  2. Convergence (end-to-end): the Phase 3 CNN-on-digits problem
     (examples/cnn_digits.py's exact architecture), trained for a few
     epochs under four configurations to isolate each fix's individual
     contribution:
        a. fp32                                    (reference)
        b. fp16, no GradScaler, no master weights   (expected: stalls
           almost immediately -- gradient underflow)
        c. fp16, GradScaler, no master weights      (expected: gradient
           underflow fixed, but update underflow still stalls it --
           see optim/optimizers.py's module docstring)
        d. fp16, GradScaler + master weights        (expected: tracks
           the fp32 curve)
     This is the concrete demonstration that BOTH fixes are needed,
     not just one -- see gradus/amp.py's module docstring for why they
     address two distinct failure modes.

Usage:
    python benchmarks/mixed_precision.py
    python benchmarks/mixed_precision.py --out benchmarks/results/mixed_precision_report.md
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gradus import Tensor
from gradus.amp import GradScaler
from gradus.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    LayerNorm,
    Linear,
    Module,
    MultiHeadAttention,
    ReLU,
)
from gradus.nn.attention import causal_mask
from gradus.optim import Adam
from gradus.utils import absolute_error, relative_error

# ----------------------------------------------------------------------
# Study 1: per-layer numerical error, fp16 vs fp64
# ----------------------------------------------------------------------


def _compare_precision(build_fn, x_data: np.ndarray, seed: int = 0, extra_fwd_kwargs=None):
    """Build the SAME layer twice (identical initial weights, via the
    same seed), run one at float64 and one at float16 (Module.half()),
    on the same input data, and compare forward output + gradient
    w.r.t. the input via relative_error/absolute_error. Reduces to a
    scalar for backward() via a fixed random projection (same reason
    as tests/_helpers.py: a plain .sum() would be a false-failure-prone
    check for a mean-subtracting layer like LayerNorm)."""
    extra_fwd_kwargs = extra_fwd_kwargs or {}

    np.random.seed(seed)
    model64 = build_fn()
    x64 = Tensor(x_data.astype(np.float64), requires_grad=True)
    out64 = model64(x64, **extra_fwd_kwargs)
    weights = np.random.default_rng(123).standard_normal(out64.shape)
    (out64 * Tensor(weights)).sum().backward()

    np.random.seed(seed)
    model16 = build_fn()
    model16.half()
    x16 = Tensor(x_data.astype(np.float16), requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out16 = model16(x16, **extra_fwd_kwargs)
        (out16 * weights.astype(np.float16)).sum().backward()

    out64_np = out64.data.astype(np.float64)
    out16_np = out16.data.astype(np.float64)
    grad64 = x64.grad.astype(np.float64)
    grad16 = x16.grad.astype(np.float64)

    return {
        "out_rel": relative_error(out64_np, out16_np),
        "out_abs": absolute_error(out64_np, out16_np),
        "grad_rel": relative_error(grad64, grad16),
        "grad_abs": absolute_error(grad64, grad16),
    }


def numerical_error_study() -> list:
    rng = np.random.default_rng(0)
    results = []

    results.append(
        ("Linear(64, 32)", _compare_precision(lambda: Linear(64, 32), rng.standard_normal((8, 64))))
    )

    def build_conv_bn_relu():
        return _Sequential3(Conv2d(2, 4, 3, padding=1), BatchNorm2d(4), ReLU())

    results.append(
        (
            "Conv2d+BatchNorm2d+ReLU",
            _compare_precision(build_conv_bn_relu, rng.standard_normal((3, 2, 8, 8))),
        )
    )
    results.append(
        ("LayerNorm(32)", _compare_precision(lambda: LayerNorm(32), rng.standard_normal((8, 32))))
    )
    results.append(
        (
            "MultiHeadAttention(32, 4h) + causal mask",
            _compare_precision(
                lambda: MultiHeadAttention(embed_dim=32, num_heads=4),
                rng.standard_normal((2, 6, 32)),
                extra_fwd_kwargs={"mask": causal_mask(6)},
            ),
        )
    )
    return results


class _Sequential3(Module):
    """Tiny local stand-in for Sequential that forwards **kwargs to the
    first layer only (Conv2d/BatchNorm2d/ReLU here take none anyway) --
    avoids importing gradus.nn.module.Sequential just to add a kwarg
    passthrough it doesn't need elsewhere."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x


def format_numerical_error_table(results: list) -> str:
    lines = [
        "| Layer | output max rel err | output max abs err | grad max rel err | grad max abs err |",
        "|---|---|---|---|---|",
    ]
    for name, r in results:
        lines.append(
            f"| {name} | {r['out_rel']:.3e} | {r['out_abs']:.3e} | "
            f"{r['grad_rel']:.3e} | {r['grad_abs']:.3e} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Study 1b: gradient-underflow fraction, with vs. without GradScaler
#
# The convergence study below (Study 2) turns out NOT to show a
# dramatic difference between "fp16, no GradScaler" and the scaled
# variants for DigitsCNN -- worth measuring directly WHY, rather than
# just reporting a flat curve and moving on. This isolates the actual
# mechanism: what fraction of each parameter's gradient elements
# underflow to exactly 0.0 in fp16 on a real backward pass, with vs.
# without loss scaling.
# ----------------------------------------------------------------------


def gradient_underflow_study(x_train, y_train, batch_size: int = 32, seed: int = 0) -> list:
    np.random.seed(seed)
    model = DigitsCNN()
    model.half()
    loss_fn = CrossEntropyLoss()
    xb = Tensor(x_train[:batch_size].astype(np.float16))
    yb = y_train[:batch_size]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
    unscaled_grads = {name: p.grad.copy() for name, p in model.named_parameters()}

    np.random.seed(seed)
    model2 = DigitsCNN()
    model2.half()
    scaler = GradScaler(init_scale=2.0**16)
    xb2 = Tensor(x_train[:batch_size].astype(np.float16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pred2 = model2(xb2)
        loss2 = loss_fn(pred2, yb)
        scaler.scale(loss2).backward()
    scaled_grads = {name: p.grad.copy() for name, p in model2.named_parameters()}

    rows = []
    for name in unscaled_grads:
        g = unscaled_grads[name]
        zero_frac_unscaled = float(np.mean(g == 0))
        # Recover the scaled run's TRUE (unscaled) gradient the same
        # way GradScaler.step() does, to check how much of the
        # zero-fraction above was recovered by scaling before it ever
        # got the chance to underflow.
        g_scaled_then_unscaled = scaled_grads[name].astype(np.float32) / scaler.get_scale()
        zero_frac_scaled = float(np.mean(g_scaled_then_unscaled == 0))
        rows.append((name, zero_frac_unscaled, zero_frac_scaled))
    return rows


def format_underflow_table(rows: list) -> str:
    lines = [
        "| parameter | zero-grad fraction, no scaling | zero-grad fraction, with GradScaler |",
        "|---|---|---|",
    ]
    for name, unscaled, scaled in rows:
        lines.append(f"| {name} | {unscaled:.2%} | {scaled:.2%} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Study 1c: synthetic "update underflow" demonstration
#
# DigitsCNN's weights stay small in magnitude throughout training
# (typically < 1), where float16's ULP is fine enough that the master-
# weight shadow rarely matters in practice for this specific model --
# Study 2 below shows all four configurations converging almost
# identically as a direct consequence. That is a real, honestly
# reported result, not a failure to find the effect: the effect is
# real, it just needs a parameter magnitude where fp16's ULP is
# actually coarser than a typical update, which this demo constructs
# directly (see optim/optimizers.py's module docstring for why larger-
# magnitude parameters are where this matters -- e.g. an embedding
# table entry, or any weight that isn't kept near-zero by
# normalization the way DigitsCNN's BatchNorm layers keep this one).
# ----------------------------------------------------------------------


def synthetic_update_underflow_demo(n_steps: int = 2000) -> dict:
    from gradus.nn.module import Parameter
    from gradus.optim import Adam as _Adam

    naive = np.array([1000.0], dtype=np.float16)
    p_master = Parameter(np.array([1000.0], dtype=np.float16))
    opt_master = _Adam([p_master], lr=1e-3, use_master_weights=True)

    naive_trace, master_trace = [], []
    for _ in range(n_steps):
        naive -= np.float16(1e-3)  # same per-step update magnitude as Adam settles to here
        p_master.grad = np.array([0.001], dtype=np.float16)
        opt_master.step()
        naive_trace.append(float(naive[0]))
        master_trace.append(float(p_master.data[0]))

    return {"naive_fp16_inplace": naive_trace, "adam_with_master_weights": master_trace}


# ----------------------------------------------------------------------
# Study 2: end-to-end convergence, fp32 vs three fp16 configurations
# ----------------------------------------------------------------------


class DigitsCNN(Module):
    """Identical architecture to examples/cnn_digits.py's DigitsCNN."""

    def __init__(self):
        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = BatchNorm2d(16)
        self.relu2 = ReLU()
        self.fc = Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def _load_digits_split(seed: int = 0):
    digits = load_digits()
    x = digits.images.astype(np.float64) / 16.0
    x = x[:, None, :, :]
    y = digits.target
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)


def _make_batches(x, y, batch_size, rng):
    n = x.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]


def train_variant(
    precision: str,
    use_scaler: bool,
    use_master_weights: bool,
    x_train,
    y_train,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> list:
    """Returns per-epoch average loss. `precision` is "fp32" or
    "fp16"; use_scaler/use_master_weights are only meaningful for
    fp16 (silently ignored for fp32, matching how a real training
    script would just not bother constructing a GradScaler for fp32)."""
    np.random.seed(seed)
    model = DigitsCNN()
    if precision == "fp16":
        model.half()
    elif precision == "fp32":
        model.float()

    opt = Adam(model.parameters(), lr=lr, use_master_weights=use_master_weights)
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler() if (precision == "fp16" and use_scaler) else None
    rng = np.random.default_rng(seed)

    history = []
    for _epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in _make_batches(x_train, y_train, batch_size, rng):
            x_tensor = Tensor(xb.astype(np.float16 if precision == "fp16" else np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pred = model(x_tensor)
                loss = loss_fn(pred, yb)

                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            # `loss` itself is always the UNSCALED loss tensor -- scaler.scale(loss)
            # returns a new tensor (loss * scale) for backward() to start from, it
            # doesn't mutate `loss` in place, so no unscaling is needed here to get
            # a fair, comparable value across all four configurations.
            epoch_loss += float(loss.data)
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
    return history


def convergence_study(x_train, y_train, epochs: int = 8) -> dict:
    configs = {
        "fp32 (reference)": dict(precision="fp32", use_scaler=False, use_master_weights=False),
        "fp16, no GradScaler, no master weights": dict(
            precision="fp16", use_scaler=False, use_master_weights=False
        ),
        "fp16, GradScaler, no master weights": dict(
            precision="fp16", use_scaler=True, use_master_weights=False
        ),
        "fp16, GradScaler + master weights": dict(
            precision="fp16", use_scaler=True, use_master_weights=True
        ),
    }
    return {
        name: train_variant(**cfg, x_train=x_train, y_train=y_train, epochs=epochs)
        for name, cfg in configs.items()
    }


def format_convergence_table(histories: dict) -> str:
    names = list(histories.keys())
    epochs = len(next(iter(histories.values())))
    lines = ["| epoch | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for e in range(epochs):
        row = [f"{histories[n][e]:.4f}" for n in names]
        lines.append(f"| {e + 1} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_convergence(histories: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, marker="o", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (unscaled)")
    ax.set_title("Mixed-precision convergence: DigitsCNN, Adam lr=1e-3")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Write a Markdown report here.")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per convergence variant.")
    parser.add_argument(
        "--plot", type=str, default="benchmarks/results/mixed_precision_convergence.png"
    )
    args = parser.parse_args()

    print("=== Numerical error study: fp16 vs fp64, per layer ===\n")
    error_results = numerical_error_study()
    table1 = format_numerical_error_table(error_results)
    print(table1)

    print("\n=== Gradient underflow: fraction of zero grad elements, with/without GradScaler ===\n")
    x_train, _x_test, y_train, _y_test = _load_digits_split()
    underflow_rows = gradient_underflow_study(x_train, y_train)
    table1b = format_underflow_table(underflow_rows)
    print(table1b)

    print("\n=== Synthetic update-underflow demo (large-magnitude parameter) ===\n")
    demo = synthetic_update_underflow_demo()
    print(
        f"parameter starts at 1000.0, receives 2000 identical tiny updates "
        f"(~1e-3 each, ~2.0 total):\n"
        f"  naive in-place fp16 update           -> {demo['naive_fp16_inplace'][-1]!r} "
        f"(never moved: each update alone is below fp16's ULP at this magnitude)\n"
        f"  Adam + fp32 master-weight shadow     -> {demo['adam_with_master_weights'][-1]!r} "
        f"(progress accumulated in fp32, visible in fp16 once it crossed an ULP boundary)"
    )

    print("\n=== Convergence study: DigitsCNN, fp32 vs fp16 configurations ===\n")
    start = time.time()
    histories = convergence_study(x_train, y_train, epochs=args.epochs)
    elapsed = time.time() - start
    table2 = format_convergence_table(histories)
    print(table2)
    print(f"\n(convergence study wall time: {elapsed:.1f}s, CPU)")

    plot_convergence(histories, args.plot)
    print(f"\nconvergence plot written to {args.plot}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("# Mixed-precision numerical-error and convergence study\n\n")
            f.write("## Numerical error: fp16 vs fp64\n\n")
            f.write(table1 + "\n\n")
            f.write("## Gradient underflow: zero-grad fraction, with/without GradScaler\n\n")
            f.write(table1b + "\n\n")
            f.write("## Synthetic update-underflow demo\n\n")
            f.write(
                "A parameter starting at 1000.0 receiving 2000 identical tiny "
                "updates (~1e-3 each, ~2.0 in total):\n\n"
                f"- naive in-place fp16 update: `{demo['naive_fp16_inplace'][-1]!r}` "
                "(never moved -- each update alone is below fp16's ULP here)\n"
                f"- Adam + fp32 master-weight shadow: "
                f"`{demo['adam_with_master_weights'][-1]!r}` "
                "(progress accumulated in fp32, surfaced once it crossed an ULP "
                "boundary)\n\n"
            )
            f.write("## Convergence: DigitsCNN, fp32 vs fp16 configurations\n\n")
            f.write(table2 + "\n\n")
            f.write(f"![convergence]({args.plot.split('/')[-1]})\n")
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
