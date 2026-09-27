"""
CPU vs GPU benchmark sweeps -- the full write-up with a per-op
relative-error table alongside this lives in BENCHMARK_REPORT.md.

Four sweeps, matching this project's validation criteria (tensor size,
batch size, model depth, CNN vs. Transformer workload):

    1. matmul          -- square matmul forward+backward, size N x N,
                           N in SIZES. The purest "is the GPU actually
                           being used" signal: matmul is the one op
                           where a real speedup is basically guaranteed
                           once N is large enough to amortize kernel-
                           launch/transfer overhead.
    2. mlp_depth       -- a stack of `depth` Linear(512,512)+ReLU
                           blocks, fixed batch size, forward+backward,
                           depth in DEPTHS. Tests how the CPU/GPU gap
                           changes as the graph gets deeper (more, but
                           individually smaller, kernel launches).
    3. cnn_batch_size  -- the examples/cnn_digits.py architecture
                           (Conv2d/BatchNorm2d/ReLU stack), forward+
                           backward, batch size in BATCH_SIZES. Conv2d
                           here is im2col+matmul (see ops.unfold), so
                           this also indirectly benchmarks unfold's
                           Python-level kernel-offset loop -- expect
                           this to be the sweep where a naive from-
                           scratch conv shows its constant-factor cost
                           most clearly, GPU or not.
    4. transformer_step -- one full training step (forward + backward +
                           Adam.step()) of a small TransformerLM sized
                           like examples/toy_transformer.py, at a couple
                           of (seq_len, batch_size) combinations.

Usage:
    python benchmarks/cpu_vs_gpu.py                  # all sweeps
    python benchmarks/cpu_vs_gpu.py --sweep matmul    # just one
    python benchmarks/cpu_vs_gpu.py --quick           # smaller sizes, fast smoke test

Output: benchmarks/results/<sweep>.csv (raw timings, one row per
config x device) and benchmarks/results/<sweep>.png (median latency
vs. the swept parameter, log-log, one line per device -- log-log
because both the parameter range and the latency range typically span
more than one order of magnitude here). Also prints a summary table
(median latency + speedup) to stdout as it goes.

Honesty note: this script's GPU path is written against the same
Phase 4 backend abstraction as the rest of Phase 4 and has NOT been
run against a real GPU as of writing (no GPU in this dev sandbox) --
run it on Kaggle for the real numbers. On a CPU-only machine it still
runs, skipping the GPU column and only reporting CPU numbers, so it
can be smoke-tested here first (matplotlib/pandas plumbing, CSV
schema, timing methodology) before trusting the GPU half of it.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gradus import Tensor
from gradus.backend import cupy_available
from gradus.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    ReLU,
    Sequential,
)
from gradus.nn.transformer import TransformerLM
from gradus.optim import Adam

RESULTS_DIR = Path(__file__).parent / "results"
GPU_AVAILABLE = cupy_available()
DEVICES = ["cpu", "cuda"] if GPU_AVAILABLE else ["cpu"]


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def _sync(device: str) -> None:
    """CUDA kernel launches are asynchronous from the host's point of
    view -- without this, time.perf_counter() around a GPU call would
    measure only queueing time, not actual execution, making the GPU
    look implausibly fast. cupy.cuda.Stream.null.synchronize() blocks
    until every queued kernel on the default stream has actually
    finished, which is what makes the CPU/GPU timings in this script
    comparable at all."""
    if device == "cuda":
        import cupy

        cupy.cuda.Stream.null.synchronize()


def time_call(fn, device: str, warmup: int = 3, repeats: int = 10) -> dict:
    """Runs fn() `warmup` times (untimed -- absorbs first-call costs
    like CUDA context/kernel compilation and NumPy/CuPy lazy imports),
    then `repeats` timed calls. Returns median/mean/min/max seconds;
    median is what the summary table and plots use, since a handful of
    OS-scheduling outliers on a shared machine (e.g. a Kaggle
    container) shouldn't swing the headline number."""
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)

    return {
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "max_s": max(times),
    }


# ----------------------------------------------------------------------
# Sweep 1: matmul forward+backward, square N x N
# ----------------------------------------------------------------------


def sweep_matmul(sizes: list[int]) -> pd.DataFrame:
    rows = []
    for n in sizes:
        for device in DEVICES:
            rng = np.random.default_rng(0)
            a_data = rng.standard_normal((n, n))
            b_data = rng.standard_normal((n, n))
            a = Tensor(a_data, requires_grad=True)
            b = Tensor(b_data, requires_grad=True)
            if device == "cuda":
                a, b = a.to("cuda"), b.to("cuda")
                a.requires_grad = b.requires_grad = True

            def step():
                a.zero_grad()
                b.zero_grad()
                (a @ b).sum().backward()

            stats = time_call(step, device)
            rows.append({"sweep": "matmul", "n": n, "device": device, **stats})
            print(f"  matmul  n={n:<6} device={device:<4} median={stats['median_s'] * 1e3:8.3f} ms")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sweep 2: MLP depth, fixed batch size
# ----------------------------------------------------------------------


def sweep_mlp_depth(depths: list[int], batch_size: int = 128, width: int = 512) -> pd.DataFrame:
    rows = []
    for depth in depths:
        for device in DEVICES:
            np.random.seed(0)
            layers = []
            for _ in range(depth):
                layers += [Linear(width, width), ReLU()]
            model = Sequential(*layers)
            if device == "cuda":
                model.to("cuda")

            x_data = np.random.default_rng(1).standard_normal((batch_size, width))
            x = Tensor(x_data)
            if device == "cuda":
                x = x.to("cuda")

            def step():
                model.zero_grad()
                model(x).sum().backward()

            stats = time_call(step, device)
            rows.append({"sweep": "mlp_depth", "depth": depth, "device": device, **stats})
            print(
                f"  mlp_depth  depth={depth:<4} device={device:<4} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sweep 3: CNN forward+backward, batch size sweep
# (same architecture as examples/cnn_digits.py, but at a configurable
#  spatial size/batch so it scales the way an actual dataset would)
# ----------------------------------------------------------------------


def _build_cnn() -> Sequential:
    return Sequential(
        Conv2d(1, 8, 3, padding=1),
        BatchNorm2d(8),
        ReLU(),
        Conv2d(8, 16, 3, stride=2, padding=1),
        BatchNorm2d(16),
        ReLU(),
    )


def sweep_cnn_batch_size(batch_sizes: list[int], image_size: int = 28) -> pd.DataFrame:
    rows = []
    for batch_size in batch_sizes:
        for device in DEVICES:
            np.random.seed(0)
            model = _build_cnn()
            if device == "cuda":
                model.to("cuda")

            x_data = np.random.default_rng(2).standard_normal(
                (batch_size, 1, image_size, image_size)
            )
            x = Tensor(x_data)
            if device == "cuda":
                x = x.to("cuda")

            def step():
                model.zero_grad()
                model(x).sum().backward()

            stats = time_call(step, device)
            row = {"sweep": "cnn_batch_size", "batch_size": batch_size, "device": device, **stats}
            rows.append(row)
            print(
                f"  cnn_batch_size  batch={batch_size:<5} device={device:<4} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sweep 4: full TransformerLM training step (forward + backward + Adam)
# ----------------------------------------------------------------------


def sweep_transformer_step(configs: list[tuple[int, int]]) -> pd.DataFrame:
    """configs: list of (seq_len, batch_size) pairs."""
    rows = []
    vocab_size, embed_dim, num_heads, ff_dim, num_layers = 64, 64, 4, 128, 2
    for seq_len, batch_size in configs:
        for device in DEVICES:
            np.random.seed(0)
            model = TransformerLM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                max_len=max(seq_len, 16),
            )
            if device == "cuda":
                model.to("cuda")
            optimizer = Adam(model.parameters(), lr=1e-3)
            loss_fn = CrossEntropyLoss()

            rng = np.random.default_rng(3)
            token_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len))
            targets = rng.integers(0, vocab_size, size=(batch_size * seq_len,))

            def step():
                optimizer.zero_grad()
                logits = model(token_ids)  # raw ndarray input, see nn/transformer.py
                loss = loss_fn(logits.reshape(-1, vocab_size), targets)
                loss.backward()
                optimizer.step()

            stats = time_call(step, device, warmup=2, repeats=5)
            rows.append(
                {
                    "sweep": "transformer_step",
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "device": device,
                    **stats,
                }
            )
            print(
                f"  transformer_step  seq={seq_len:<4} batch={batch_size:<4} device={device:<4} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


_TIMING_COLS = {"median_s", "mean_s", "min_s", "max_s"}


def _print_speedup_table(df: pd.DataFrame, param_cols: list[str]) -> None:
    if not GPU_AVAILABLE:
        print("  (GPU unavailable in this environment -- no speedup column; "
              "run on a Kaggle GPU notebook for the full comparison.)")
        return
    pivot = df.pivot_table(index=param_cols, columns="device", values="median_s")
    if "cpu" in pivot.columns and "cuda" in pivot.columns:
        pivot["speedup_cpu_over_gpu"] = pivot["cpu"] / pivot["cuda"]
    print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))


def _plot(df: pd.DataFrame, x_col: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for device, group in df.groupby("device"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group["median_s"] * 1e3, marker="o", label=device.upper())
    ax.set_xlabel(x_col)
    ax.set_ylabel("median latency (ms)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        choices=["matmul", "mlp_depth", "cnn_batch_size", "transformer_step", "all"],
        default="all",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Smaller sizes/fewer repeats, for a fast smoke test."
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU available (CuPy + CUDA device visible): {GPU_AVAILABLE}")
    if not GPU_AVAILABLE:
        print(
            "Running CPU-only. This is expected on the dev sandbox this project "
            "was built in -- run this same script on a free Kaggle GPU notebook "
            "(pip install cupy-cuda12x, matching the CUDA version nvidia-smi "
            "reports) to get the CUDA column too."
        )
    print()

    sizes = [64, 128, 256, 512] if args.quick else [64, 128, 256, 512, 1024, 2048]
    depths = [2, 4] if args.quick else [2, 4, 8, 16, 32]
    batch_sizes = [16, 64] if args.quick else [16, 64, 256, 1024, 4096]
    transformer_configs = [(32, 16)] if args.quick else [(32, 16), (64, 32), (128, 64), (256, 32)]

    sweeps = {
        "matmul": (lambda: sweep_matmul(sizes), "n"),
        "mlp_depth": (lambda: sweep_mlp_depth(depths), "depth"),
        "cnn_batch_size": (lambda: sweep_cnn_batch_size(batch_sizes), "batch_size"),
        "transformer_step": (lambda: sweep_transformer_step(transformer_configs), "seq_len"),
    }

    to_run = sweeps if args.sweep == "all" else {args.sweep: sweeps[args.sweep]}

    for name, (fn, x_col) in to_run.items():
        print(f"=== {name} ===")
        df = fn()
        csv_path = RESULTS_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  saved raw results -> {csv_path}")
        # Exact-match against the known timing-column names, not a
        # substring filter -- "batch_size" contains "_s" (from
        # "_size"), so a `like="_s"` filter here wrongly swallowed it
        # as if it were a timing column, collapsing cnn_batch_size's
        # printed summary into one row instead of one per batch size.
        # This only ever affected this printed table, never the CSV/
        # PNG outputs, which use the untouched DataFrame directly.
        param_cols = [c for c in df.columns if c not in {"sweep", "device"} | _TIMING_COLS]
        _print_speedup_table(df, param_cols)
        _plot(df, x_col, name, RESULTS_DIR / f"{name}.png")
        print()

    print("Done. See benchmarks/results/ for CSVs and plots.")


if __name__ == "__main__":
    main()
