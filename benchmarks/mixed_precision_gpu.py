"""
Mixed-precision GPU throughput + memory benchmark (Phase 7 -- the real-
hardware half of ROADMAP.md's "throughput/memory/convergence/
numerical-error study vs. FP32"; the CPU-only convergence/numerical-
error half lives in benchmarks/mixed_precision.py).

This ONLY compares fp32 vs. fp16 ON CUDA -- unlike cpu_vs_gpu.py's
CPU-vs-GPU sweeps, there is no interesting CPU throughput story for
fp16 here: NumPy has no hardware-accelerated float16 arithmetic (it's
typically emulated via float32 internally, sometimes SLOWER than
float32 on CPU), so the entire point of fp16 -- feeding real tensor-
core hardware smaller operands -- only shows up on a real GPU. This
project has no GPU in its dev sandbox (see ROADMAP.md's Phase 4
verification history), so this script's CUDA path is, like
cpu_vs_gpu.py's, unverified against real hardware until run on Kaggle.

Four throughput sweeps, matching cpu_vs_gpu.py's shape exactly (same
architectures, same sizes) so the two reports are directly comparable:
matmul, mlp_depth, cnn_batch_size, transformer_step -- each run once at
fp32 and once at fp16 on CUDA.

One memory sweep: peak CUDA memory-pool usage (via CuPy's own
get_default_memory_pool(), not a proxy or an estimate) for a forward+
backward pass at each of a few model sizes, fp32 vs fp16 -- the direct
evidence for fp16's "half the memory" claim, measured, not assumed.

Usage (Kaggle, GPU + Internet on, after `pip install -e .`):
    python benchmarks/mixed_precision_gpu.py                  # everything
    python benchmarks/mixed_precision_gpu.py --sweep matmul
    python benchmarks/mixed_precision_gpu.py --quick           # smaller sizes
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gradus import Tensor
from gradus.backend import cupy_available
from gradus.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Linear, ReLU, Sequential
from gradus.nn.transformer import TransformerLM
from gradus.optim import Adam

RESULTS_DIR = Path(__file__).parent / "results"
GPU_AVAILABLE = cupy_available()
PRECISIONS = ["fp32", "fp16"]


def _require_gpu() -> None:
    if not GPU_AVAILABLE:
        raise SystemExit(
            "This script needs a real CUDA GPU (via CuPy) -- there is no CPU "
            "fallback here, unlike cpu_vs_gpu.py, because fp16 has no interesting "
            "throughput/memory story on CPU (see this module's docstring). Run it "
            "on a Kaggle GPU notebook -- see KAGGLE_STEPS.md."
        )


def _cast(tensor_or_module, precision: str):
    return tensor_or_module.half() if precision == "fp16" else tensor_or_module.float()


def _sync() -> None:
    import cupy

    cupy.cuda.Stream.null.synchronize()


def time_call(fn, warmup: int = 3, repeats: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "max_s": max(times),
    }


# ----------------------------------------------------------------------
# Throughput sweeps (same architectures/sizes as cpu_vs_gpu.py)
# ----------------------------------------------------------------------


def sweep_matmul(sizes: list[int]) -> pd.DataFrame:
    rows = []
    for n in sizes:
        for precision in PRECISIONS:
            rng = np.random.default_rng(0)
            a = _cast(Tensor(rng.standard_normal((n, n)), requires_grad=True).to("cuda"), precision)
            b = _cast(Tensor(rng.standard_normal((n, n)), requires_grad=True).to("cuda"), precision)
            a.requires_grad = b.requires_grad = True

            def step():
                a.zero_grad()
                b.zero_grad()
                (a @ b).sum().backward()

            stats = time_call(step)
            rows.append({"sweep": "matmul", "n": n, "precision": precision, **stats})
            ms = stats["median_s"] * 1e3
            print(f"  matmul  n={n:<6} precision={precision:<4} median={ms:8.3f} ms")
    return pd.DataFrame(rows)


def sweep_mlp_depth(depths: list[int], batch_size: int = 128, width: int = 512) -> pd.DataFrame:
    rows = []
    for depth in depths:
        for precision in PRECISIONS:
            np.random.seed(0)
            layers = []
            for _ in range(depth):
                layers += [Linear(width, width), ReLU()]
            model = _cast(Sequential(*layers).to("cuda"), precision)

            x = _cast(
                Tensor(np.random.default_rng(1).standard_normal((batch_size, width))).to("cuda"),
                precision,
            )

            def step():
                model.zero_grad()
                model(x).sum().backward()

            stats = time_call(step)
            rows.append({"sweep": "mlp_depth", "depth": depth, "precision": precision, **stats})
            print(
                f"  mlp_depth  depth={depth:<4} precision={precision:<4} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


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
        for precision in PRECISIONS:
            np.random.seed(0)
            model = _cast(_build_cnn().to("cuda"), precision)
            x_data = np.random.default_rng(2).standard_normal(
                (batch_size, 1, image_size, image_size)
            )
            x = _cast(Tensor(x_data).to("cuda"), precision)

            def step():
                model.zero_grad()
                model(x).sum().backward()

            stats = time_call(step)
            rows.append(
                {
                    "sweep": "cnn_batch_size",
                    "batch_size": batch_size,
                    "precision": precision,
                    **stats,
                }
            )
            print(
                f"  cnn_batch_size  batch={batch_size:<5} precision={precision:<4} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


def sweep_transformer_step(configs: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    vocab_size, embed_dim, num_heads, ff_dim, num_layers = 64, 64, 4, 128, 2
    for seq_len, batch_size in configs:
        for precision in PRECISIONS:
            np.random.seed(0)
            model = TransformerLM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                max_len=max(seq_len, 16),
            )
            model.to("cuda")
            _cast(model, precision)
            optimizer = Adam(model.parameters(), lr=1e-3)
            loss_fn = CrossEntropyLoss()

            rng = np.random.default_rng(3)
            token_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len))
            targets = rng.integers(0, vocab_size, size=(batch_size * seq_len,))

            def step():
                optimizer.zero_grad()
                logits = model(token_ids)
                loss = loss_fn(logits.reshape(-1, vocab_size), targets)
                loss.backward()
                optimizer.step()

            stats = time_call(step, warmup=2, repeats=5)
            rows.append(
                {
                    "sweep": "transformer_step",
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "precision": precision,
                    **stats,
                }
            )
            print(
                f"  transformer_step  seq={seq_len:<4} batch={batch_size:<4} "
                f"precision={precision:<4} median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Memory sweep: real CUDA memory-pool usage, fp32 vs fp16
# ----------------------------------------------------------------------


def sweep_memory(widths: list[int], depth: int = 8, batch_size: int = 256) -> pd.DataFrame:
    """Real CUDA memory-pool usage, fp32 vs fp16, isolated per iteration.

    Gradus's autograd graph is pure Python: a Tensor's backward closure
    commonly closes back over the very output Tensor it belongs to (to
    write into its `.grad`), which is a reference cycle. Plain
    refcounting `del` never reclaims a cycle -- only Python's cyclic
    garbage collector does, and nothing was forcing that to run between
    iterations. Left alone, that means GPU memory from a PRIOR
    iteration (or, when this runs as part of `--sweep all`, a prior
    sweep entirely) can still be alive, uncollected, when the NEXT
    iteration takes its "baseline" reading -- inflating that baseline
    unpredictably, and then vanishing mid-measurement whenever the
    interpreter's automatic GC threshold happens to trip. That produces
    exactly the impossible negative / non-monotonic deltas this bug
    caused on the first real Kaggle run (e.g. -893 MB "peak" at
    width=4096) -- a benchmark-measurement bug, not an autograd
    correctness bug (nothing here affects gradient values, only which
    Python objects are still reachable). The fix is an explicit
    `gc.collect()` immediately before AND after each iteration, so every
    iteration's baseline/peak reflects only that iteration's own
    allocations.
    """
    import cupy

    rows = []
    for width in widths:
        for precision in PRECISIONS:
            gc.collect()  # reclaim any cyclic garbage from a prior iteration/sweep
            pool = cupy.get_default_memory_pool()
            pool.free_all_blocks()
            cupy.cuda.Stream.null.synchronize()

            np.random.seed(0)
            layers = []
            for _ in range(depth):
                layers += [Linear(width, width), ReLU()]
            model = _cast(Sequential(*layers).to("cuda"), precision)
            x = _cast(
                Tensor(np.random.default_rng(1).standard_normal((batch_size, width))).to("cuda"),
                precision,
            )

            baseline = pool.used_bytes()
            out = model(x)
            out.sum().backward()
            _sync()
            peak = pool.used_bytes()
            delta = peak - baseline
            if delta < 0:
                print(
                    f"  WARNING: negative memory delta ({delta} bytes) at "
                    f"width={width} precision={precision} even after gc.collect() "
                    "-- a deeper reference-cycle issue than this fix accounts for; "
                    "treat this row's number as unreliable."
                )

            rows.append(
                {
                    "sweep": "memory",
                    "width": width,
                    "depth": depth,
                    "precision": precision,
                    "peak_bytes_over_baseline": delta,
                }
            )
            print(
                f"  memory  width={width:<5} precision={precision:<4} "
                f"peak={delta / 1e6:8.2f} MB"
            )
            del model, x, out
            gc.collect()  # reclaim this iteration's cycles before the next baseline
            pool.free_all_blocks()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

_TIMING_COLS = {"median_s", "mean_s", "min_s", "max_s"}


def _print_speedup_table(
    df: pd.DataFrame, param_cols: list[str], value_col: str = "median_s"
) -> None:
    pivot = df.pivot_table(index=param_cols, columns="precision", values=value_col)
    if "fp32" in pivot.columns and "fp16" in pivot.columns:
        pivot["fp16_speedup"] = pivot["fp32"] / pivot["fp16"]
    print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))


def _plot(
    df: pd.DataFrame, x_col: str, y_col: str, ylabel: str, title: str, out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for precision, group in df.groupby("precision"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker="o", label=precision)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


def main() -> None:
    _require_gpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        choices=["matmul", "mlp_depth", "cnn_batch_size", "transformer_step", "memory", "all"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [64, 128, 256, 512] if args.quick else [64, 128, 256, 512, 1024, 2048]
    depths = [2, 4] if args.quick else [2, 4, 8, 16, 32]
    batch_sizes = [16, 64] if args.quick else [16, 64, 256, 1024, 4096]
    transformer_configs = [(32, 16)] if args.quick else [(32, 16), (64, 32), (128, 64), (256, 32)]
    widths = [256, 512] if args.quick else [256, 512, 1024, 2048, 4096]

    sweeps = {
        "matmul": (lambda: sweep_matmul(sizes), "n", "median_s", "median latency (ms)"),
        "mlp_depth": (lambda: sweep_mlp_depth(depths), "depth", "median_s", "median latency (ms)"),
        "cnn_batch_size": (
            lambda: sweep_cnn_batch_size(batch_sizes),
            "batch_size",
            "median_s",
            "median latency (ms)",
        ),
        "transformer_step": (
            lambda: sweep_transformer_step(transformer_configs),
            "seq_len",
            "median_s",
            "median latency (ms)",
        ),
        "memory": (
            lambda: sweep_memory(widths),
            "width",
            "peak_bytes_over_baseline",
            "peak CUDA memory (bytes)",
        ),
    }

    to_run = sweeps if args.sweep == "all" else {args.sweep: sweeps[args.sweep]}

    for name, (fn, x_col, y_col, ylabel) in to_run.items():
        print(f"=== {name} ===")
        df = fn()
        csv_path = RESULTS_DIR / f"mixed_precision_gpu_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  saved raw results -> {csv_path}")
        param_cols = [
            c for c in df.columns if c not in {"sweep", "precision"} | _TIMING_COLS | {y_col}
        ]
        _print_speedup_table(df, param_cols, value_col=y_col)
        if y_col == "median_s":
            df = df.assign(**{y_col: df[y_col] * 1e3})  # seconds -> ms for the plot
        _plot(df, x_col, y_col, ylabel, name, RESULTS_DIR / f"mixed_precision_gpu_{name}.png")
        print()

    print("Done. See benchmarks/results/ for CSVs and plots.")


if __name__ == "__main__":
    main()
