"""
Gradient checkpointing: compute/memory trade-off (Phase 7).

Two measurements, matching gradus.utils.checkpoint's documented trade
(an extra recompute, in exchange for not holding a sublayer's
activations across the forward/backward gap):

  1. CPU wall-clock overhead -- runs on this dev sandbox, no GPU
     needed. A stack of TransformerBlocks, forward+backward, with
     use_checkpoint=True vs False, at a few stack depths. Expect
     checkpointed to be slower (it recomputes each block's attention +
     FFN sublayer during backward) -- this measures by how much.

  2. Real GPU peak memory -- needs CuPy + a CUDA device (run on
     Kaggle, like every other GPU-only script in this project). Same
     model, same forward+backward, but measuring
     cupy.get_default_memory_pool().used_bytes() before and after
     rather than wall time. Expect checkpointed to use LESS peak
     memory, and the gap to widen as the model gets deeper (more
     activations NOT retained at once).

Usage:
    python benchmarks/checkpoint_memory.py                # CPU timing only
    python benchmarks/checkpoint_memory.py --gpu-memory    # + GPU memory sweep (needs CUDA)
    python benchmarks/checkpoint_memory.py --quick
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
from gradus.nn.module import Module
from gradus.nn.transformer import TransformerBlock

RESULTS_DIR = Path(__file__).parent / "results"
GPU_AVAILABLE = cupy_available()


class BlockStack(Module):
    """`num_layers` TransformerBlocks in sequence -- deliberately not
    the full TransformerLM (no embedding/vocab head), so this isolates
    exactly the part checkpoint() actually wraps."""

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, use_checkpoint: bool
    ):
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def time_call(fn, warmup: int = 3, repeats: int = 15) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(times), "mean_s": statistics.mean(times)}


# ----------------------------------------------------------------------
# CPU wall-clock overhead
# ----------------------------------------------------------------------


def sweep_cpu_overhead(
    depths: list[int], batch_size=4, seq_len=16, embed_dim=64, num_heads=4, ff_dim=128
):
    rows = []
    for depth in depths:
        for use_checkpoint in (False, True):
            np.random.seed(0)
            model = BlockStack(embed_dim, num_heads, ff_dim, depth, use_checkpoint)
            x_data = np.random.default_rng(1).standard_normal((batch_size, seq_len, embed_dim))

            def step():
                x = Tensor(x_data.copy(), requires_grad=True)
                model.zero_grad()
                model(x).sum().backward()

            stats = time_call(step)
            rows.append(
                {
                    "depth": depth,
                    "use_checkpoint": use_checkpoint,
                    **stats,
                }
            )
            print(
                f"  depth={depth:<3} use_checkpoint={str(use_checkpoint):<5} "
                f"median={stats['median_s'] * 1e3:8.3f} ms"
            )
    return pd.DataFrame(rows)


def format_overhead_table(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="depth", columns="use_checkpoint", values="median_s")
    pivot.columns = ["no_checkpoint_s", "checkpoint_s"]
    pivot["overhead_ratio"] = pivot["checkpoint_s"] / pivot["no_checkpoint_s"]
    return pivot.to_string(float_format=lambda x: f"{x:.6f}")


# ----------------------------------------------------------------------
# GPU peak memory (real CuPy memory pool, not an estimate)
# ----------------------------------------------------------------------


def sweep_gpu_memory(
    depths: list[int], batch_size=8, seq_len=64, embed_dim=128, num_heads=4, ff_dim=512
):
    """Real GPU peak memory, checkpointed vs not, isolated per iteration.

    Same fix as mixed_precision_gpu.py's sweep_memory (see its docstring
    for the full explanation): gradus's pure-Python autograd graph forms
    reference cycles (a Tensor's backward closure typically closes back
    over the output Tensor itself), which plain `del` cannot reclaim.
    Without an explicit `gc.collect()`, GPU memory left over from a
    PRIOR (depth, use_checkpoint) iteration could still be alive when
    the NEXT iteration takes its baseline reading -- which is exactly
    what produced the first real-Kaggle run's impossible numbers (a
    negative -74 MB "peak" at depth=8, and depth=4 showing less memory
    than depth=2). This does not affect gradient correctness -- only
    which now-unreachable Python/CuPy objects are still resident when
    each measurement is taken.
    """
    if not GPU_AVAILABLE:
        raise SystemExit(
            "No CUDA GPU visible to CuPy -- this measurement needs real GPU memory-"
            "pool accounting, there is no meaningful CPU proxy for it. Run on Kaggle."
        )
    import cupy

    rows = []
    pool = cupy.get_default_memory_pool()
    for depth in depths:
        for use_checkpoint in (False, True):
            gc.collect()  # reclaim any cyclic garbage from the prior iteration
            pool.free_all_blocks()
            cupy.cuda.Stream.null.synchronize()

            np.random.seed(0)
            model = BlockStack(embed_dim, num_heads, ff_dim, depth, use_checkpoint)
            model.to("cuda")
            x_data = np.random.default_rng(1).standard_normal((batch_size, seq_len, embed_dim))
            x = Tensor(x_data).to("cuda")
            x.requires_grad = True

            baseline = pool.used_bytes()
            out = model(x)
            out.sum().backward()
            cupy.cuda.Stream.null.synchronize()
            peak = pool.used_bytes()
            delta = peak - baseline
            if delta < 0:
                print(
                    f"  WARNING: negative memory delta ({delta} bytes) at "
                    f"depth={depth} use_checkpoint={use_checkpoint} even after "
                    "gc.collect() -- treat this row's number as unreliable."
                )

            rows.append(
                {
                    "depth": depth,
                    "use_checkpoint": use_checkpoint,
                    "peak_bytes_over_baseline": delta,
                }
            )
            print(
                f"  depth={depth:<3} use_checkpoint={str(use_checkpoint):<5} "
                f"peak={delta / 1e6:8.2f} MB"
            )
            del model, x, out
            gc.collect()  # reclaim this iteration's cycles before the next baseline
            pool.free_all_blocks()
    return pd.DataFrame(rows)


def format_memory_table(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(
        index="depth", columns="use_checkpoint", values="peak_bytes_over_baseline"
    )
    pivot.columns = ["no_checkpoint_bytes", "checkpoint_bytes"]
    pivot["memory_saved_fraction"] = 1 - pivot["checkpoint_bytes"] / pivot["no_checkpoint_bytes"]
    return pivot.to_string(float_format=lambda x: f"{x:.4f}")


def _plot(df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for use_checkpoint, group in df.groupby("use_checkpoint"):
        group = group.sort_values("depth")
        label = "checkpointed" if use_checkpoint else "not checkpointed"
        ax.plot(group["depth"], group[value_col], marker="o", label=label)
    ax.set_xlabel("num_layers (TransformerBlocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-memory", action="store_true", help="Also run the GPU memory sweep.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    depths = [2, 4] if args.quick else [2, 4, 8, 16, 32]

    print("=== CPU wall-clock overhead: checkpointed vs not ===\n")
    cpu_df = sweep_cpu_overhead(depths)
    cpu_df.to_csv(RESULTS_DIR / "checkpoint_cpu_overhead.csv", index=False)
    print()
    print(format_overhead_table(cpu_df))
    _plot(
        cpu_df,
        "median_s",
        "median latency (s)",
        "Checkpointing: CPU wall-clock overhead",
        RESULTS_DIR / "checkpoint_cpu_overhead.png",
    )

    if args.gpu_memory:
        print("\n=== GPU peak memory: checkpointed vs not ===\n")
        print(f"GPU available: {GPU_AVAILABLE}")
        gpu_df = sweep_gpu_memory(depths)
        gpu_df.to_csv(RESULTS_DIR / "checkpoint_gpu_memory.csv", index=False)
        print()
        print(format_memory_table(gpu_df))
        _plot(
            gpu_df,
            "peak_bytes_over_baseline",
            "peak CUDA memory (bytes)",
            "Checkpointing: real GPU peak memory",
            RESULTS_DIR / "checkpoint_gpu_memory.png",
        )
    else:
        print(
            "\n(skipping GPU memory sweep -- pass --gpu-memory on a CUDA machine, "
            "e.g. a Kaggle GPU notebook, for the real memory-savings numbers.)"
        )

    print("\nDone. See benchmarks/results/ for CSVs and plots.")


if __name__ == "__main__":
    main()
