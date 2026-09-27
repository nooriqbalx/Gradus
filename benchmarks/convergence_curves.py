"""
Convergence curves (Phase 5) -- the "loss decreases, accuracy
increases" leg of ROADMAP.md's validation criteria, rendered as plots
rather than left as the raw epoch-by-epoch text in
examples/results/*_output.txt.

Parses the saved training logs from examples/cnn_digits.py and
examples/toy_transformer.py (rather than hardcoding the numbers here,
so this stays correct if either example is ever re-run) and plots:
    - CNN: train loss AND train accuracy vs. epoch (twin y-axes)
    - Transformer: train loss vs. epoch (log y-axis -- loss spans
      2.59 -> 0.19, more than an order of magnitude, so linear scale
      compresses the early, fastest-improving epochs)

Usage:
    python benchmarks/convergence_curves.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLES_RESULTS = Path(__file__).parent.parent / "examples" / "results"
OUT_DIR = Path(__file__).parent / "results"


def _parse_cnn_log(text: str) -> tuple[list[int], list[float], list[float]]:
    # "epoch  3/15  train_loss=0.2906  train_acc=0.9624"
    pattern = re.compile(r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)")
    epochs, losses, accs = [], [], []
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        accs.append(float(m.group(3)))
    return epochs, losses, accs


def _parse_transformer_log(text: str) -> tuple[list[int], list[float]]:
    # "epoch  4/20  loss=0.4060"
    pattern = re.compile(r"epoch\s+(\d+)/\d+\s+loss=([\d.]+)")
    epochs, losses = [], []
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))
    return epochs, losses


def plot_cnn_convergence() -> None:
    text = (EXAMPLES_RESULTS / "cnn_digits_output.txt").read_text()
    epochs, losses, accs = _parse_cnn_log(text)
    if not epochs:
        raise ValueError("No epoch lines parsed from cnn_digits_output.txt -- log format changed?")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, losses, marker="o", color="tab:blue", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, marker="s", color="tab:orange", label="train accuracy")
    ax2.set_ylabel("train accuracy", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("CNN on scikit-learn digits (examples/cnn_digits.py)")
    ax1.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    out_path = OUT_DIR / "convergence_cnn.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"saved -> {out_path}  "
        f"({len(epochs)} epochs, final loss={losses[-1]}, final acc={accs[-1]})"
    )


def plot_transformer_convergence() -> None:
    text = (EXAMPLES_RESULTS / "toy_transformer_output.txt").read_text()
    epochs, losses = _parse_transformer_log(text)
    if not epochs:
        raise ValueError(
            "No epoch lines parsed from toy_transformer_output.txt -- log format changed?"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, losses, marker="o", color="tab:green")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        "TransformerLM on an original synthetic corpus\n(examples/toy_transformer.py)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    out_path = OUT_DIR / "convergence_transformer.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}  ({len(epochs)} logged epochs, final loss={losses[-1]})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_cnn_convergence()
    plot_transformer_convergence()


if __name__ == "__main__":
    main()
