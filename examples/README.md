# Examples

Runnable scripts and notebooks demonstrating the API end-to-end, plus
the saved output from an actual run of each.

- `cnn_digits.py` / `cnn_digits.ipynb` -- a small CNN (Conv2d,
  BatchNorm2d, ReLU, Linear, CrossEntropyLoss, Adam, Trainer) on
  scikit-learn's `digits` dataset. **97.78% held-out test accuracy.**
  (CIFAR-10 was the original plan, substituted for `digits` to keep
  scope achievable without a GPU in the dev sandbox and without
  weakening the correctness claim -- the point is verifying the CNN's
  forward/backward mechanics, not chasing a harder dataset's accuracy
  ceiling.)
- `toy_transformer.py` / `toy_transformer.ipynb` -- a small decoder-only
  `TransformerLM` trained as a char-level language model on an original
  synthetic corpus. **Loss 2.59 -> 0.19** over 20 epochs, generating
  coherent multi-sentence continuations.
- `results/` -- saved stdout logs from the `.py` scripts (used by
  `../benchmarks/convergence_curves.py` to render the convergence
  plots in `../BENCHMARK_REPORT.md`).

Run either script directly (`python examples/cnn_digits.py`, after
`pip install -e .`), or open the matching `.ipynb` in Jupyter -- both
notebooks are already executed and checked into the repo with real
output and plots, so they're readable without re-running anything, and
reproducible (same seeds) if you do.
