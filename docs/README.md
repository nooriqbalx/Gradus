# Docs

- **[API.md](API.md)** -- the public API reference: every class and
  function in `gradus`, with signatures and minimal usage snippets,
  organized by module (`Tensor`, `ops`, `nn`, `optim`, `Trainer`,
  `backend`, `utils`).

For design rationale (*why* things are built the way they are, not
just what's callable), read the module docstrings in the source
directly -- `gradus/ops.py`, `gradus/tensor.py`, and
`gradus/nn/module.py` each explain their own design choices in detail,
and API.md deliberately doesn't repeat that. For the four-layer
architecture, see `../README.md`'s Design section. For evaluation
results, see `../BENCHMARK_REPORT.md`.
