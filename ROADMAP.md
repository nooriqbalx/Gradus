# Gradus Roadmap

**Gradus: A CPU/GPU Differentiable Computing Framework**

A from-scratch deep learning framework -- tensor-based reverse-mode
autodiff, standard NN layers through a Transformer block, a shared
CPU (NumPy) / GPU (CuPy) backend, and a rigorous empirical evaluation
of correctness and hardware performance.

**Research question:** How does the design of tensor operations and
automatic differentiation affect computational efficiency across
CPU/GPU execution -- varying tensor size, model depth, batch size,
operation type, and workload (CNN vs. Transformer)?

## Architecture

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

## Phases

- [x] **Phase 0 -- Setup.** Repo scaffold, package structure,
      pyproject.toml, license, .gitignore, pytest + CI from day one.
- [x] **Phase 1 -- Foundations: Autodiff Engine (Weeks 1-2).**
      `Tensor` class, computational graph construction, reverse-mode
      backward pass with broadcasting, parameter/leaf tracking, core
      ops (add, sub, mul, div, neg, pow, exp, log, relu, matmul, sum,
      mean, reshape, transpose), a gradient-check harness (analytical
      vs. finite-difference). 40 tests passing, ruff-clean.
      Parameter/leaf *tracking* proper (a registry other layers query)
      lands with `Module` in Phase 2 -- Phase 1 only needs backward()
      to walk the graph, which it does via each Tensor's own `_prev`.
- [x] **Phase 2 -- Deep Learning Layers (Weeks 3-4).** `Module` base
      class + parameter registry (`parameters()`/`named_parameters()`,
      recursing into submodules and lists of submodules), `Sequential`,
      Linear, Conv2d (im2col via a new `unfold` primitive + matmul),
      BatchNorm2d, LayerNorm, Dropout, Embedding (via a new
      `embedding_lookup` primitive), activations (ReLU, GELU, Tanh),
      losses (MSELoss, CrossEntropyLoss), SGD (with momentum), Adam,
      a minimal `Trainer`/`fit()` API. 82 tests passing, ruff-clean.
      Two new ops.py primitives were needed (`unfold`, `embedding_lookup`
      -- things with no way to express as a composition of Phase 1
      ops); everything else (BatchNorm2d, LayerNorm, softmax,
      cross-entropy, GELU) is a *composition* of already-verified ops,
      correct by the chain rule with no new backward rule to derive.
- [x] **Phase 3 -- Modern Architectures (Weeks 5-6).** Multi-head
      causal self-attention, sinusoidal positional encoding,
      pre-norm TransformerBlock (attention + FFN, each with a residual
      + LayerNorm), `TransformerLM` (a small GPT-style decoder-only
      LM). 93 tests passing, ruff-clean (11 new attention/Transformer
      tests on top of Phase 2's 82).

      **Substitution from the original plan:** CIFAR-10 + PyTorch
      parity checks are deferred to Phase 4-5. This dev sandbox has no
      GPU and no network access to download CIFAR-10, and installing
      PyTorch here (a "free budget" project) wasn't worth the weight
      for a sanity check our own gradient-check harness already
      covers more rigorously. Instead:
        - CNN functional correctness demonstrated on scikit-learn's
          `digits` dataset (1797 images, ships with sklearn, no
          download): `examples/cnn_digits.py`, **97.78% held-out test
          accuracy** (loss 1.56 -> 0.02 over 15 epochs). Architecture
          uses strided Conv2d instead of a separate MaxPool layer
          (the "all-convolutional net" approach), avoiding the need
          for a new max-pooling backward primitive.
        - Transformer functional correctness demonstrated on an
          original synthetic char-level corpus (15 short sentences
          written for this project, shuffled/repeated):
          `examples/toy_transformer.py`, **loss 2.59 -> 0.19** over 20
          epochs (~2000 steps), with the model generating fully
          correct multi-sentence continuations from the training
          distribution.
        - The real CIFAR-10 + GPU run happens in Phase 4-5 on Kaggle
          (free GPU, and CIFAR-10 is a built-in Kaggle dataset --
          both constraints this sandbox has disappear there).
        - PyTorch-parity comparison, if still wanted once torch is
          available (e.g. on Kaggle), is a nice-to-have cross-check,
          not a substitute for the gradient-check harness, which is
          already the more rigorous of the two (it tests the actual
          mathematical derivative, not just "does it land near another
          implementation's answer").

      **Two real methodological findings surfaced by this project's
      own test suite** (documented in code, not just here, since they
      generalize beyond this project -- anyone writing a from-scratch
      autodiff engine will hit both):
        1. Gradient-checking a layer via `output.sum()` silently
           breaks for any mean-subtracting operation (LayerNorm,
           BatchNorm): the output sums to ~0 for *any* input by
           construction, so both the analytical and numerical gradient
           are correctly ~1e-16, but their *relative* difference can
           read as a false failure from pure floating-point noise. Fix:
           reduce via a fixed random linear projection, `(out * W).sum()`,
           instead of a plain sum (see `check_gradient`'s docstring in
           `gradus/utils/grad_check.py`).
        2. Relative error alone is an insufficient pass criterion:
           some gradients are legitimately ~0 for real mathematical
           reasons (e.g. a multi-head attention key-projection bias --
           softmax's per-row shift-invariance makes it provably unable
           to affect the output at all), and relative error is
           meaningless when comparing two near-zero numbers. Fix: pass
           if EITHER relative error < 1e-4 OR absolute error < 1e-6
           (`GradCheckResult.passed`) -- the standard fix documented in
           gradient-check literature (e.g. CS231n's notes).
- [x] **Phase 4 -- GPU Backend (Weeks 7-8).** *Implemented AND verified
      on real GPU hardware.* Backend abstraction dispatching every op
      to NumPy (CPU) or CuPy (GPU) via `get_array_module`/
      `array_module_for_device` (`gradus.backend`), a real
      `.to('cpu')`/`.to('cuda')` on both `Tensor` and `Module` (the
      latter also moving non-Parameter buffers -- BatchNorm2d's
      running stats -- via a new `register_buffer`), CPU/GPU
      correctness parity tests (`tests/test_gpu_parity.py`, 9 tests:
      every layer, an end-to-end TransformerLM, CrossEntropyLoss, and 5
      steps of Adam, each comparing forward output and every gradient
      between a CPU run and a GPU run from identical initial weights),
      and an initial benchmarking harness (`benchmarks/cpu_vs_gpu.py`:
      matmul size, MLP depth, CNN batch size, and full Transformer
      training-step sweeps, each producing a CSV + plot).

      **Verification history:** this project was developed in a
      CPU-only sandbox with no GPU and no way to install CuPy, so every
      GPU-specific line (`embedding_lookup`'s `xp.add.at` scatter-add,
      `one_hot`'s `xp.put_along_axis`, Adam's `xp.sqrt`, ...) was
      reasoned through carefully, defended with a documented fallback
      where the CuPy API was uncertain, and dry-run end-to-end against
      a fake "CuPy" backed by NumPy before ever touching real hardware.
      **All 9 parity tests were then run on a Kaggle notebook (2x
      Tesla T4, CuPy 14.0.1) and passed** -- see `KAGGLE_STEPS.md` for
      the reproduction steps and BENCHMARK_REPORT.md Section 3 for the
      full result. The GPU backend is confirmed correct on real
      hardware, not just dry-run logic.
- [x] **Phase 5 -- Scientific Evaluation (Weeks 9-10).** Full
      gradient-check report (`benchmarks/grad_check.py`, 30/30 checks
      passed across every Phase 1-3 op/layer/loss), convergence curves
      (`benchmarks/convergence_curves.py`, rendered from the Phase 3
      training logs), and the full CPU vs. GPU benchmark suite from
      Phase 4 re-run on real Kaggle GPU hardware (up to 14.7x speedup
      at scale, with a consistent overhead-dominated crossover at small
      configurations across all 4 sweeps) -- all written up in
      **`BENCHMARK_REPORT.md`**, the project's flagship evidence
      artifact.
- [x] **Phase 6 -- Packaging & Engineering Hygiene (Weeks 10-11,
      overlapping).** Full test suite (98 CPU + 9 GPU parity, all
      passing -- see Phase 4-5), `.github/workflows/ci.yml` (lint +
      test on Python 3.10-3.12, GPU tests intentionally excluded since
      GitHub-hosted runners have no CUDA GPU), a verified
      pip-installable package (`python -m build` produces a clean
      wheel and sdist with no deprecation warnings; both installed and
      smoke-tested in a throwaway virtualenv, not just `pip install -e
      .` in the dev environment), API docs (`docs/API.md`, the full
      public surface with signatures and usage snippets) + two
      pre-executed example notebooks (`examples/cnn_digits.ipynb`,
      `examples/toy_transformer.ipynb`, both run end-to-end with real
      output/plots embedded, not just code), and a README with
      minimal-API examples (already in place from earlier phases,
      re-verified here).

      **Two real packaging bugs this phase caught:** (1) `pyproject.toml`
      required `torch>=2.1` in the `dev` extra even though nothing in
      the codebase imports it (it was a Phase 0 placeholder for a
      PyTorch cross-check that was never built) -- moved to its own
      `compare` extra so CI and a normal contributor install don't pay
      torch's install cost for nothing. (2) `.gitignore` excluded
      `benchmarks/results/` as "disposable, regeneratable junk", a
      Phase 0 guess that turned out wrong the moment Phase 5 happened:
      `BENCHMARK_REPORT.md` embeds the actual Kaggle-measured CSVs/PNGs
      in that directory as its evidence, and ignoring it would have
      silently broken every image link in the report on push.
- [x] **Phase 7 -- Stretch (Weeks 11-12, only if ahead of schedule).**
      Both stretch options implemented, not just one: mixed precision
      (`Tensor`/`Module`.half(), `gradus.amp.GradScaler`, fp32 master-
      weight support in SGD/Adam) AND gradient checkpointing
      (`gradus.utils.checkpoint`, wired into `TransformerBlock` as
      `use_checkpoint=True`). Full write-up in **PHASE7_REPORT.md**:
      numerical-error study (fp16 vs fp64 per layer), a direct
      measurement of gradient underflow and GradScaler recovering it,
      a synthetic update-underflow demo proving the master-weight
      fix, an honest convergence study (all four fp32/fp16
      configurations converge similarly for this specific shallow,
      BatchNorm-regularized model -- reported as a real,
      scope-dependent finding rather than forced into a more dramatic
      story), and checkpointing's CPU wall-clock overhead (~1.1-1.7x
      across depths 2-32). **GPU throughput/memory numbers for both
      features are now verified for real on a Kaggle notebook (2x
      Tesla T4)**: fp16 matmul reaches 7.34x speedup at n=2048 and
      uses exactly half the memory of fp32 at every width tested;
      gradient checkpointing saves up to 83.9% peak GPU memory at
      depth=32, growing steadily from 28.4% at depth=2 -- see
      `KAGGLE_STEPS.md`'s Phase 7 addendum for the reproduction steps
      and `PHASE7_REPORT.md` Sections 1.7/2.3 for the full results.
      138 tests total, all passing (127 CPU-only + 11 GPU-only, the
      GPU ones verified for real on that same Kaggle run), ruff-clean.
      Distributed training remains explicitly out of scope.

      **Four real bugs this phase's own work surfaced** (documented
      in code, not just here): (1) `ops._as_tensor` hardcoded
      `dtype=float`, silently upcasting any fp16 Tensor mixed with a
      raw constant back to float64 -- invisible through Phases 1-6
      (everything was float64 anyway), a real bug the moment fp16
      existed; (2) `causal_mask`'s `-1e9` constant overflows to `-inf`
      when cast to float16 (still numerically correct by luck --
      `exp(-inf)` is exactly 0 -- but relying on overflow behavior
      rather than a deliberate choice, and noisy); fixed by using
      `-1e4`, which was always sufficient even at float64 and needs no
      dtype-specific branching. (3) `checkpoint()`'s first
      implementation decided whether to track gradients based on the
      external input's `requires_grad`, with no visibility into
      whether the wrapped function closes over Parameters that need
      gradients too -- silently dropped every parameter gradient for
      a checkpointed layer fed a non-grad-requiring input, caught by
      this project's own fake-GPU dry run before ever reaching Kaggle.
      (4) The first real Kaggle run of the GPU memory benchmarks
      (`mixed_precision_gpu.py`'s memory sweep, `checkpoint_memory.py
      --gpu-memory`) produced impossible negative/non-monotonic peak-
      memory numbers -- traced to this project's pure-Python autograd
      graph forming reference cycles (a backward closure typically
      closes back over its own output Tensor), which plain `del`
      cannot reclaim and nothing was forcing Python's cyclic garbage
      collector to clean up between measurement iterations. Fixed with
      an explicit `gc.collect()` bracketing each iteration in both
      scripts; a bug the fake-CuPy dry run methodology could not have
      caught, since it has no real GPU memory pool to corrupt --
      exactly why the Kaggle round-trip step still matters even after
      thorough local verification.
- [ ] **Phase 8 -- Wrap-up (buffer).** Condense results into CV/SOP
      language, short demo video, final repo polish.

## Validation criteria (Phase 5)

Three distinct forms of correctness, evaluated separately:

1. **Numerical correctness** -- gradient checking: analytical gradient
   vs. finite-difference approximation vs. PyTorch autograd, per op,
   reported as a relative-error table.
2. **Functional correctness** -- models actually train: loss
   decreases, accuracy increases. Demonstrated in Phase 3 on
   scikit-learn's `digits` (CNN, 97.78% held-out accuracy) and an
   original synthetic char-level corpus (Transformer, loss 2.59 ->
   0.19); the CIFAR-10 run moves to Phase 4-5 on Kaggle (see Phase 3's
   substitution note above).
3. **Performance characterization** -- runtime, memory, throughput,
   and CPU/GPU scaling across tensor size, model depth, batch size,
   and operation type.

Accuracy parity with PyTorch is *not* a goal in itself -- Gradus is
not trying to beat or match PyTorch, only to demonstrate a correct,
benchmarked implementation of the same underlying mechanics.

## Explicitly out of scope

- Distributed / data-parallel training (cut to protect the ~10-12 week
  solo timeline; the CPU/GPU backend-abstraction story already carries
  the systems-engineering signal this project needs).
- Claiming novelty as a framework -- the contribution is the scope,
  implementation rigor, and experimental evaluation, not a new idea.
  (Comparable educational frameworks: micrograd, tinygrad, Needle.)
