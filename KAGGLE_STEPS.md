# Running Gradus's Phase 4 GPU work on Kaggle

Phase 4 (CPU/GPU backend abstraction) is fully implemented and passes
all 98 CPU tests, but the GPU code path itself has **never been run
against a real GPU** -- this dev sandbox has neither a GPU nor CuPy
installed. Everything below is written to be run on Kaggle (free GPU,
no cost), and the results you get back are what actually validates (or
finds bugs in) Phase 4.

This is a copy-paste, cell-by-cell walkthrough. Follow it in order.

## 0. What you're about to run, in one paragraph

Gradus's autograd engine now dispatches every operation to either NumPy
(CPU) or CuPy (GPU) via `Module.to("cuda")` / `Tensor.to("cuda")`.
`tests/test_gpu_parity.py` builds each layer (Linear, Conv2d+BatchNorm,
LayerNorm, Embedding, MultiHeadAttention, a full TransformerLM, Adam)
twice from identical weights -- once on CPU, once moved to GPU -- feeds
both the same input, and checks the forward output and every gradient
match to ~1e-6. `benchmarks/cpu_vs_gpu.py` then times CPU vs. GPU across
tensor size, MLP depth, CNN batch size, and a full Transformer training
step, saving CSVs and plots. You run both, and send me back what they
print (and the CSVs/PNGs) so I can tell you whether Phase 4 actually
works and, if not, exactly which op broke.

## 1. Create the notebook

1. On [kaggle.com](https://kaggle.com), click **Create > New Notebook**.
2. In the right sidebar, under **Settings**:
   - **Accelerator**: GPU T4 x2 (or whatever GPU option is offered --
     any single/double T4 or P100 works, this project doesn't need
     much VRAM).
   - **Internet**: On (needed to `pip install` if CuPy isn't already
     present in the image).

## 2. Get the `gradus` code onto Kaggle

You have the `gradus.zip` I sent you. Since we're not pushing to
GitHub yet, upload it as a Kaggle **Dataset** (this is the standard way
to get your own files into a Kaggle notebook):

1. On Kaggle, go to **Datasets > New Dataset**.
2. Upload `gradus.zip` (no need to unzip it first -- Kaggle unzips
   dataset uploads automatically, or you can unzip it yourself in the
   notebook, either works).
3. Give it any title (e.g. "gradus"), click **Create**.
4. Back in your notebook, click **Add Input** (or **+ Add Data**) in
   the right sidebar, and attach the dataset you just created.
5. It will now appear under `/kaggle/input/<your-dataset-name>/`.

In the notebook's first cell, run (adjust the path to match what the
sidebar shows for your dataset -- click the little copy icon next to
the folder if unsure):

```python
!ls /kaggle/input/
```

Then, assuming the dataset folder is `/kaggle/input/gradus` and it
contains `gradus.zip` directly (rather than the already-unzipped
project -- check with the `ls` above and adjust if Kaggle already
unzipped it for you):

```python
!cp -r /kaggle/input/gradus/* /kaggle/working/ 2>/dev/null || true
%cd /kaggle/working
!unzip -oq gradus.zip -d . 2>/dev/null || true
!ls
```

You should end up in a directory that has `pyproject.toml`, `src/`,
`tests/`, `benchmarks/`, `examples/`, `ROADMAP.md` in it. If the
listing looks different (e.g. everything is one level deeper, inside a
`gradus/` folder), `%cd` into that folder instead -- the rest of these
commands assume you're in the directory containing `pyproject.toml`.

## 3. Confirm the GPU is actually visible

```python
!nvidia-smi
```

This should print a GPU name (e.g. "Tesla T4") and a CUDA version
(e.g. "CUDA Version: 12.2") in the top-right of the table. **Note the
CUDA major version** (12.x vs 11.x) -- you'll need it in the next step.
If this errors or shows no GPU, go back to the notebook Settings and
confirm the Accelerator is actually set to a GPU, then re-run this cell
(a notebook sometimes needs a restart after changing this setting).

## 4. Install Gradus + CuPy

```python
!pip install -e . --quiet
```

Then check whether CuPy is already available (some Kaggle GPU images
ship with it preinstalled for RAPIDS):

```python
try:
    import cupy
    print("cupy already installed:", cupy.__version__)
except ImportError:
    print("cupy not installed yet")
```

If it printed "not installed yet", install the build matching the CUDA
version `nvidia-smi` showed above:

```python
# If nvidia-smi showed CUDA 12.x:
!pip install -q cupy-cuda12x
# If it showed CUDA 11.x instead, use this one instead:
# !pip install -q cupy-cuda11x
```

Then verify:

```python
import cupy
print(cupy.__version__)
print(cupy.cuda.runtime.getDeviceCount(), "GPU(s) visible to CuPy")
x = cupy.array([1.0, 2.0, 3.0])
print((x * 2).sum())  # sanity check: should print 12.0
```

If any of this cell fails, **stop here and send me the full error** --
that's a CuPy/CUDA-version mismatch, not a Gradus bug, and the fix is
almost always installing the other `cupy-cudaXXx` build.

## 5. Run the CPU test suite (should already pass -- sanity check)

```python
!pytest -q
```

Expect something like `98 passed`. If anything fails here, something
went wrong in steps 1-4 (e.g. a partial/corrupted unzip), not in the
GPU work itself -- send me the output before continuing.

## 6. Run the GPU parity tests (the main event)

```python
!pytest tests/test_gpu_parity.py -v
```

This is the test that has never run against real hardware. **Send me
this entire output**, pass or fail:

- If everything **passes**: Phase 4's GPU backend is numerically
  verified, and I'll write that up.
- If something **fails**: the printed `AssertionError` names which
  parameter/output mismatched and by how much (e.g. `grad[weight]: CPU/
  GPU mismatch, max abs diff 3.2e-02 at index (4, 1)`) -- that pinpoints
  which layer's GPU path has a bug. Paste the full traceback, not just
  the last line.

## 7. Run the benchmark suite

Quick smoke test first (~1-2 minutes):

```python
!python benchmarks/cpu_vs_gpu.py --quick
```

Then the full sweep (a few minutes; matmul up to 2048x2048 and CNN
batch size up to 4096 are the slowest parts):

```python
!python benchmarks/cpu_vs_gpu.py
```

This prints a per-sweep speedup table and saves everything to
`benchmarks/results/`:

```python
!ls benchmarks/results/
```

You should see 4 CSVs and 4 PNGs (`matmul.*`, `mlp_depth.*`,
`cnn_batch_size.*`, `transformer_step.*`).

## 8. Send results back to me

Three things, in order of usefulness:

1. **Paste the full text output** of steps 6 and 7 directly into our
   chat (or as a text file) -- this is enough for me to tell what
   happened even before seeing any files.
2. **Download `benchmarks/results/`** (zip it in the notebook with
   `!zip -r results.zip benchmarks/results/` and use Kaggle's file
   download, or just download each file from the notebook's output
   pane) and attach that zip here.
3. If you want, download the whole `/kaggle/working` directory as a
   Kaggle notebook "Output" so I can look at anything else that ran.

Once I have that, I'll fix anything that broke, or move on to writing
up the results if it all passed.

## Phase 7 addendum: mixed precision + gradient checkpointing on GPU

Everything in steps 1-8 above already covers the *mechanics* (upload a
dataset, install, run pytest, run a benchmark script, zip+download
`benchmarks/results/`) -- this section is just the Phase 7-specific
commands to run once you're set up the same way, using the new zip.

**What's new to verify:** Phase 7 added float16 (`Module.half()`) and
gradient checkpointing (`use_checkpoint=True`), both implemented and
tested on CPU already (127 tests passing locally), but two things
genuinely need real GPU hardware to mean anything: (1) does fp16 +
CuPy actually agree numerically with fp16 + NumPy (two new parity
tests), and (2) does fp16 / checkpointing actually deliver the
throughput/memory benefits they're supposed to on real tensor-core
hardware and a real CUDA memory pool (two new benchmark scripts) --
this project has no GPU in its dev sandbox, so neither of those two
questions can be answered without you running this.

1. Upload the new zip as a new version of your existing Kaggle
   dataset (or a fresh dataset -- either works, just re-attach it and
   redo the `cp -r ... && pip install -e .` sequence from steps 2 and
   4 above; remember `/kaggle/working` is wiped on every session
   restart, so re-run that copy step if you see `ModuleNotFoundError`
   or stale code).

2. Re-run the full test suite and the GPU parity tests (now 11 tests,
   up from 9 -- two new ones: `test_fp16_linear_parity_cpu_vs_gpu` and
   `test_transformer_block_checkpoint_parity_cpu_vs_gpu`):

   ```python
   !pytest -q
   !pytest tests/test_gpu_parity.py -v
   ```

   Send me the full output either way. A failure in one of the two new
   tests pinpoints exactly which Phase 7 code path has a real-CuPy bug
   this project's fake-CuPy dry run couldn't have caught.

3. Mixed-precision throughput + memory sweep (fp32 vs fp16 on CUDA --
   there's no CPU column here on purpose, see the script's own
   docstring for why):

   ```python
   !python benchmarks/mixed_precision_gpu.py --quick   # ~1-2 min sanity check
   !python benchmarks/mixed_precision_gpu.py           # full sweep, a few minutes
   ```

4. Gradient-checkpointing memory sweep (the CPU wall-clock half of
   this already ran on my end -- this adds the real GPU memory-pool
   numbers, which is the number that actually matters for
   checkpointing's whole point):

   ```python
   !python benchmarks/checkpoint_memory.py --gpu-memory
   ```

5. Zip and send back the results the same way as before:

   ```python
   !zip -r results_phase7.zip benchmarks/results/
   ```

   Download `results_phase7.zip` and attach it here, along with the
   pasted text output from steps 2-4. Once I have that, I'll fold the
   real numbers into PHASE7_REPORT.md the same way Phase 4-5's numbers
   went into BENCHMARK_REPORT.md.

## Troubleshooting quick-reference

- **`RuntimeError: CuPy is installed but no CUDA GPU is visible`** --
  the Accelerator setting reverted to "None" (Kaggle sometimes does
  this after an idle timeout). Re-check Settings, restart the session.
- **`ModuleNotFoundError: No module named 'gradus'`** -- step 4's
  `pip install -e .` was run from the wrong directory, or wasn't run
  at all. Confirm `pyproject.toml` is in your current directory
  (`!pwd && ls`) before re-running it.
- **CuPy import works but every op raises a CUDA error mentioning
  "no kernel image is available"** -- the installed `cupy-cudaXXx`
  build doesn't match the notebook's actual CUDA version; re-check
  `nvidia-smi`'s CUDA version and reinstall the matching build (step 4).
- **A parity test fails with a tiny (~1e-10) mismatch, not a large
  one** -- that's likely just floating-point summation-order
  differences between NumPy's and CuPy's reduction implementations,
  not a real bug; send it anyway and I'll confirm which case it is from
  the magnitude.
