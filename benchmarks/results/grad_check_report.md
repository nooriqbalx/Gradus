# Gradient-check report

**30/30 checks passed.**

## Phase 1 primitives (+ unfold/embedding_lookup)

| Operation | Max relative error | Max absolute error | Status |
|---|---|---|---|
| add | 1.29e-10 | 3.32e-11 | PASS |
| add (broadcast) | 1.29e-10 | 3.58e-11 | PASS |
| sub | 2.96e-11 | 1.38e-11 | PASS |
| mul | 2.21e-10 | 8.71e-12 | PASS |
| div | 1.70e-10 | 3.66e-10 | PASS |
| neg | 9.03e-11 | 3.58e-11 | PASS |
| pow (p=3) | 6.65e-08 | 1.51e-10 | PASS |
| exp | 1.27e-10 | 2.79e-11 | PASS |
| log | 6.15e-11 | 5.85e-11 | PASS |
| relu | 8.46e-12 | 4.66e-12 | PASS |
| tanh | 2.21e-10 | 5.12e-11 | PASS |
| matmul | 1.65e-10 | 3.06e-11 | PASS |
| matmul (broadcast batch) | 1.26e-09 | 1.11e-10 | PASS |
| sum (all axes) | 1.89e-11 | 3.79e-11 | PASS |
| sum (axis=0) | 3.05e-11 | 7.68e-12 | PASS |
| mean | 1.34e-11 | 2.23e-12 | PASS |
| reshape | 9.03e-11 | 3.58e-11 | PASS |
| transpose | 3.57e-11 | 1.06e-11 | PASS |
| unfold (Conv2d's im2col) | 2.39e-08 | 7.90e-10 | PASS |
| embedding_lookup (incl. repeated index) | 4.30e-10 | 1.07e-10 | PASS |

## Phase 2-3 composed layers

| Operation | Max relative error | Max absolute error | Status |
|---|---|---|---|
| Linear | 1.66e-10 | 3.00e-11 | PASS |
| Conv2d | 7.05e-09 | 2.59e-10 | PASS |
| BatchNorm2d (training) | 8.65e-09 | 5.87e-10 | PASS |
| LayerNorm | 2.21e-09 | 1.17e-10 | PASS |
| GELU | 1.22e-09 | 2.71e-11 | PASS |
| MultiHeadAttention (no mask) | 4.44e-03 | 1.10e-10 | PASS |
| TransformerBlock (attn+FFN+2xLayerNorm) | 8.88e-03 | 4.68e-10 | PASS |
| softmax | 3.36e-10 | 1.21e-11 | PASS |

## Losses

| Operation | Max relative error | Max absolute error | Status |
|---|---|---|---|
| MSELoss | 9.55e-10 | 2.96e-11 | PASS |
| CrossEntropyLoss | 6.02e-10 | 1.69e-11 | PASS |

