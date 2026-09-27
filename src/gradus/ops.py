"""
Differentiable tensor operations.

Each function here implements one operation as a forward pass plus a
local backward rule (the vector-Jacobian product for that op), wires
the result into the computational graph (Tensor._prev / _backward via
_make_output), and -- at the bottom of this module -- is attached onto
Tensor as the matching dunder method (__add__, __mul__, ...) or named
method (.sum(), .relu(), ...).

Why the split from tensor.py: tensor.py owns the graph *mechanics*
(how backward() walks the graph); this module owns what each
operation actually *computes* and how it's differentiated. Keeping
them separate means adding a new op never touches the graph-walking
code, and the graph engine never needs to know what "matmul" means.
It also means operator-overload wiring lives right next to the op
that implements it, rather than scattered across tensor.py.

Implemented here:
    Phase 1 -- add, sub, mul, div, neg, pow, exp, log, relu, matmul,
               sum, mean, reshape, transpose.
    Phase 2 -- tanh (for GELU), unfold (im2col, for Conv2d),
               embedding_lookup (for Embedding).

Deliberately NOT implemented as new primitives here: softmax,
log-softmax, cross-entropy, BatchNorm, LayerNorm, GELU, and
scaled-dot-product attention. Every one of those is expressible as a
composition of the ops already above, so it is correct by the chain
rule automatically -- no new backward rule to derive or verify. They
live in gradus.nn / gradus.nn.functional instead (Phase 2-3). Keeping
this file's primitive count minimal is a deliberate design choice:
each new primitive here is something with no other way to express it
(a sliding-window read, an index-based lookup), not a convenience
wrapper.
"""

from __future__ import annotations

import numpy as np

from gradus._grad_mode import is_grad_enabled
from gradus.backend import array_module_for_device, get_array_module
from gradus.tensor import Tensor


def _resolve_device(*args) -> str:
    """The device a new Tensor built from these op arguments should
    live on: whichever device the first actual Tensor among them is
    on (mixing two different *Tensor* devices in one op is not
    supported -- see the note on Tensor.device -- but a raw Python/
    NumPy constant mixed with a Tensor is expected and handled by
    _as_tensor creating that constant on the Tensor's device)."""
    for x in args:
        if isinstance(x, Tensor):
            return x.device
    return "cpu"


def _resolve_dtype(*args):
    """The dtype a converted raw operand should take: whichever real
    Tensor among these args carries a dtype already (mixing an fp16
    Tensor with a raw Python/NumPy constant should produce an fp16
    constant, not silently upcast the whole op to float64 -- see the
    Phase 7 note on _as_tensor below), falling back to a bare NumPy/
    CuPy array's own dtype if no Tensor is present, and None (meaning
    "let _as_tensor pick its float64 default") only when nothing here
    carries dtype information at all (e.g. two plain Python floats)."""
    for x in args:
        if isinstance(x, Tensor):
            return x.data.dtype
    for x in args:
        if hasattr(x, "dtype"):
            return x.dtype
    return None


def _as_tensor(x, device: str = "cpu", dtype=None) -> Tensor:
    """Wrap a raw (non-Tensor) operand as a Tensor on `device`.

    Phase 1-6 this unconditionally forced dtype=float (float64) here.
    That was a real bug once Phase 7 added fp16 support: every binary
    op mixing a Tensor with a raw Python scalar or NumPy/CuPy array
    (e.g. `some_fp16_tensor + 1.0`, or LayerNorm's `x - mean` where
    `mean` was built as a plain array) silently upcast the *entire*
    downstream computation to float64, defeating the whole point of
    casting a model to fp16 for memory/throughput, with no error or
    warning to say so. Now the caller (each binary op below, via
    _resolve_dtype) passes the sibling Tensor's own dtype explicitly;
    only when there's truly no dtype to inherit (dtype=None, x has no
    .dtype of its own -- i.e. a bare Python scalar/list with no other
    Tensor operand at all) does this fall back to float64, matching
    the original default behavior exactly for plain-old fp64 code.
    """
    if isinstance(x, Tensor):
        return x
    xp = array_module_for_device(device)
    if dtype is None:
        dtype = getattr(x, "dtype", None)
        if dtype is None:
            dtype = float
    return Tensor(xp.asarray(x, dtype=dtype), device=device)


def _make_output(data, parents: tuple, backward_fn, device: str) -> Tensor:
    """Build an output Tensor, wiring it into the graph only if at
    least one parent requires grad -- nothing to backprop through
    otherwise, so we skip retaining graph references for it. Also
    gated on is_grad_enabled() (Phase 7): inside a `with no_grad():`
    block, every op's output has requires_grad=False regardless of its
    parents, so nothing anywhere inside that block retains _prev/
    _backward -- see gradus._grad_mode and gradus.utils.checkpoint."""
    requires_grad = is_grad_enabled() and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=requires_grad, device=device)
    if requires_grad:
        out._prev = parents
        out._backward = backward_fn
    return out


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` (shaped like a broadcasted output) back down to
    `shape` (the original, pre-broadcast operand shape), by summing
    over the dimensions NumPy broadcasting introduced or stretched."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# ----------------------------------------------------------------------
# Elementwise binary ops
# ----------------------------------------------------------------------

def add(a, b) -> Tensor:
    device = _resolve_device(a, b)
    dtype = _resolve_dtype(a, b)
    a, b = _as_tensor(a, device, dtype), _as_tensor(b, device, dtype)
    out_data = a.data + b.data

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad, a.data.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad, b.data.shape)
            b.grad = g if b.grad is None else b.grad + g

    out = _make_output(out_data, (a, b), _backward, a.device)
    out._op = "add"
    return out


def sub(a, b) -> Tensor:
    device = _resolve_device(a, b)
    dtype = _resolve_dtype(a, b)
    a, b = _as_tensor(a, device, dtype), _as_tensor(b, device, dtype)
    out_data = a.data - b.data

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad, a.data.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(-out.grad, b.data.shape)
            b.grad = g if b.grad is None else b.grad + g

    out = _make_output(out_data, (a, b), _backward, a.device)
    out._op = "sub"
    return out


def mul(a, b) -> Tensor:
    device = _resolve_device(a, b)
    dtype = _resolve_dtype(a, b)
    a, b = _as_tensor(a, device, dtype), _as_tensor(b, device, dtype)
    out_data = a.data * b.data

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad * b.data, a.data.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad * a.data, b.data.shape)
            b.grad = g if b.grad is None else b.grad + g

    out = _make_output(out_data, (a, b), _backward, a.device)
    out._op = "mul"
    return out


def div(a, b) -> Tensor:
    device = _resolve_device(a, b)
    dtype = _resolve_dtype(a, b)
    a, b = _as_tensor(a, device, dtype), _as_tensor(b, device, dtype)
    out_data = a.data / b.data

    def _backward():
        if a.requires_grad:
            g = unbroadcast(out.grad / b.data, a.data.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(-out.grad * a.data / (b.data ** 2), b.data.shape)
            b.grad = g if b.grad is None else b.grad + g

    out = _make_output(out_data, (a, b), _backward, a.device)
    out._op = "div"
    return out


def neg(a) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    out_data = -a.data

    def _backward():
        if a.requires_grad:
            g = -out.grad
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "neg"
    return out


def pow_(a, p: float) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    out_data = a.data ** p

    def _backward():
        if a.requires_grad:
            g = out.grad * p * (a.data ** (p - 1))
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "pow"
    return out


# ----------------------------------------------------------------------
# Elementwise unary ops
# ----------------------------------------------------------------------

def exp(a) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    out_data = xp.exp(a.data)

    def _backward():
        if a.requires_grad:
            g = out.grad * out.data
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "exp"
    return out


def log(a) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    out_data = xp.log(a.data)

    def _backward():
        if a.requires_grad:
            g = out.grad / a.data
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "log"
    return out


def relu(a) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    out_data = xp.maximum(a.data, 0.0)

    def _backward():
        if a.requires_grad:
            g = out.grad * (a.data > 0)
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "relu"
    return out


def tanh(a) -> Tensor:
    """Added in Phase 2: needed for the tanh-approximation of GELU used
    by the Transformer feed-forward block (Phase 3)."""
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    out_data = xp.tanh(a.data)

    def _backward():
        if a.requires_grad:
            g = out.grad * (1.0 - out.data ** 2)
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "tanh"
    return out


# ----------------------------------------------------------------------
# Matrix multiplication
# ----------------------------------------------------------------------

def matmul(a, b) -> Tensor:
    device = _resolve_device(a, b)
    dtype = _resolve_dtype(a, b)
    a, b = _as_tensor(a, device, dtype), _as_tensor(b, device, dtype)
    xp = get_array_module(a.data)
    out_data = a.data @ b.data

    def _backward():
        if a.requires_grad:
            g = out.grad @ xp.swapaxes(b.data, -1, -2)
            g = unbroadcast(g, a.data.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = xp.swapaxes(a.data, -1, -2) @ out.grad
            g = unbroadcast(g, b.data.shape)
            b.grad = g if b.grad is None else b.grad + g

    out = _make_output(out_data, (a, b), _backward, a.device)
    out._op = "matmul"
    return out


# ----------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------

def sum_(a, axis=None, keepdims: bool = False) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    out_data = a.data.sum(axis=axis, keepdims=keepdims)

    def _backward():
        if not a.requires_grad:
            return
        g = out.grad
        if axis is not None and not keepdims:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes_norm = sorted(ax % a.data.ndim for ax in axes)
            for ax in axes_norm:
                g = xp.expand_dims(g, ax)
        g = xp.broadcast_to(g, a.data.shape)
        a.grad = g.copy() if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "sum"
    return out


def mean(a, axis=None, keepdims: bool = False) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    if axis is None:
        count = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = 1
        for ax in axes:
            count *= a.data.shape[ax]
    return mul(sum_(a, axis=axis, keepdims=keepdims), 1.0 / count)


# ----------------------------------------------------------------------
# Shape ops
# ----------------------------------------------------------------------

def reshape(a, *shape) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    out_data = a.data.reshape(shape)
    orig_shape = a.data.shape

    def _backward():
        if a.requires_grad:
            g = out.grad.reshape(orig_shape)
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "reshape"
    return out


def transpose(a, *axes) -> Tensor:
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    if len(axes) == 0:
        axes = tuple(reversed(range(a.data.ndim)))
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    out_data = xp.transpose(a.data, axes)
    # argsort here operates on a plain Python tuple of ints (the axis
    # permutation itself), not on device-resident array data, so this
    # one stays np.argsort regardless of backend.
    inv_axes = tuple(int(i) for i in np.argsort(axes))

    def _backward():
        if a.requires_grad:
            g = xp.transpose(out.grad, inv_axes)
            a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "transpose"
    return out


# ----------------------------------------------------------------------
# Structural ops (Phase 2)
#
# Everything above composes freely under the chain rule -- a layer
# built purely from add/mul/matmul/etc. is automatically correct if
# each of those primitives is correct (that's the whole point of
# autodiff). The two ops below exist because "extract a sliding
# window" (convolution) and "select rows by integer index"
# (embedding) are NOT expressible as a composition of the ops above --
# they need their own forward/backward rule, so each gets its own
# gradient check just like a Phase 1 primitive.
# ----------------------------------------------------------------------

def unfold(a, kernel_size: tuple, stride: int = 1, padding: int = 0) -> Tensor:
    """im2col: extract sliding local kh x kw patches from a (N, C, H, W)
    tensor, laid out as (N, C*kh*kw, out_h*out_w) so that a convolution
    becomes a single matmul against a (out_channels, C*kh*kw) weight
    matrix (see gradus.nn.layers.Conv2d).

    The backward pass (col2im) is the exact linear adjoint of this
    extraction: it scatter-adds each patch's gradient back to the
    input positions it was read from, correctly accumulating overlaps
    between neighboring windows (the standard fold/unfold behavior in
    every deep learning framework that supports convolution).

    Note on performance: this loops over the kh*kw kernel offsets in
    Python (vectorized across N and C at each step), rather than using
    a single as-strided view. That keeps the forward/backward
    unmistakably correct and easy to verify by gradient checking, at
    some speed cost versus a fully vectorized im2col -- an acceptable
    trade for a CPU correctness/training demo; performance is the
    explicit subject of Phase 4-5, not this op.
    """
    a = _as_tensor(a, _resolve_device(a))
    xp = get_array_module(a.data)
    kh, kw = kernel_size
    N, C, H, W = a.data.shape

    if padding > 0:
        padded = xp.pad(a.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        padded = a.data
    Hp, Wp = padded.shape[2], padded.shape[3]
    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1

    def _row_slice(i, out_len, stride_):
        return slice(i, i + stride_ * out_len, stride_)

    patches = []
    for c in range(C):
        for i in range(kh):
            i_slice = _row_slice(i, out_h, stride)
            for j in range(kw):
                j_slice = _row_slice(j, out_w, stride)
                patch = padded[:, c, i_slice, j_slice]
                patches.append(patch.reshape(N, -1))  # (N, out_h*out_w)
    out_data = xp.stack(patches, axis=1)  # (N, C*kh*kw, out_h*out_w)

    def _backward():
        if not a.requires_grad:
            return
        grad_padded = xp.zeros_like(padded)
        idx = 0
        for c in range(C):
            for i in range(kh):
                i_slice = _row_slice(i, out_h, stride)
                for j in range(kw):
                    j_slice = _row_slice(j, out_w, stride)
                    grad_patch = out.grad[:, idx, :].reshape(N, out_h, out_w)
                    grad_padded[:, c, i_slice, j_slice] += grad_patch
                    idx += 1
        if padding > 0:
            g = grad_padded[:, :, padding : Hp - padding, padding : Wp - padding]
        else:
            g = grad_padded
        a.grad = g if a.grad is None else a.grad + g

    out = _make_output(out_data, (a,), _backward, a.device)
    out._op = "unfold"
    return out


def _scatter_add(xp, target, indices, values) -> None:
    """target[indices] += values, in place, with correct accumulation
    when an index repeats (plain `target[indices] += values` under
    NumPy fancy-indexing semantics silently keeps only one of several
    writes to the same index -- xp.add.at exists specifically to fix
    that, which is why embedding_lookup's backward needs it rather
    than ordinary indexed assignment).

    GPU note: CuPy's `add.at` support has varied across versions; if
    it's unavailable, this falls back to `cupyx.scatter_add`, CuPy's
    own older name for the same operation. Both paths are exercised
    and verified correct on real CUDA hardware (`tests/test_gpu_parity.py`,
    the Embedding parity test specifically covers a repeated index to
    exercise this scatter-add case).
    """
    try:
        xp.add.at(target, indices, values)
    except (AttributeError, TypeError):
        import cupyx

        cupyx.scatter_add(target, indices, values)


def embedding_lookup(weight, indices) -> Tensor:
    """Select rows of `weight` (shape (vocab_size, embed_dim)) by
    integer `indices` (any shape) -- i.e. weight[indices], with the
    correct scatter-add backward: a token that appears at multiple
    positions accumulates gradient from every position it appeared in,
    exactly like a shared/tied weight would in any framework.
    """
    weight = _as_tensor(weight, _resolve_device(weight))
    xp = get_array_module(weight.data)
    indices = xp.asarray(indices)  # match weight's backend, not the caller's
    out_data = weight.data[indices]

    def _backward():
        if weight.requires_grad:
            g = xp.zeros_like(weight.data)
            _scatter_add(xp, g, indices, out.grad)
            weight.grad = g if weight.grad is None else weight.grad + g

    out = _make_output(out_data, (weight,), _backward, weight.device)
    out._op = "embedding_lookup"
    return out


# ----------------------------------------------------------------------
# Wire these onto Tensor as operator overloads / methods. Done here
# (not in tensor.py) so each operator's implementation lives with the
# op that defines it -- see this module's docstring. gradus/__init__.py
# imports this module (for its side effects) right after importing
# Tensor, so the full operator API is available on any `from gradus
# import Tensor`.
# ----------------------------------------------------------------------

Tensor.__add__ = lambda self, other: add(self, other)
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__sub__ = lambda self, other: sub(self, other)
Tensor.__rsub__ = lambda self, other: sub(other, self)
Tensor.__mul__ = lambda self, other: mul(self, other)
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__truediv__ = lambda self, other: div(self, other)
Tensor.__rtruediv__ = lambda self, other: div(other, self)
Tensor.__neg__ = lambda self: neg(self)
Tensor.__pow__ = lambda self, p: pow_(self, p)
Tensor.__matmul__ = lambda self, other: matmul(self, other)

Tensor.sum = lambda self, axis=None, keepdims=False: sum_(self, axis=axis, keepdims=keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis=axis, keepdims=keepdims)
Tensor.reshape = lambda self, *shape: reshape(self, *shape)
Tensor.transpose = lambda self, *axes: transpose(self, *axes)
Tensor.exp = lambda self: exp(self)
Tensor.log = lambda self: log(self)
Tensor.relu = lambda self: relu(self)
Tensor.tanh = lambda self: tanh(self)
Tensor.T = property(lambda self: transpose(self))
