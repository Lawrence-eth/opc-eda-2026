"""Minimal torch stand-in for the packaged solver executable.

Why this exists: the official evaluation wraps our executable with
op_wrapper.py, which spawns the binary once per test case and measures the
whole call (spawn + imports + solve + I/O) as the case's runtime. Bundling
real torch costs seconds of import time per case, which would forfeit the
runtime-factor floor the submission strategy depends on. The wrapper delivers
every input as plain JSON lists, and my_optimizer.py only touches a tiny
tensor surface on them, so a list-backed stand-in is sufficient.

This is NOT a general tensor library. It implements exactly the operations
my_optimizer.py performs on its inputs: dim/shape/len/iteration, int/tuple/
slice indexing (including the (slice, int) column pattern), item/tolist/max/
sum, and scalar comparisons (scalars are float subclasses, so comparisons and
float()/int() work natively). Equivalence with real torch is enforced by
packaging/equivalence_test.py, which requires bit-identical positions on all
100 validation cases.
"""


class _Scalar(float):
    """Float that also supports .item() like a 0-d tensor."""

    def item(self):
        return float(self)


def _wrap(v):
    return Tensor(v) if isinstance(v, list) else _Scalar(v)


class Tensor:
    __slots__ = ("_d",)

    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data._d
        self._d = data

    # --- structure -------------------------------------------------------
    def dim(self):
        d, x = 0, self._d
        while isinstance(x, list):
            d += 1
            x = x[0] if x else None
        return d

    @property
    def shape(self):
        s, x = [], self._d
        while isinstance(x, list):
            s.append(len(x))
            x = x[0] if x else None
        return tuple(s)

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return (_wrap(v) for v in self._d)

    def __bool__(self):
        x = self._d
        while isinstance(x, list):
            if len(x) != 1:
                raise ValueError(
                    "The truth value of a multi-element Tensor is ambiguous")
            x = x[0]
        return bool(x)

    def __repr__(self):
        return f"Tensor({self._d!r})"

    # --- indexing --------------------------------------------------------
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self._get_tuple(self._d, idx)
        if isinstance(idx, slice):
            return Tensor(self._d[idx])
        return _wrap(self._d[idx])

    @classmethod
    def _get_tuple(cls, data, idx):
        if not idx:
            return _wrap(data)
        k, rest = idx[0], idx[1:]
        if isinstance(k, slice):
            out = []
            for row in data[k]:
                sub = cls._get_tuple(row, rest)
                out.append(sub._d if isinstance(sub, Tensor) else float(sub))
            return Tensor(out)
        return cls._get_tuple(data[k], rest)

    # --- reductions / conversions ----------------------------------------
    def _flat(self):
        out = []
        stack = [self._d]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(reversed(x))
            else:
                out.append(x)
        return out

    def max(self):
        return _Scalar(max(self._flat()))

    def min(self):
        return _Scalar(min(self._flat()))

    def sum(self):
        return _Scalar(sum(self._flat()))

    def item(self):
        flat = self._flat()
        if len(flat) != 1:
            raise ValueError("only one-element Tensors can use .item()")
        return float(flat[0])

    def numel(self):
        return len(self._flat())

    def tolist(self):
        import copy
        return copy.deepcopy(self._d)

    # --- elementwise comparisons (1-D masks, used by _n_soft etc.) --------
    def _cmp(self, other, op):
        if isinstance(other, Tensor):
            other = other._d
        if isinstance(self._d, list):
            return Tensor([1.0 if op(v, other) else 0.0 for v in self._d])
        return op(self._d, other)

    def __ne__(self, other):
        return self._cmp(other, lambda a, b: a != b)

    def __eq__(self, other):
        return self._cmp(other, lambda a, b: a == b)

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    # torch tensors are hashable by identity; some code may use them in sets
    def __hash__(self):
        return id(self)


# --- module-level factory functions (only used by dormant code paths) -----

# dtype placeholders (only ever passed through as inert keyword arguments by
# dormant code paths; my_optimizer's live paths never use dtypes). NOTE: do
# not alias `float` here — rebinding the builtin inside this module breaks
# every internal `float(...)` call.
float32 = "float32"
float64 = "float64"


def tensor(data, dtype=None):
    if isinstance(data, Tensor):
        return Tensor(data.tolist())
    if isinstance(data, (list, tuple)):
        return Tensor([list(r) if isinstance(r, (list, tuple)) else r
                       for r in data])
    return Tensor(data)


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)

    def build(dims):
        if len(dims) == 1:
            return [0.0] * dims[0]
        return [build(dims[1:]) for _ in range(dims[0])]

    return Tensor(build(list(shape)))


def full(shape, fill_value, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)

    def build(dims):
        if len(dims) == 1:
            return [fill_value] * dims[0]
        return [build(dims[1:]) for _ in range(dims[0])]

    return Tensor(build(list(shape)))
