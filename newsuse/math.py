from collections.abc import Mapping
from functools import singledispatch
from math import log
from typing import Any

import numpy as np

__all__ = ("entropy",)


@singledispatch
def entropy(
    probs,
    axis: int | tuple[int, ...] | None = None,
    *,
    normalized: bool = False,
    base: float = 2,
    tol: float | None = None,
) -> np.ndarray:
    """Calculate Shannon entropy of the distribution defined by ``probs``.

    Value in ``probs`` can be non-normalized, but cannot be negative.
    Use ``normalized`` to get normalized entropy in :math:`[0, 1]`.

    Examples
    --------
    >>> float(entropy([1, 0]))
    0.0
    >>> probs = [[2, 1], [1, 3]]
    >>> float(entropy(probs))
    1.84237...
    >>> np.round(entropy(probs, axis=0), 4)
    array([0.9183, 0.8113])
    >>> np.round(entropy(probs, axis=1), 4)
    array([0.9183, 0.8113])
    """
    probs = np.asarray(probs)
    if axis is not None and not isinstance(axis, tuple):
        axis = (axis,)
    norm = probs.sum(axis=axis)
    if axis:
        norm = np.expand_dims(norm, axis)
    probs = probs / norm

    tol = float(np.finfo(probs.dtype).eps if tol is None else tol)
    probs[np.abs(probs) <= tol] = 0

    if (probs < 0).any():
        errmsg = "probabilities must be non-negative"
        raise ValueError(errmsg)

    n_symbols = sum(probs.shape[a] for a in axis) if axis else probs.size

    probs[probs == 0] = 1
    H = np.abs(-(probs * np.emath.logn(base, probs)).sum(axis=axis))

    if normalized:
        H /= log(n_symbols, base)

    return H


@entropy.register
def _(probs: Mapping, *args: Any, **kwargs: Any) -> np.ndarray:
    return entropy(list(probs.values()), *args, **kwargs)
