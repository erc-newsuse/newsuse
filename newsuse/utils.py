from collections.abc import Callable, Mapping
from functools import singledispatch
from inspect import Parameter, Signature, _empty, get_annotations, signature
from typing import Any

import joblib
import pydantic

__all__ = ("inthash", "get_signature", "match_signature", "validate_call")


def inthash(x: Any, shift: int = 0, **kwargs: Any) -> int:
    """Make deterministic integer hash from an object and add ``shift``.

    ``**kwargs`` are passed to :func:`joblib.hash`.

    Examples
    --------
    >>> tup = (1, 2)
    >>> inthash(tup) == inthash(tup)
    True
    >>> inthash(tup) == inthash(tup, 2)
    False
    """
    return int(joblib.hash(x, **kwargs), base=16) + shift


@singledispatch
def get_signature(func) -> Signature:
    """Get signature of a callable object.

    If ``func`` is a type, then signature is derived from class annotations.
    """
    return signature(func)


@get_signature.register
def _(func: type) -> Signature:
    params = []
    for name, ann in get_annotations(func).items():
        param = Parameter(
            name=name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=getattr(func, name, _empty),
            annotation=ann,
        )
        params.append(param)

    return Signature(params, return_annotation=func)


def match_signature(
    sig: Signature | Callable[..., Any], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Match arguments to the signature of a callable."""
    if not isinstance(sig, Signature):
        sig = get_signature(sig)
    posargs = sig.bind(*args).arguments if args else {}
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return {**posargs, **kwargs}


def validate_call(
    func: Callable[..., Any] | None = None,
    /,
    *,
    config: Mapping | None = None,
    **kwargs: Any,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Validate call arguments using :func:`pydantic.validate_call`."""
    config = pydantic.ConfigDict({"arbitrary_types_allowed": True, **(config or {})})
    return validate_call(func, config=config, **kwargs)
