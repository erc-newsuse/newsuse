from collections.abc import Mapping, MutableMapping
from functools import singledispatch
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ("dotget", "dotset", "dotdel")


@singledispatch
def _get(obj, name: str) -> Any:
    return getattr(obj, name)


@_get.register
def _(obj: Mapping, name: str) -> Any:
    return obj[name]


@singledispatch
def _set(obj, name: str, value: Any) -> None:
    setattr(obj, name, value)


@_set.register
def _(obj: Mapping, name: str, value: Any) -> None:
    if not isinstance(obj, MutableMapping):
        errmsg = f"cannot set items on '{type(obj)}' instances"
        raise TypeError(errmsg)
    obj[name] = value


@singledispatch
def _del(obj, name: str) -> None:
    delattr(obj, name)


@_del.register
def _(obj: Mapping, name: str) -> None:
    if not isinstance(obj, MutableMapping):
        errmsg = f"cannot delete items on '{type(obj)}' instances"
        raise TypeError(errmsg)
    del obj[name]


def dotget(obj, dotpath: str) -> Any:
    """Extract (nested) object attribute based on dot-path specification.

    For mapping item access is used instead of attribute access.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).

    Examples
    --------
    >>> from pathlib import Path
    >>> path = Path(".").absolute()
    >>> dotget(path, "parent.parent") == path.parent.parent
    True
    """
    for name in dotpath.split("."):
        obj = _get(obj, name)
    return obj


def dotset(obj: Any, dotpath: str, value: Any) -> None:
    """Set (nested) object attribute based on dot-path specification.

    For mapping item access is used instead of attribute access.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.
    value
        New attribute value.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).

    Examples
    --------
    >>> obj = {"a": {"b": 1}}
    >>> dotset(obj, "a.b", 2)
    >>> dotget(obj, "a.b")
    2
    """
    try:
        dotpath, name = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath)
    except ValueError:
        name = dotpath
    else:
        _set(obj, name, value)


def dotdel(obj: Any, dotpath: str) -> None:
    """Delete (nested) object attribute based on dot-path specification.

    For mapping item access is used instead of attribute access.

    Parameters
    ----------
    obj
        Arbitrary object.
    dotpath
        Dot-path of the form ``"name1.name2..."``.

    Raises
    ------
    AttributeError
        If the provided dot-path contains non-existent attribute name(s).

    Examples
    --------
    >>> obj = {"a": {"b": 1}}
    >>> dotdel(obj, "a.b")
    >>> obj["a"]
    {}
    """
    try:
        dotpath, name = dotpath.rsplit(".", 1)
        obj = dotget(obj, dotpath)
    except ValueError:
        name = dotpath
    else:
        _del(obj, name)


def dotimport(dotpath: str) -> Any:
    """Import module and extract object based on dot-path specification.

    Parameters
    ----------
    dotpath
        Dot-path of the form ``"module1.module2...modulen:obj.attr1.attr2..."``.
        The part on the left of ``:`` defines the module import specification
        and the part on the right is passed to :func:`dotget` and used
        to extract an object or its attribute from the imported module.

    Raises
    ------
    ImportError
        If the import specification is incorrect.
    AttributeError
        If the object/attribute specification is incorrect.
    ValueError
        If the dot-path is incorrect and has more than one ``:`` separators.

    Examples
    --------
    >>> dotimport("typing:Any") is Any
    True
    >>> dotimport("typing:Any:__name__")
    Traceback (most recent call last):
    ValueError: ...
    """
    if dotpath.count(":") > 1:
        errmsg = f"dot-path '{dotpath}' has more than one ':' separator"
        raise ValueError(errmsg)
    module_spec, *obj_spec = dotpath.split(":")
    module = import_module(module_spec)
    if not obj_spec:
        return module
    return dotget(module, obj_spec.pop())


def dotsource(dotpath: str) -> Path | None:
    """Get package or module path a from dot-path.

    For packages ``__path__[0]`` is returned, for module ``__file__``,
    and for other objects an attempt at returning ``__file__`` of the module
    specified by the object ``__module__``  attribute is made.

    Examples
    --------
    >>> path = dotsource("typing:Any")
    >>> path.name
    'typing.py'
    """
    obj = dotimport(dotpath)
    if isinstance(obj, ModuleType):
        source = obj.__path__[0] if obj.__name__ == obj.__package__ else obj.__file__
    else:
        try:
            source = dotimport(obj.__module__).__file__
        except AttributeError:
            return None
    return Path(str(source))
