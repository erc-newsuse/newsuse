import os
from types import SimpleNamespace
from typing import Any

from newsuse.dotpath import dotdel, dotget, dotset

PathLike = str | os.PathLike


class Namespace(SimpleNamespace):
    """Namespace.

    A simple namespace type supporting both attribute and item access.

    Examples
    --------
    >>> ns = Namespace(a=1)
    >>> ns.a == ns["a"] == 1
    True
    """

    def __getitem__(self, name: str) -> Any:
        if "." in name:
            return dotget(self, name)
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if "." in name:
            dotset(self, name, value)
        else:
            setattr(self, name, value)

    def __delitem__(self, name: str) -> None:
        if "." in name:
            dotdel(self, name)
        else:
            delattr(self, name)
