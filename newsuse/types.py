import os
from types import SimpleNamespace
from typing import Any

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
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def __delitem__(self, name: str) -> None:
        delattr(self, name)
