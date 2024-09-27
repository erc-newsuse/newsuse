from collections.abc import Mapping
from pathlib import Path
from typing import Any

from confection import Config as _Config

from newsuse.dotpath import dotdel, dotget, dotset

from .paths import Paths

__all__ = ("Config", "Paths")


class Config(_Config):
    """Config with predefined defaults based on :class:`confection.Config`.

    Examples
    --------
    >>> config = Config({"a": 1}, __defaults__=False)
    >>> config
    Config({'a': 1})

    Attribute access also works.

    >>> config.a
    1

    Nested mappings are returned as config objects supporting attribute access.

    >>> config = Config({"obj": {"a": 1}}, __defaults__=False)
    >>> config.obj.a
    1
    """

    def __init__(self, *args: Any, __defaults__: bool = True, **kwargs: Any) -> None:
        if __defaults__:
            path = Path(__file__).parent / "base.cfg"
            base = _Config().from_disk(path)
        else:
            base = _Config()
        conf = base.merge(_Config(*args, **kwargs))
        super().__init__(conf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __getitem__(self, key: str) -> Any:
        obj = super().__getitem__(key)
        if isinstance(obj, Mapping) and not isinstance(obj, type(self)):
            obj = Config(obj, __defaults__=False)
            self[key] = obj
        return obj

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            cn = self.__class__.__name__
            errmsg = f"'{cn}' object has no attribute '{name}'"
            raise AttributeError(errmsg) from exc
