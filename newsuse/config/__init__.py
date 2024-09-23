from collections.abc import Mapping
from pathlib import Path
from typing import Any

from confection import Config as _Config

from .paths import Paths

__all__ = ("Config", "Paths")


class Config(_Config):
    """Config with predefined defaults based on :class:`confection.Config`.

    Examples
    --------
    >>> config = Config({"a": 1})
    >>> config
    Config({'a': 1})

    Attribute access also works.

    >>> config.a
    1

    Nested mappings are returned as config objects supporting attribute access.

    >>> config = Config({"obj": {"a": 1}})
    >>> config.obj.a
    1
    """

    def __init__(self, *args: Any, __defaults__: bool = True, **kwargs: Any) -> None:
        if __defaults__:
            path = Path(__file__).parent / "base.cfg"
            base = _Config().from_disk(path)
        else:
            base = Config()
        base.update(*args, **kwargs)
        super().__init__(base)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __getitem__(self, key: str) -> Any:
        obj = super().__getitem__(key)
        if isinstance(obj, Mapping):
            obj = Config(obj, __defaults__=False)
        return obj

    def __getattr__(self, name: str) -> Any:
        return self[name]
