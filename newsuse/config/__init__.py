from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from confection import Config as _Config

from newsuse.dotpath import dotget

from .paths import Paths

__all__ = ("Config", "RootConfig", "Paths", "BASE_CONFIG_PATH")

BASE_CONFIG_PATH = Path(__file__).parent.absolute() / "base.cfg"


class Config(_Config):
    """Config with predefined defaults based on :class:`confection.Config`.

    Examples
    --------
    >>> config = Config({"a": 1})
    >>> config
    {'a': 1}

    Attribute access also works.

    >>> config.a
    1

    Nested mappings are returned as config objects supporting attribute access.

    >>> config = Config({"obj": {"a": 1}})
    >>> config.obj.a
    1

    Access using dotpath syntax is also supported.
    >>> config["obj.a"]
    1
    """

    is_initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__["is_initialized"] = True

    def __getitem__(self, key: str) -> Any:
        if self.__dict__.get("is_initialized") and "." in key:
            obj = dotget(self, key)
        else:
            obj = super().__getitem__(key)
        if isinstance(obj, Mapping) and not isinstance(obj, type(self)):
            obj = Config(obj)
            self[key] = obj
        return obj

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except (KeyError, AttributeError) as exc:
            cn = self.__class__.__name__
            errmsg = f"'{cn}' object has no attribute '{name}'"
            raise AttributeError(errmsg) from exc


class RootConfig(Config):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        base = Config().from_disk(BASE_CONFIG_PATH)
        config = Config(*args, **kwargs)
        super().__init__(base.merge(config))

    def from_str(self, *args: Any, **kwargs: Any) -> Self:
        base = Config().from_disk(BASE_CONFIG_PATH)
        config = super().from_str(*args, **kwargs)
        return self.__class__(base.merge(config))
