from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import copy
from typing import Any, Self

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, ConfigValueError

from newsuse.dotpath import dotimport

from .paths import Paths

__all__ = ("Config",)


if not OmegaConf.has_resolver(name := "import"):
    OmegaConf.register_new_resolver(name, dotimport)
if not OmegaConf.has_resolver(name := "eval"):
    OmegaConf.register_new_resolver(name, eval)


def make(self, *args: Any, **kwargs: Any) -> Any:
    spec = dict(self)
    constructor = dotimport(spec.pop("make!"))
    return constructor(*args, **{**spec, **kwargs})


DictConfig.make = make
DictConfig.__call__ = make


class Config(DictConfig):
    """Config based on :class:`omegaconf.DictConfig`.

    It comes with special pre-defined resolvers:

    import
        Imports object using :func:`newsuse.dotpath.dotimport`.

    eval
        Evaluates string the string content of a field
        using :func:`ast.literal_eval`.

    Moreover, it allows initializing objects from special nodes with 'make!'
    field, which is used to import a constructor callable and passes all other fields
    as ``**kwargs``.

    Examples
    --------
    >>> config = Config({
    ...     "sum": "${import:builtins:sum}",
    ...     "obj": {
    ...         "make!": ":dict",
    ...         "a": 1
    ...     },
    ...     "frac": "${eval:1/2}"
    ... })
    >>> config["sum"]([1, 2])
    3
    >>> pre_make = dict(config["obj"])
    >>> config["obj"].make(b=2)  # extra args and kwargs can be passed
    {'a': 1, 'b': 2}
    >>> config.obj()  # Calling work too
    {'a': 1}
    >>> config["obj"] == pre_make
    True
    >>> config["frac"]
    0.5

    Special ``defaults!`` key-directive can be used to define a section using
    another section as a source of default values (it is flat copied).

    >>> config = Config({
    ...     "a": {"a1": 1, "a2": 2},
    ...     "b": {"defaults!": "${..a}", "a2": 0, "a3": 3}
    ... })
    >>> config.resolve()
    {'a': {'a1': 1, 'a2': 2}, 'b': {'a1': 1, 'a2': 0, 'a3': 3}}
    """

    def __init__(self, content: Mapping | None = None, *args: Any, **kwargs: Any) -> None:
        content = content or {}
        if not isinstance(content, DictConfig):
            content = dict(content)
        super().__init__(content, *args, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping):
            value = self.__class__(value)
        super().__setitem__(key, value)

    def __copy__(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().__copy__(*args, **kwargs))

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().__deepcopy__(*args, **kwargs))

    def merge(self, *configs: Mapping) -> Self:
        """Merge with other ``*config`` in LIFO fashion."""
        config = self.__class__(self)
        for other in configs:
            config = self.__class__({**self, **other})
        return config

    def resolve(self) -> Self:
        """Resolve interpolations using :meth:`omegaconf.OmegaConf.resolve`."""
        OmegaConf.resolve(self)
        self._resolve_defaults()
        return self

    def _resolve_defaults(self) -> None:
        _key = "defaults!"
        if _key in self:
            new = {**self.pop(_key), **self}
            self.clear()
            self.update(new)
        for v in self.values():
            if isinstance(v, Config):
                v._resolve_defaults()
