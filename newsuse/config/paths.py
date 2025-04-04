from pathlib import Path
from typing import Any, Self

from newsuse.types import Namespace, PathLike

__all__ = ("Paths",)


class Paths(Namespace):
    """Paths namespace.

    Attributes
    ----------
    root
        Root path against which other relative paths are resolved.
        Default to the current working directory.
        Is always resolved to an absolute path.

    Examples
    --------
    >>> from tempfile import mkdtemp
    >>> temp = mkdtemp()
    >>> paths = Paths(root=temp)
    >>> paths.root == Path(temp)
    True
    >>> paths.script = "script.py"
    >>> paths.script == Path(temp) / "script.py"
    True

    Absolute paths remain absolute.

    >>> paths.datasets = "/datasets"
    >>> paths.datasets == Path("/datasets")
    True

    Paths can also be made relative to other paths using '@' prefix.

    >>> paths = Paths(temp, data="data", proc="@data/proc")
    >>> paths.proc == Path(temp) / "data" / "proc"
    True
    """

    def __init__(self, root: PathLike | None = None, *args: Any, **kwargs: Any) -> None:
        self.root = Path(root or ".").absolute()
        super().__init__(*args, **kwargs)
        for k, v in self.__dict__.items():
            self.__dict__[k] = Path(v)

    def __setattr__(self, name: str, value: PathLike) -> None:
        path = Path(value)
        if name == "root":
            path = path.absolute()
        super().__setattr__(name, path)

    def __getattribute__(self, name: str) -> Path:
        obj = super().__getattribute__(name)
        if name.startswith("__") and name.endswith("__"):
            return obj
        path = Path(obj)
        if (anchor := path.parts[0]).startswith("@"):
            path = self[anchor[1:]] / Path(*path.parts[1:])
        elif not path.root:
            path = self.root / path
        return path

    def __copy__(self, **kwargs: PathLike) -> Self:
        """Make a copy with additional paths given by ``**kwargs``.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> temp = Path(mkdtemp())
        >>> paths = Paths(temp, data="data")
        >>> new_paths = paths.__copy__(proc="@data/proc")
        >>> new_paths.proc == paths.data / "proc"
        True
        """
        dct = dict(self.__dict__)
        root = dct.pop("root")
        return self.__class__(root, **{**dct, **kwargs})
