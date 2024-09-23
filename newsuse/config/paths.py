from pathlib import Path
from typing import Any

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
        if not path.root:
            path = self.root / path
        return path
