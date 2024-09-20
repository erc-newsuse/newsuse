import os
from pathlib import Path
from typing import Any

from newsuse.types import PathLike


class ProjectPaths:
    """Project paths container.

    Attributes
    ----------
    config
        Path to project ``config.py`` file, which is expected to the located
        in the top-level module of the project package.
    root
        Path to the project root directory.
    data
        Path to the project data directory.
    raw
        Path to the project data/raw directory.
    proc
        Path to the project data/proc directory.

    Examples
    --------
    >>> from tempfile import mkdtemp
    >>> temp = Path(mkdtemp())
    >>> fpath = temp/"project"/"config.py"
    >>> paths = ProjectPaths(fpath)
    >>> paths.config.parts[-2:]
    ('project', 'config.py')
    >>> paths.root == temp
    True

    It is possible to set additional paths,
    which are automatically converted to :class:`pathlib.Path` instances.
    Non-absolute paths are interpreted relative to the root

    >>> paths.custom = "script.py"
    >>> paths.custom.parts == (*temp.parts, "script.py")
    True
    """

    def __init__(self, config: PathLike) -> None:
        self._config = Path(config).absolute()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, str | os.PathLike):
            value = Path(value)
            if not value.root:
                value = self.root / value
        super().__setattr__(name, value)

    @property
    def config(self) -> Path:
        return self._config

    @property
    def root(self) -> Path:
        return self.config.parent.parent

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def proc(self) -> Path:
        return self.data / "proc"

    @property
    def raw(self) -> Path:
        return self.data / "raw"
