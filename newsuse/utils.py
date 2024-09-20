from pathlib import Path

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
    """

    def __init__(self, config: PathLike) -> None:
        self.config = Path(config)

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
