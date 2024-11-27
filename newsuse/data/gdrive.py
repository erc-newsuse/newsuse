import os
import warnings
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from newsuse.types import PathLike


def connect(
    profile: str,
    dirpath: PathLike | None = None,
) -> GoogleDrive:
    """Connect GoogleDrive using a service account with given ``profile``.

    ``dirpath`` specifies the directory containing profile files.
    It defaults to the current value of ``os.environ["NEWSUSE_SECRETS"]``
    when its defined.
    """
    dirpath = Path(dirpath or os.environ.get("NEWSUSE_SECRETS", "."))
    path = dirpath / profile
    settings = {
        "client_config_backend": "service",
        "service_config": {"client_json_file_path": str(Path(path).absolute())},
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        auth = GoogleAuth(settings=settings)
        auth.ServiceAuth()
        drive = GoogleDrive(auth)
    return drive
