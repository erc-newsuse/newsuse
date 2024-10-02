from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from newsuse.types import PathLike


def connect(path: PathLike) -> GoogleDrive:
    """Connect GoogleDrive using a service account with secrets defined in ``path`."""
    settings = {
        "client_config_backend": "service",
        "service_config": {"client_json_file_path": str(Path(path).absolute())},
    }
    auth = GoogleAuth(settings=settings)
    auth.ServiceAuth()
    drive = GoogleDrive(auth)
    return drive
