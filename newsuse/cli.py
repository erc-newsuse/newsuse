import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import Any

from pydantic import BaseModel

from newsuse.utils import match_signature

__all__ = ("command",)


def _parse_arg(arg: str):
    if arg.startswith("--") and "=" in arg:
        name, arg = arg.removeprefix("--").rsplit("=", 1)
        name = name.replace("-", "_")
    else:
        name = None
    value = json.loads(arg)
    return name, value


def command(spec: type) -> Callable[..., Any]:
    if not is_dataclass(spec) and not issubclass(spec, BaseModel):
        spec = dataclass(frozen=True)(spec)

    def interface() -> spec:
        _, *argv = sys.argv

        kwargs = {}

        if argv and not argv[0].startswith("--f="):
            args = []
            kwargs = {}
            for arg in argv:
                name, value = _parse_arg(arg)
                if name:
                    kwargs[name] = value
                else:
                    args.append(value)
            kwargs = match_signature(spec, *args, **kwargs)

        return spec(**kwargs)

    return interface
