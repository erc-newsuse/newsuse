from datetime import date, datetime, time
from typing import Any

import pandas as pd

from .frame import DataFrame

__all__ = ("read_data",)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(name: str) -> str:
        name = str(name).lower()
        if name.startswith("reactions_"):
            ns, name = name.split("_", 1)
            name = f"{ns}_{name.upper()}"
        return name

    processed = set()
    remap = {}
    for col in df.columns:
        renamed = normalize(col)
        if renamed in processed:
            del df[col]
        else:
            processed.add(renamed)
            processed.add(col)
            remap[col] = renamed
    return df.rename(columns=remap)


def read_data(*args: Any, **kwargs: Any) -> DataFrame:
    """Read _SoTrender_ data files.

    Parameters
    ----------
    *args, **kwargs
        Passed to :meth:`newsuse.data.frame.DataFrame.read_many`.
    """
    kwargs = {"key": "fb_post_id", "drop_before_key": True, **kwargs}
    key = kwargs["key"]
    data = (
        pd.concat(
            [df.pipe(_normalize_columns) for df in DataFrame.read_many(*args, **kwargs)],
            axis=0,
            ignore_index=True,
        )
        .pipe(DataFrame)
        .convert_dtypes()
        .drop_duplicates(subset=key, ignore_index=True)
    )

    if (col := "date") in data and pd.api.types.is_string_dtype(data[col]):
        data[col] = data[col].map(date.fromisoformat)
    if (col := "hour") in data:
        if pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].map(time.fromisoformat)
        elif pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].map(lambda t: pd.Timestamp(t * 1e9).time())

    if "date" in data and "hour" in data:
        pos = data.columns.tolist().index("date")
        ts = [
            datetime.combine(d, t)
            for d, t in zip(data.pop("date"), data.pop("hour"), strict=True)
        ]
        data.insert(pos, "timestamp", ts)

    data.drop_duplicates(subset=key, ignore_index=True, inplace=True)
    return data
