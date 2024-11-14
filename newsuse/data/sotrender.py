import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd

from newsuse.types import PathLike

from .frame import DataFrame

__all__ = (
    "read_data",
    "normalize_text_content",
    "merge_content_and_snippet",
)


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


def read_data(
    *sources: PathLike,
    metadata: Mapping[str, str | re.Pattern] | None = None,
    **kwargs: Any,
) -> DataFrame:
    """Read _SoTrender_ data files.

    Parameters
    ----------
    *args, **kwargs
        Passed to :meth:`newsuse.data.frame.DataFrame.read_many`.
    metadata
        Mapping from column names (inserted after the key column)
        to regex patterns for extracting values from file names.
        Value to assign should be returned by the first match group.
    """
    # ruff: noqa: C901
    metadata = metadata or {}
    metadata = {k: re.compile(v) if isinstance(v, str) else v for k, v in metadata.items()}

    def _iter():
        for source, df in DataFrame.read_many(*sources, **kwargs):
            df = (
                _normalize_columns(df)
                .convert_dtypes()
                .rename(columns={"fb_post_id": "key"})
            )
            for field, rx in reversed(metadata.items()):
                df.insert(1, field, rx.match(str(source)).group(1))
            yield df

    data = pd.concat(_iter(), axis=0, ignore_index=True).drop_duplicates(
        subset="key", ignore_index=True
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

    prefix = "sotrender@"
    data["key"] = prefix + data["key"].astype(str).str.removeprefix(prefix)

    intcols = [
        "likes",
        "comments",
        "shares",
        "ini",
        *[c for c in data.columns if c.startswith("reactions_")],
    ]
    for col in intcols:
        data[col] = data[col].astype("int64[pyarrow]")

    if (field := "content_type") not in data:
        data.insert(data.shape[1], field, "fb_post")

    return data.convert_dtypes()


def normalize_text_content(
    data: pd.DataFrame,
    na_vals: Iterable[str] = (),
    content_col: str = "text",
    snippet_cols: Iterable[str] = ("link_title", "link_content"),
    type_col: str = "type",
) -> pd.DataFrame:
    """Normalize text and snippet content.

    This removes all fields like ``link_title`` and ``link_content``
    and moves the data to a generic ``snippet`` field depending
    on post type.

    Parameters
    ----------
    data
        Sotrender data frame as returned by :func:`read_data`.
    na_vals
        Additional values (beyond empty string) to be considered as empty content.
    content_col, snippet_cols
        Column names with main content and snippet content.
    """
    nan_remap = {"": pd.NA}
    for val in na_vals:
        if val not in nan_remap:
            nan_remap[val] = pd.NA

    snippet_cols = list(snippet_cols)
    text_cols = [content_col, *snippet_cols]
    for col in text_cols:
        data[col] = data[col].str.strip()
    data = data.replace({col: nan_remap for col in text_cols})

    if snippet_cols:
        snippet = data.pop(snippet_cols[0])
        for col in snippet_cols[1:]:
            if col in data:
                snippet = snippet.combine_first(data.pop(col))
        idx = data.columns.tolist().index(content_col) + 1
        data.insert(idx, "snippet", snippet)

    text_cols = ["text"]
    if (field := "snippet") in data:
        text_cols.append(field)
    data.fillna({col: "" for col in text_cols}, inplace=True)

    data.loc[data[type_col] == "photo", "snippet"] = ""

    return data.convert_dtypes()


def merge_content_and_snippet(
    data: pd.DataFrame,
    content_col: str = "text",
) -> pd.DataFrame:
    """Merge main content and snippet content.

    This removes ``snippet`` column and merges it with the main content.
    New ``has_snippet`` column is added for selecting records with non-empty snippet data.
    """
    data = data.copy()
    for col in [content_col, "snippet"]:
        data[col] = data[col].fillna("").str.strip()

    content_duplication = np.array(
        [t.startswith(s) for t, s in zip(data[content_col], data["snippet"], strict=True)]
    )
    data[content_col] = np.where(
        content_duplication, data[content_col], data[content_col] + "\n\n" + data["snippet"]
    )
    data[content_col] = data[content_col].str.strip()
    has_snippet = ~content_duplication & (data["snippet"] != "")
    idx = data.columns.tolist().index(content_col) + 1
    data.insert(idx, "has_snippet", has_snippet)
    data.drop(columns="snippet", inplace=True)

    return data.convert_dtypes()
