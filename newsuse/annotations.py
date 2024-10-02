import io
from collections.abc import Hashable, Sequence
from functools import singledispatchmethod
from typing import Any, Self

import numpy as np
import pandas as pd
from pydrive2.drive import GoogleDrive
from pydrive2.files import GoogleDriveFile

from newsuse.config import Config
from newsuse.data import DataFrame
from newsuse.types import PathLike
from newsuse.utils import inthash


class Annotations:
    """Annotations class for managing iterative data annotation processes.

    It is assumed that annotations are stored in an Excel file.

    Attributes
    ----------
    source
        Address string for accessing annotations Excel sheet.
        By default it is assumed it is a path to a local Excel file.
    key
        Name of the key column with unique observation identifiers.
        Defaults to the first column when ``None``.
    sheet_index_name
        Name of the column storing names of multiple sheets in ``source``.
    """

    ann: DataFrame
    source: PathLike | None
    default_config = Config({"key": None, "dtype_backend": "pyarrow"})

    def __init__(
        self,
        config: Config,
        ann: pd.DataFrame | None = None,
        sampler: np.random.Generator | None = None,
    ) -> None:
        self.ann = ann
        self.config = Config(self.default_config.merge(config))
        self._sampler = sampler

    @property
    def annotator_cols(self) -> list[str]:
        return [a if a.startswith("@") else f"@{a}" for a in self.config.annotators]

    @property
    def sampler(self) -> np.random.Generator:
        if self._sampler is None:
            self._sampler = self.get_sampler()
        return self._sampler

    def get_sampler(self, df: pd.DataFrame | None = None) -> np.random.Generator:
        if df is None:
            df = self.ann
        data = df["key"].to_numpy()
        data.sort()
        seed = inthash(data) + self.config.seed
        return np.random.default_rng(seed)

    def get_key(self, ann: pd.DataFrame) -> str:
        if key := getattr(self.config, "key", None):
            return key
        return ann.columns[0]

    def _process_after_read(self, sheets: dict[str, pd.DataFrame]) -> DataFrame:
        if self.config.sheet_index_name:
            ann = pd.concat(
                sheets, axis=0, ignore_index=False, names=[self.config.sheet_index_name]
            )
            key = self.get_key(ann)
            ann = (
                ann.reset_index(self.config.sheet_index_name)
                .set_index(key)
                .reset_index(key)
                .reset_index(drop=True)
            )
        else:
            ann = pd.concat(sheets, axis=0, ignore_index=True)
            key = self.key or ann.columns[0]
        ann.dropna(subset=[key, self.config.content_name], ignore_index=True, inplace=True)
        acols = self.get_annotator_cols(ann)
        for acol in acols:
            if acol not in ann:
                ann.insert(ann.shape[1] - 1, acol, "")
            ann[acol] = self.sanitize_labels(ann[acol])
        ann = self.order_cols(ann).rename(columns={key: "key"})
        ann = ann.convert_dtypes(dtype_backend=self.config.dtype_backend)
        return DataFrame(ann)

    @singledispatchmethod
    def read(self, source, **kwargs: Any) -> Self:
        """Read annotations' data from ``source``."""
        sheets = DataFrame.from_excel(source, sheet_name=None, **kwargs)
        self.ann = self._process_after_read(sheets, **kwargs)
        return self

    @read.register
    def _(self, source: GoogleDriveFile, **kwargs: Any) -> Self:
        if not source.metadata:
            source.FetchMetadata()
        if (ext := source["fileExtension"]) != "xlsx":
            errmsg = f"cannot read annotations in '.{ext}' format"
            raise ValueError(errmsg)
        sheets = DataFrame.from_gdrive_file(source, sheet_name=None, **kwargs)
        self.ann = self._process_after_read(sheets, **kwargs)
        return self

    @read.register
    def _(self, source: GoogleDrive, id: str, **kwargs: Any) -> Self:
        file = source.CreateFile({"id": id})
        return self.read(file, **kwargs)

    def order_cols(self, ann: pd.DataFrame) -> pd.DataFrame:
        acols = self.annotator_cols
        front = [c for c in ann.columns if c not in [*acols, self.config.content_name]]
        return ann[[*front, *acols, self.config.content_name]]

    def get_annotator_cols(self, ann: pd.DataFrame | None = None) -> list[str]:
        acols = self.annotator_cols
        if ann is not None:
            for acol in ann:
                if acol.startswith("@") and acol not in acols:
                    acols.append(acol)
        return acols

    def sanitize_labels(self, acol: pd.Series) -> pd.Series:
        """Sanitize labels by mapping integer-like values to proper labels.

        Examples
        --------
        >>> annotations = Annotations({"labels": ["OTHER", "POLITICAL"]})
        >>> s = pd.Series(["OTHER", "1", "0", "0.0", "1.00", "1.0"])
        >>> annotations.sanitize_labels(s)
        0        OTHER
        1    POLITICAL
        2        OTHER
        3        OTHER
        4    POLITICAL
        5    POLITICAL
        dtype: string
        """
        acol = acol.astype(pd.StringDtype("pyarrow"))
        mask = acol.notnull()
        remap = {rf"^{i}(\.0+)?\b": label for i, label in enumerate(self.config.labels)}
        acol[mask] = acol[mask].str.strip().replace(remap, regex=True).str.strip()
        acol = acol.astype(pd.StringDtype(self.config.dtype_backend))
        use = acol.isnull()
        for label in self.config.labels:
            use |= acol.str.startswith(label)
        acol[~use] = pd.NA
        return acol

    def _sample_from(
        self,
        data: pd.DataFrame,
        n: int,
        *,
        groups: str | Sequence[str] = (),
        weights: Hashable | None = None,
    ) -> DataFrame:
        ignore = self.ann["key"].to_numpy()
        sheet_name = self.config.sheet_index_name
        if groups and isinstance(groups, str):
            groups = [groups]
        groups = list(groups)

        key = self.get_key(data)
        data = data[~data[key].isin(ignore)]

        if sheet_name:
            groups = [sheet_name, *groups]
        if groups:
            data = data.groupby(groups)  # type: ignore

        sample = data.sample(
            n=n, replace=False, weights=weights, random_state=self.sampler
        ).reset_index(drop=True)
        sample.insert(0, "key", sample.pop(key))

        usecols = ["key", self.config.content_name]
        if sheet_name:
            usecols.append(sheet_name)

        sample = sample[usecols]
        return DataFrame(sample)

    def append_from(
        self,
        data: pd.DataFrame,
        n: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Append records sample from ``path``.

        Parameters
        ----------
        path
            Path to a data source.
        n
            Number of samples to draw.
            Do not append if non-positive.
        **kwargs
            Passed to :meth:`sample_from`.
        """
        if n > 0:
            msg = f"Adding {n} samples per group"
            sample = self._sample_from(data, n, **kwargs)
            self.ann = DataFrame(pd.concat([self.ann, sample], axis=0, ignore_index=True))
        else:
            msg = "Adding no new samples"
            print(msg)
        return self

    def shuffle_empty(self, *, groups: Hashable | Sequence[Hashable] | None = None) -> Self:
        """Shuffle examples without annotations, possibly in ``groups``."""
        acols = self.annotator_cols

        msg = "Shuffling non-annotated examples"
        if groups is not None:
            if isinstance(groups, str):
                groups = [groups]
            msg += f" by {', '.join(f"'{g}'" for g in groups)}"
        print(msg)

        def shuffle(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values(by="key")
            has_annotations = df[acols].notnull().any(axis=1)
            filled = df[has_annotations]
            empty = df[~has_annotations]
            sampler = self.get_sampler(filled)
            empty_shuffled = empty.sample(
                frac=1,
                replace=False,
                random_state=sampler,
                ignore_index=True,
            )
            return pd.concat([filled, empty_shuffled], axis=0, ignore_index=True)

        if groups is None:
            self.ann = shuffle(self.ann)
        else:
            self.ann = self.ann.groupby(groups).apply(shuffle)
        self.ann.reset_index(drop=True, inplace=True)

        return self

    @singledispatchmethod
    def write(
        self,
        target,
        *,
        col_width: float = 20,
        textcol_width: float = 750,
        engine: str = "xlsxwriter",
    ) -> None:
        def do_write(
            df: pd.DataFrame,
            writer: pd.ExcelWriter,
            *,
            sheet_name: str = "Sheet1",
            index_name: str | None = None,
        ) -> None:
            if index_name and index_name in df:
                del df[index_name]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            max_row, max_col = df.shape
            max_col -= 1
            textf = writer.book.add_format({"text_wrap": True})
            sheet = writer.sheets[sheet_name]
            sheet.set_column(0, max_col, col_width)
            sheet.set_column_pixels(max_col, max_col, textcol_width, textf)
            sheet.autofilter(0, 0, max_row, max_col)
            sheet.freeze_panes(1, 0)

        with pd.ExcelWriter(target, engine=engine) as writer:
            if index_name := self.config.sheet_index_name:
                for key, df in self.ann.groupby([index_name]):
                    sheet_name, *_ = key
                    do_write(df, writer, sheet_name=sheet_name, index_name=index_name)
            else:
                do_write(self.ann, writer, index_name=index_name)

    @write.register
    def _(self, target: GoogleDriveFile, **kwargs: Any) -> None:
        DataFrame.check_gdrive_file(target)
        buffer = io.BytesIO()
        self.write(buffer, **kwargs)
        target.content = buffer
        target.Upload()

    @write.register
    def _(self, target: GoogleDrive, id: str, **kwargs: Any) -> None:
        file = target.CreateFile({"id": id})
        self.write(file, **kwargs)

    def with_labels(
        self,
        *,
        __validate_uniqueness: bool = True,
        **labels: PathLike,
    ) -> DataFrame:
        """Get annotations data frame with human and possibly some external labels.

        Final ``"label"`` column is built by coalescing label columns
        starting from human annotations.

        Parameters
        ----------
        **labels
            Key-value pairs from label names to data paths.

        Raises
        ------
        ValueError
            If there are duplicated records after left-joining
            with external labels.
        """
        acols = [a for a in self.get_annotator_cols(self.ann) if a in self.ann]
        ann = self.ann.assign(human=lambda df: df[acols].mode(axis=1, dropna=True)[0])
        ann.drop(columns=acols, inplace=True)
        for name, path in labels.items():
            data = DataFrame.from_(path)
            key = self.get_key(data)
            rename = {key: "key", "label": name}
            if "prob" in data:
                rename["prob"] = f"{name}_prob"
            data.rename(columns=rename, inplace=True)
            ann = ann.merge(data, how="left", on="key")

        if __validate_uniqueness and len(ann) != ann["key"].nunique():
            errmsg = "there are duplicated records after joining with external labels"
            raise ValueError(errmsg)

        ann["label"] = ann["human"]
        for label in labels:
            ann["label"] = ann["label"].combine_first(ann[label])

        return ann

    def update_columns(self, data: pd.DataFrame, *columns: str) -> Self:
        """Update values in ``*columns`` using new ``data``."""
        key = self.get_key(data)
        remap = {key: "key", **{c: f"{c}_new" for c in columns}}
        data = data[list(remap)].rename(columns=remap)
        ann = self.ann.merge(data, how="left", on="key")
        for col in columns:
            ann[col] = ann.pop(f"{col}_new").combine_first(ann.pop(col))
        self.ann = ann
        return self
