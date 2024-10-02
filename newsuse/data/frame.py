import io
import re
from collections.abc import Callable, Iterator, Mapping
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
from pydrive2.drive import GoogleDrive
from pydrive2.files import GoogleDriveFile
from pyreadr import read_r
from tqdm.auto import tqdm

from newsuse.types import PathLike

_ExcelEngines = Literal["openpyxl"]


__all__ = ("DataFrame", "Series")


class Series(pd.Series):
    """Simple wrapper around :class:`pandas.Series`
    for interoperability with a customize :class:`DataFrame` class.
    """

    dtype_backend = "pyarrow"

    @property
    def _constructor(self) -> type["Series"]:
        return self.__class__

    @property
    def _constructor_expanddim(self) -> type["DataFrame"]:
        return DataFrame

    def convert_dtypes(
        self, *args: Any, dtype_backend: str = dtype_backend, **kwargs: Any
    ) -> Self:
        """Convert columns to best possible dtypes"""
        return super().convert_dtypes(*args, dtype_backend=dtype_backend, **kwargs)


class DataFrame(pd.DataFrame):
    """Simple wrapper around :class:`pandas.DataFrame`
    providing customized constructor methods.
    """

    dtype_backend = "pyarrow"
    parquet_engine = "pyarrow"

    @property
    def _constructor(self) -> type["DataFrame"]:
        return self.__class__

    @property
    def _constructor_sliced(self) -> type["Series"]:
        return Series

    @classmethod
    def _make_read_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        return {"dtype_backend": cls.dtype_backend, **kwargs}

    def convert_dtypes(
        self, *args: Any, dtype_backend: str = dtype_backend, **kwargs: Any
    ) -> Self:
        """Convert columns to best possible dtypes"""
        return super().convert_dtypes(*args, dtype_backend=dtype_backend, **kwargs)

    @staticmethod
    def guess_storage_type(source: Any) -> str:
        """Get data storage type for source address.

        Parameters
        ----------
        source
            Any possible data source, e.g. a path-like object of a connection.

        Examples
        --------
        >>> DataFrame.guess_storage_type("data.jsonl.gz")
        'jsonl'
        >>> DataFrame.guess_storage_type(1)
        Traceback (most recent call last):
        ValueError: ...
        """
        if isinstance(source, PathLike):
            ftype = Path(source).name.split(".")[1:2]
            if not ftype:
                errmsg = "path-like sources without extensions are not supported"
                raise ValueError(errmsg)
            return ftype.pop()
        errmsg = f"cannot interpret '{source}' source"
        raise ValueError(errmsg)

    def _get_writer(self, ext: str) -> Callable[[Self, ...], None]:
        try:
            return getattr(self, f"to_{ext}")
        except AttributeError as exc:
            errmsg = f"'{ext}' data sources are not supported"
            raise AttributeError(errmsg) from exc

    @singledispatchmethod
    def to_(self, target, *args: Any, **kwargs: Any) -> None:
        """Guess desired storage type and try to write data to it.

        Parameters
        ----------
        target
            Any possible data storage, e.g. path-like object or a connection.
            In general, its structure is used to determine an appropriate `from_*`
            constructor method, e.g. :meth:`from_csv` when `source` is path-like with
            the '.csv' file extension. Compression suffixes, e.g. `.gz` are handled
            automatically for many standard file types.
        *args, **kwargs
            Passed to a selected reading method.

        Raises
        ------
        AttributeError
            If there is no writer method for a given target.
        ValueError
            If a given target cannot be parsed.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.csv.gz"
        >>> df = DataFrame({"a": [1,2]})
        >>> df.to_(path, index=False)
        >>> bool(DataFrame.from_(path).eq(df).all().all())
        True
        """
        ext = self.guess_storage_type(target)
        writer = self._get_writer(ext)
        return writer(target, *args, **kwargs)

    def to_jsonl(self, *args: Any, orient: str = "records", **kwargs: Any) -> None:
        """Write data as JSON lines.

        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`pandas.DataFrame.to_json`.
            Argument ``lines=True`` is always used.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.jsonl"
        >>> df = DataFrame({"a": [1,2]})
        >>> df.to_jsonl(path)
        >>> bool(DataFrame.from_jsonl(path).eq(df).all().all())
        True
        """
        self.to_json(*args, lines=True, orient=orient, **kwargs)

    def to_tsv(self, *args: Any, **kwargs: Any) -> None:
        """Write data as TSV file.

        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`pandas.DataFrame.to_csv`.
            Argument ``sep="\\t"` is always used.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.tsv"
        >>> df = DataFrame({"a": [1,2]})
        >>> df.to_tsv(path, index=False)
        >>> bool(DataFrame.from_tsv(path).eq(df).all().all())
        True
        """
        self.to_csv(*args, sep="\t", **kwargs)

    def to_parquet(
        self,
        *args,
        index: bool | None = False,
        engine: str = parquet_engine,
        compression: str = "zstd",
        compression_level=9,
        **kwargs: Any,
    ) -> None:
        """Save as Parquet file.

        See :meth:`pandas.DataFrame.to_parquet` for details.
        """
        super().to_parquet(
            *args,
            index=index,
            engine=engine,
            compression=compression,
            compression_level=compression_level,
            **kwargs,
        )

    def to_excel(
        self,
        excel_writer: PathLike | pd.ExcelWriter,
        sheet_name: str = "Sheet1",
        *,
        engine: _ExcelEngines = _ExcelEngines.__args__[0],
        **kwargs: Any,
    ) -> None:
        """Write to Excel file.

        See :meth:`pandas.DataFrame.to_excel` and :class:`pandas.ExcelWriter`
        for details.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> temp = Path(mkdtemp())
        >>> path = temp/"test1.xlsx"
        >>> DataFrame({"a": [1, 2]}).to_excel(path, index=False)
        >>> DataFrame.from_excel(path)
           a
        0  1
        1  2

        Hierarchical columns must be saved together with row index,
        this is a limitation of :mod:`pandas`, but saving index is the default
        behavior.

        >>> path = temp/"test2.xlsx"
        >>> DataFrame({("DATA", "key1", "key2"): [1, 2]}).to_excel(path)
        >>> DataFrame.from_excel(path, header=[0,1,2], index_col=0)
          DATA
          key1
          key2
        0    1
        1    2

        More complex tasks such as writing new sheets in an exsistiing workbook
        can be achieved using :class:`pandas.ExcelWriter`.

        >>> path = temp/"test4.xlsx"
        >>> df = DataFrame({"a": [1,2]})
        >>> df.to_excel(path, index=False)
        >>> # write to a new sheet
        >>> with pd.ExcelWriter(path, mode="a", engine="openpyxl") as writer:
        ...     df.to_excel(writer, "Sheet2", index=False)
        >>> data = DataFrame.from_excel(path, sheet_name=["Sheet1", "Sheet2"])
        >>> data["Sheet1"]
           a
        0  1
        1  2
        >>> data["Sheet2"]
           a
        0  1
        1  2

        Appending works also when writing data frames with hierarchical columns
        and simple index.

        >>> path = temp/"test5.xlsx"
        >>> df = DataFrame({
        ...     ("DATA", "k1"): [1,2],
        ...     ("DATA", "k2"): [2,3],
        ...     ("TEXT", "title"): ["a", "b"],
        ...     ("TEXT", "content"): ["c", "d"]
        ... })
        >>> df.to_excel(path, "Sheet1")
        >>> with pd.ExcelWriter(path, mode="a", engine="openpyxl") as writer:
        ...     df.to_excel(writer, "Sheet2")
        >>> # Read all sheets
        >>> data = DataFrame.from_excel(
        ...     path, sheet_name=None, header=[0,1], index_col=0
        ... )
        >>> data["Sheet1"]
                  DATA     TEXT
            k1 k2 title content
        0    1  2     a       c
        1    2  3     b       d
        >>> data["Sheet2"]
                  DATA     TEXT
            k1 k2 title content
        0    1  2     a       c
        1    2  3     b       d
        """
        super().to_excel(
            excel_writer,
            sheet_name=sheet_name,
            engine=engine,
            **kwargs,
        )

    def to_xlsx(self, *args: Any, **kwargs: Any) -> None:
        """Write to XLSX file.

        See :meth:`to_excel` for details.
        """
        self.to_excel(*args, **kwargs)

    def to_gdrive_file(self, target: GoogleDriveFile, *args: Any, **kwargs: Any) -> None:
        """Write to a :class:`pydrive2.files.GoogleDriveFile`.

        The file must define proper MIME type and file extension.
        """
        self.check_gdrive_file(target)
        ext = str(target["ext"]).removeprefix(".")
        writer = self._get_writer(f".{ext}")
        buffer = io.BytesIO()
        writer(buffer, *args, **kwargs)
        target.content = buffer
        target.Upload()

    def to_gdrive(self, target: GoogleDrive, id: str, *args: Any, **kwargs: Any) -> None:
        """Write to :class:`pydrive2.drives.GoogleDrive` file by ``id``."""
        file = target.CreateFile({"id": id})
        self.to_gdrive(file, *args, **kwargs)

    @staticmethod
    def check_gdrive_file(target: GoogleDriveFile) -> None:
        """Check if a :class:`pydrive2.files.GoogleDriveFile` defines MIME and extension."""
        if not target.metadata:
            target.FetchMetadata()
        if not target.get("mimeType"):
            errmsg = "'target' file must define MIME type"
            raise ValueError(errmsg)
        if not target.get("fileExtension"):
            errmsg = "'target' file must define file extension"
            raise ValueError(errmsg)

    @to_.register
    def _(self, target: GoogleDriveFile, *args: Any, **kwargs: Any) -> None:
        self.to_gdrive_file(target, *args, **kwargs)

    @to_.register
    def _(self, target: GoogleDrive, *args: Any, **kwargs: Any) -> None:
        self.to_gdrive(target, *args, **kwargs)

    @classmethod
    def _get_reader(cls, ext: str) -> Callable[..., Self]:
        try:
            return getattr(cls, f"from_{ext}")
        except AttributeError as exc:
            errmsg = f"'{ext}' data sources are not supported"
            raise AttributeError(errmsg) from exc

    @singledispatchmethod
    @classmethod
    def from_(cls, source: Any, *args: Any, **kwargs: Any) -> Self:
        """Guess storage type and try to use it to construct a data frame.

        Currently only path-like sources and :class:`GoogleDriveFile`s are supported.

        Parameters
        ----------
        source
            Any possible data source, e.g. a path-like object of a connection.
            In general, its structure is used to determine an appropriate `from_*`
            constructor method, e.g. :meth:`from_csv` when `source` is path-like with
            the '.csv' file extension. Compression suffixes, e.g. `.gz` are handled
            automatically for many standard file types.
        *args, **kwargs
            Passed to a selected reading method.

        Raises
        ------
        AttributeError
            If there is no reader method for a given source.
        ValueError
            If a given source cannot be parsed.

            Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.tsv"
        >>> with open(path, "w") as fh:
        ...     _ = fh.write("a\\tb\\n1\\t2")
        >>> df = DataFrame.from_(path)
        >>> df
           a  b
        0  1  2
        >>> DataFrame.from_("data.spss")
        Traceback (most recent call last):
        AttributeError: ...
        """
        ext = cls.guess_storage_type(source)
        reader = cls._get_reader(ext)
        return reader(source, *args, **kwargs)

    @from_.register
    @classmethod
    def _(cls, source: GoogleDriveFile, *args: Any, **kwargs: Any) -> Self:
        return cls.from_gdrive_file(source, *args, **kwargs)

    @from_.register
    @classmethod
    def _(cls, source: GoogleDrive, *args: Any, **kwargs: Any) -> Self:
        return cls.from_gdrive(source, *args, **kwargs)

    @classmethod
    def from_csv(cls, *args: Any, **kwargs: Any) -> Self:
        """See :func:`pandas.read_csv`.

        Examples
        --------
        >>> import io
        >>> buff = io.StringIO("a,b\\n1,2")
        >>> df = DataFrame.from_csv(buff)
        >>> df
           a  b
        0  1  2
        >>> isinstance(df, DataFrame)
        True
        """
        kwargs = cls._make_read_kwargs(**kwargs)
        return cls(pd.read_csv(*args, **kwargs))

    @classmethod
    def from_tsv(cls, *args: Any, **kwargs: Any) -> Self:
        """See :func:`read_csv` with `sep="\\t"`

        Examples
        --------
        Examples
        --------
        >>> import io
        >>> buff = io.StringIO("a\\tb\\n1\\t2")
        >>> df = DataFrame.from_tsv(buff)
        >>> df
           a  b
        0  1  2
        >>> isinstance(df, DataFrame)
        True
        """
        return cls.from_csv(*args, sep="\t", **kwargs)

    @classmethod
    def from_json(cls, *args: Any, **kwargs: Any) -> Self:
        """See :func:`pandas.read_json`.

        Examples
        --------
        >>> import io
        >>> buff = io.StringIO('{"a": [1], "b": [2]}')
        >>> df = DataFrame.from_json(buff)
        >>> df
           a  b
        0  1  2
        >>> isinstance(df, DataFrame)
        True
        """
        kwargs = cls._make_read_kwargs(**{"convert_dates": False, **kwargs})
        return cls(pd.read_json(*args, **kwargs))

    @classmethod
    def from_jsonl(cls, *args: Any, **kwargs: Any) -> Self:
        """See :func:`pandas.read_json` with `lines=True`.

        Examples
        --------
        >>> import io
        >>> buff = io.StringIO('{"a": 1, "b": 2}\\n{"a": 2, "b": 3}')
        >>> DataFrame.from_jsonl(buff)
           a  b
        0  1  2
        1  2  3
        """
        return cls.from_json(*args, lines=True, **kwargs)

    @classmethod
    def from_parquet(cls, *args: Any, engine: str = parquet_engine, **kwargs: Any) -> Self:
        """See :func:`pandas.read_parquet`.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.parquet"
        >>> df = DataFrame({"a": [1,2]})
        >>> df.to_parquet(path)
        >>> new_df = DataFrame.from_parquet(path)
        >>> bool(new_df.eq(df).all().all())
        True
        >>> isinstance(new_df, DataFrame)
        True
        """
        kwargs = cls._make_read_kwargs(**kwargs)
        return cls(pd.read_parquet(*args, engine=engine, **kwargs))

    @classmethod
    def from_excel(
        cls,
        *args: Any,
        engine: _ExcelEngines = _ExcelEngines.__args__[0],
        **kwargs: Any,
    ) -> Self | dict[str, Self]:
        """See :func:`pandas.read_excel`.

        Parameters
        ----------
        index_col
            It defaults to ``0`` when header is a list.
            The purpose is to facilitate proper reading of data frames
            with hierarchical columns.

        Examples
        --------
        >>> from tempfile import mkdtemp
        >>> from pathlib import Path
        >>> path = Path(mkdtemp())/"test.xlsx"
        >>> DataFrame({"a": [1, 2]}).to_excel(path, index=False)
        >>> df = DataFrame.from_excel(path)
        >>> df
           a
        0  1
        1  2
        >>> isinstance(df, DataFrame)
        True
        """
        data = pd.read_excel(*args, **cls._make_read_kwargs(engine=engine, **kwargs))
        if isinstance(data, Mapping):
            data = {k: cls(v) for k, v in data.items()}
        else:
            data = cls(data)
        return data

    @classmethod
    def from_xlsx(cls, *args: Any, **kwargs: Any) -> Self:
        """See :meth:`from_excel`."""
        return cls.from_excel(*args, **kwargs)

    @classmethod
    def from_rds(cls, *args: Any, **kwargs: Any) -> Self:
        """Read from ``.rds`` files using :mod:`pyreadr`.

        See :func:`pyreadr.read_r` for details.
        """
        return cls(read_r(*args, **kwargs)[None]).convert_dtypes()

    @classmethod
    def read_many(
        cls,
        *paths: PathLike,
        key: str | None = None,
        drop_before_key: bool = False,
        read_opts: Mapping[str, Any] | None = None,
        progress: Mapping[str, Any] | bool = False,
        re_flags: int = 0,
        re_ignorecase: bool = False,
        **meta: str,
    ) -> Iterator[Self]:
        """Iterate over and read multiple data files.

        Parameters
        ----------
        path
            Path to file(s).
            Multiple source files can be specified using GLOB patterns.
        key
            Name of the key column.
        progress
            Should the iterator be wrapped in progress bar.
            Can be passed as bool or as a mapping with :func:`tqdm.tqdm` options.
        drop_before_key
            Should columns before the key column be dropped after reading.
        re_flags, re_ignorecase
            Options passed to :func:`re.compile`
            used for extracting metadata from file names.
        **meta
            Names to regex pairings used for adding metadata columns to data frames.
            Regular expression must Capture the metadata value in the first match group.
        """
        read_opts = read_opts or {}
        if re_ignorecase:
            re_flags |= re.IGNORECASE
        meta_rx = {name: re.compile(pattern, re_flags) for name, pattern in meta.items()}

        files = []
        for path in map(Path, paths):
            for file in path.parent.glob(path.name):
                files.append(file)  # noqa

        def _iter():
            for file in files:
                df = DataFrame.from_(file, **read_opts)
                if key:
                    if drop_before_key:
                        df = df.loc[:, key:]
                    df.insert(0, key, df.pop(key))
                meta_pos = 1 if key else 0
                for name, rx in reversed(meta_rx.items()):
                    value = rx.match(file.name).group(1)
                    df.insert(meta_pos, name, value)
                df = cls(df).convert_dtypes()
                yield df

        if progress:
            tqdm_kwargs = {} if isinstance(progress, bool) else progress
            yield from tqdm(_iter(), **{"total": len(files), **tqdm_kwargs})
        else:
            yield from _iter()

    @classmethod
    def from_gdrive_file(cls, source: GoogleDriveFile, *args: Any, **kwargs: Any) -> Self:
        """Construct from :class:`GoogleDriveFile` instance.

        ``*args`` and ``**kwargs`` are passed to an appropriate format reader.
        """
        if not source.metadata:
            source.FetchMetadata()
        reader = cls._get_reader(source["fileExtension"])
        buffer = io.BytesIO(source.GetContentIOBuffer().read())
        return reader(buffer, *args, **kwargs)

    @classmethod
    def from_gdrive(cls, source: GoogleDrive, id: str, *args: Any, **kwargs: Any) -> Self:
        """Construct from :class:`GoogleDrive`.

        ``id`` is the Google Drive file id.
        ``*args`` and ``*kwargs`` are passed to an appropriate format reader.
        """
        file = source.CreateFile({"id": id})
        return cls.from_gdrive_file(file, *args, **kwargs)
