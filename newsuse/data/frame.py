from __future__ import annotations

import io
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal, Self, overload

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
    def _constructor(self) -> type[Series]:
        return self.__class__

    @property
    def _constructor_expanddim(self) -> type[DataFrame]:
        return DataFrame

    def convert_dtypes(
        self, *args: Any, dtype_backend: str = dtype_backend, **kwargs: Any
    ) -> Self:
        """Convert columns to best possible dtypes"""
        return super().convert_dtypes(*args, dtype_backend=dtype_backend, **kwargs)

    def mode(
        self,
        *,
        dropna: bool = True,
        sort_values: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        """Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna
            Don't consider counts of NaN/NaT.
        sort_values
            Should values be sorted when there are multiple modes.
            When ``True`` defaults to sorting lexicographically by values.
            However, this can be modified by using ``**kwargs`` which are
            passed to :func:`sorted`.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        Examples
        --------
        >>> s = Series(["B", "A", "A", "B", "C"])
        >>> s.mode()
        0    B
        1    A
        dtype: ...
        >>> s.mode(sort_values=True)
        0    A
        1    B
        dtype: ...
        >>> Series([]).mode()
        Series([], dtype: ...)
        """
        s = self.dropna() if dropna else self
        if s.size <= 0:
            return s
        counter = Counter(s)
        max_count = max(counter.values())
        modes = [(v, c) for v, c in counter.most_common() if c == max_count]
        if sort_values:
            kwargs = {"key": lambda m: m[0], **kwargs}
            modes = sorted(modes, **kwargs)
        return self._constructor([m[0] for m in modes]).convert_dtypes()


class DataFrame(pd.DataFrame):
    """Simple wrapper around :class:`pandas.DataFrame`
    providing customized constructor methods.
    """

    dtype_backend = "pyarrow"
    parquet_engine = "pyarrow"

    @property
    def _constructor(self) -> type[DataFrame]:
        return self.__class__

    @property
    def _constructor_sliced(self) -> type[Series]:
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

    @singledispatchmethod
    def to_gdrive(self, target: GoogleDriveFile, *args: Any, **kwargs: Any) -> None:
        """Write to a :class:`pydrive2.files.GoogleDriveFile`.

        The file must define proper MIME type and file extension.
        """
        self.check_gdrive_file(target)
        ext = str(target["fileExtension"]).removeprefix(".")
        writer = self._get_writer(ext)
        buffer = io.BytesIO()
        writer(buffer, *args, **kwargs)
        target.content = buffer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            target.Upload()

    @to_gdrive.register
    def _(self, target: GoogleDrive, id: str, *args: Any, **kwargs: Any) -> None:
        """Write to :class:`pydrive2.drives.GoogleDrive` file by ``id``."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            file = target.CreateFile({"id": id})
        self.to_gdrive(file, *args, **kwargs)

    @staticmethod
    def check_gdrive_file(target: GoogleDriveFile) -> None:
        """Check if a :class:`pydrive2.files.GoogleDriveFile` defines MIME and extension."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        self.to_gdrive(target, *args, **kwargs)

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
        return cls.from_gdrive(source, *args, **kwargs)

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

    @overload
    @classmethod
    def from_json(cls, *args: Any, chunksize: None, **kwargs: Any) -> Self:
        ...

    @overload
    @classmethod
    def from_json(cls, *args: Any, chunksize: int, **kwargs: Any) -> JsonReader:
        ...

    @classmethod
    def from_json(cls, *args: Any, chunksize: int | None = None, **kwargs: Any):
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
        kwargs = cls._make_read_kwargs(
            **{"convert_dates": False, "chunksize": chunksize, **kwargs}
        )
        output = pd.read_json(*args, **kwargs)
        if isinstance(output, pd.DataFrame):
            return cls(output)
        return JsonReader._from_json_reader(output)

    @overload
    @classmethod
    def from_jsonl(cls, *args: Any, chunksize: None, **kwargs: Any) -> Self:
        ...

    @overload
    @classmethod
    def from_jsonl(cls, *args: Any, chunksize: int, **kwargs: Any) -> JsonReader:
        ...

    @classmethod
    def from_jsonl(cls, *args: Any, chunksize: int | None = None, **kwargs: Any):
        """See :func:`pandas.read_json` with `lines=True`.

        Examples
        --------
        >>> import io
        >>> buff = io.StringIO('{"a": 1, "b": 2}\\n{"a": 2, "b": 3}')
        >>> DataFrame.from_jsonl(buff)
           a  b
        0  1  2
        1  2  3

        This method supports also incremental reading in chunks
        by passing `chunksize` argument.
        >>> buff = io.StringIO('{"a": 1, "b": 2}\\n{"a": 2, "b": 3}')
        >>> for chunk in DataFrame.from_jsonl(buff, chunksize=1):
        ...     print(chunk)
           a  b
        0  1  2
           a  b
        1  2  3

        And the returned chunks are :class:`newsuse.data.DataFrame` instances.
        >>> all(
        ...     isinstance(chunk, DataFrame)
        ...     for chunk in DataFrame.from_jsonl(buff, chunksize=1)
        ... )
        True
        """
        return cls.from_json(*args, lines=True, chunksize=chunksize, **kwargs)

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

    @singledispatchmethod
    @classmethod
    def from_gdrive(cls, source: GoogleDriveFile, *args: Any, **kwargs: Any) -> Self:
        """Construct from :class:`GoogleDriveFile` instance.

        ``*args`` and ``**kwargs`` are passed to an appropriate format reader.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if not source.metadata:
                source.FetchMetadata()
            reader = cls._get_reader(source["fileExtension"])
            buffer = io.BytesIO(source.GetContentIOBuffer().read())
        return reader(buffer, *args, **kwargs)

    @from_gdrive.register
    @classmethod
    def _(cls, source: GoogleDrive, id: str, *args: Any, **kwargs: Any) -> Self:
        """Construct from :class:`GoogleDrive`.

        ``id`` is the Google Drive file id.
        ``*args`` and ``*kwargs`` are passed to an appropriate format reader.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            file = source.CreateFile({"id": id})
        return cls.from_gdrive(file, *args, **kwargs)

    def mode(self, axis, **kwargs: Any) -> pd.DataFrame:
        return self.apply(Series.mode, axis=axis, **kwargs)

    @classmethod
    def iter_sources(cls, *sources: Any) -> Iterator[Any]:
        """Iterate over glob-expanded ``*sources``.

        Parameters
        ----------
        *sources
            Data sources of any supported type.
            Path-like sources with glob patterns will be expanded.
        """
        for source in sources:
            if isinstance(source, str | Path):
                source = Path(source)
                if source.is_absolute():
                    _sources = Path("/").glob(str(source)[1:])
                else:
                    _sources = Path.cwd().glob(str(source))
            else:
                _sources = [source]
            yield from _sources

    @classmethod
    def read_many(
        cls,
        *sources: Any,
        progress: Mapping[str, Any] | bool = False,
    ) -> Iterator[tuple[Any, Self]]:
        """Iterate over and read multiple data files.

        Parameters
        ----------
        *sources
            Data sources of any supported type.
            Tuples and mappings are interpreted as ``*args`` and ``*kwargs`` respectively.
            Path-like sources may inclue glob patterns.
            In all cases sources will be read using :meth:`from_` method.
        progress
            Should the iterator be wrapped in progress bar.
            Can be passed as bool or as a mapping with :func:`tqdm.tqdm` options.

        Yields
        ------
        source
            Expanded source.
        df
            Data frame read from the ``source``.
        """
        _sources = list(cls.iter_sources(*sources))

        def _iter():
            for source in _sources:
                if isinstance(source, tuple):
                    yield source, cls.from_(*source)
                elif isinstance(source, Mapping):
                    yield source, cls.from_(**source)
                else:
                    yield source, cls.from_(source)

        data = _iter()
        if progress:
            tqdm_kwargs = {} if isinstance(progress, bool) else progress
            data = tqdm(data, **{"total": len(_sources), **tqdm_kwargs})
        yield from data

    @classmethod
    def from_many(
        cls,
        *sources: Any,
        progress: Mapping[str, Any] | bool = False,
        **kwargs: Any,
    ) -> Self:
        """Build data frame from many sources.

        See :meth:`read_many` for details.
        Additional ``**kwargs`` are passed to :func:`pandas.concat`
        (by default ``axis=0`` and ``ignore_index=True`` are used).
        """
        kwargs = {"axis": 0, "ignore_index": True, **kwargs}
        return pd.concat(
            (df for _, df in cls.read_many(*sources, progress=progress)), **kwargs
        )


class JsonReader(pd.io.json._json.JsonReader):
    """Simple wrapper around :class:`pandas.io.json._json.JsonReader`
    to allow for iterating over JSON lines.
    """

    @classmethod
    def _from_json_reader(cls, reader: pd.io.json._json.JsonReader) -> Self:
        new_reader = cls.__new__(cls)
        new_reader.__dict__.update(reader.__dict__)
        return new_reader

    def read(self) -> DataFrame | Series:
        obj: DataFrame | Series
        obj = super().read()
        if isinstance(obj, pd.DataFrame):
            return DataFrame(obj)
        return Series(obj)
