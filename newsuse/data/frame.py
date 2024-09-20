from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd

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

    def to_(self, target: Any, *args: Any, **kwargs: Any) -> None:
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
        stype = self.guess_storage_type(target)
        try:
            writer: Callable[..., None] = getattr(self, f"to_{stype}")
        except AttributeError as exc:
            errmsg = f"'{stype}' data storage is not supported"
            raise AttributeError(errmsg) from exc
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

    @classmethod
    def from_(cls, source: Any, *args: Any, **kwargs: Any) -> Self:
        """Guess storage type and try to use it to construct a data frame.

        Currently only path-like sources are supported.

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
        stype = cls.guess_storage_type(source)
        try:
            reader: Callable[..., Self] = getattr(cls, f"from_{stype}")
        except AttributeError as exc:
            errmsg = f"'{stype}' data sources are not supported"
            raise AttributeError(errmsg) from exc
        return reader(source, *args, **kwargs)

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
