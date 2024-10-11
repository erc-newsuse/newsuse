from collections.abc import Callable, Iterable, Sequence
from functools import singledispatchmethod
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import datasets
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from datasets import load_from_disk
from datasets.features import Features
from transformers.pipelines.pt_utils import KeyDataset as _KeyDataset

from newsuse.annotations import Annotations
from newsuse.types import PathLike
from newsuse.utils import inthash

__all__ = ("Dataset", "DatasetDict", "SimpleDataset", "KeyDataset")


class DatasetDict(datasets.DatasetDict):
    @property
    def features(self) -> Features:
        try:
            first = list(self)[0]
            return self[first].features
        except IndexError:
            return Features()

    def tokenize(self, *args: Any, **kwargs: Any) -> Self:
        """See :meth:`Dataset.tokenize`."""

        def _tokenize(dataset):
            if isinstance(dataset, datasets.Dataset):
                dataset = Dataset.from_dataset(dataset)
            return dataset.tokenize(*args, **kwargs)

        return self.__class__({s: _tokenize(d) for s, d in self.items()})

    def select_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().select_columns(*args, **kwargs))


class Dataset(datasets.Dataset):
    @classmethod
    def from_annotations(cls, annotations: Annotations, **labels: PathLike) -> Self:
        """Construct from instances of :class:`newsuse.annotations.Annotations`.

        Parameters
        ----------
        **labels
            Label names and paths to files with additional labels.
            Passed to :meth:`newsuse.annotations.Annotations.with_labels`.
        """
        label_columns = ("label", "human", *labels)
        ann = annotations.remove_notes().with_labels(**labels).data

        dataset = cls.from_pandas(ann)
        for name in label_columns:
            dataset = dataset.class_encode_column(name)
        return cls.from_dataset(dataset)

    def train_test_split(self, *args: Any, **kwargs: Any) -> Self:
        seed = kwargs.pop("seed", None)
        if "key" in self.column_names:
            seed = seed or 0
            seed += inthash(tuple(self["key"]))
        return super().train_test_split(*args, seed=seed, **kwargs)

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        *args: Any,
        clear_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        path = Path(dataset_dict_path)
        if path.exists():
            rmtree(path)
        super().save_to_disk(dataset_dict_path, *args, **kwargs)
        if clear_cache:
            for cached in path.rglob("cache-*.arrow"):
                cached.unlink()

    @classmethod
    def from_dataset(cls, dataset: datasets.Dataset) -> Self | DatasetDict:
        return cls(dataset.data)

    @classmethod
    def from_disk(cls, *args: Any, **kwargs: Any) -> Self | DatasetDict:
        """Load using :func:`datasets.load_from_disk`."""
        dataset = load_from_disk(*args, **kwargs)
        if isinstance(dataset, datasets.DatasetDict):
            return DatasetDict(dataset)
        return cls.from_dataset(dataset)

    def tokenize(
        self,
        tokenizer: Callable[[str, ...], list[int]],
        text_field: str = "text",
        *,
        padding: str = "max_length",
        truncation: bool = True,
        batched: bool = True,
        **kwargs: Any,
    ) -> Self:
        def tokenize(example):
            return tokenizer(
                example[text_field], padding=padding, truncation=truncation, **kwargs
            )

        dataset = self.map(tokenize, batched=batched)
        return self.from_dataset(dataset)

    def select_columns(self, *args: Any, **kwargs: Any) -> Self:
        dataset = super().select_columns(*args, **kwargs)
        return self.from_dataset(dataset)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple indexable :mod:`torch` dataset
    fetching examples from elements of ``self.data``.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [1,2,3]})
    >>> dataset = SimpleDataset(df)
    >>> dataset[0]
    {'a': 1, 'b': 1}
    >>> dataset[:2]
    {'a': [1, 2], 'b': [1, 2]}

    It is compatible with :class:`newsuse.ml.KeyDaset`.

    >>> keyed = KeyDataset(dataset, "a")
    >>> keyed[0]
    1
    >>> keyed[:2]
    [1, 2]
    """

    _ExampleT = dict[str, Any]

    def __init__(
        self, data: Sequence[_ExampleT] | Iterable[_ExampleT] | pd.DataFrame
    ) -> None:
        if not isinstance(data, Sequence | pd.DataFrame):
            data = list(data)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> _ExampleT:
        return self._get(self.data, idx)

    @singledispatchmethod
    def _get(self, data, idx: int) -> _ExampleT:
        return data[idx]

    @_get.register
    def _(self, data: pd.Series, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            out = out.tolist()
        return out

    @_get.register
    def _(self, data: pd.DataFrame, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            return out.to_dict(orient="list")
        return out.to_dict()


class KeyDataset(_KeyDataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset | Sequence | pd.DataFrame, key: str
    ) -> None:
        if not isinstance(dataset, torch.utils.data.Dataset):
            dataset = SimpleDataset(dataset)
        super().__init__(dataset, key)
