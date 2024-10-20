from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import singledispatchmethod
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import datasets
import numpy as np
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name, split in self.items():
            self[name] = split

    def __setitem__(self, name: str, value: "Dataset") -> None:
        if isinstance(value, datasets.Dataset) and not isinstance(value, Dataset):
            value = Dataset.from_dataset(value)
        super().__setitem__(name, value)

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

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().filter(*args, **kwargs))

    def select(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().select(*args, **kwargs))

    def map(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().map(*args, **kwargs))

    def class_encode_column(self, *args: Any, **kwargs) -> Self:
        return self.__class__(super().class_encode_columns(*args, **kwargs))

    def rename_column(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().rename_column(*args, **kwargs))

    def rename_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().rename_columns(*args, **kwargs))

    def sample(
        self,
        size: int | float | Mapping[str, int | float],  # noqa
        *,
        seed: int | np.random.Generator | None = None,
        hash_key: str = "key",
        **kwargs: Any,
    ) -> Self:
        if not isinstance(seed, np.random.Generator):
            if hash_key and hash_key in self.column_names:
                seed = seed or 0
                seed += inthash(tuple(self[hash_key]))
            rng = np.random.default_rng(seed)
        else:
            rng = seed
        if not isinstance(size, Mapping):
            size = {n: size for n in self}
        dsets = {self[n].sample(s, seed=rng, **kwargs) for n, s in size.items()}
        return self.__class__(dsets)

    def select_training_columns(self, *args: Any, **kwargs: Any) -> Self:
        """Select columns for training."""
        return self.__class__(
            {
                name: split.select_training_columns(*args, **kwargs)
                for name, split in self.items()
            }
        )


class Dataset(datasets.Dataset):
    @classmethod
    def from_annotations(
        cls,
        annotations: Annotations,
        *,
        top_n: int | None = None,
        seed: int = 0,
        metadata: str | Iterable[str] = (),
    ) -> Self:
        """Construct from instances of :class:`newsuse.annotations.Annotations`.

        Parameters
        ----------
        top_n
            Use only ``top_n`` observations with greatest number of annotations
            per sheet (randomly shuffled).
        metadata
            Column with additional metadata to retain.
        """
        data = annotations.data

        if top_n:
            seed = (seed + inthash(data["key"])) % (2**32 - 1)
            n_annotations = data[annotations.annotator_cols].notnull().sum(axis=1)
            data = (
                data.assign(n_annotations=n_annotations)
                .groupby(annotations.config.sheet_index_name)
                .sample(frac=1, replace=False, random_state=seed)
                .sort_values("n_annotations", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        if isinstance(metadata, str):
            metadata = [metadata]
        data = annotations.data[["key", *metadata, "human", "text"]].dropna(
            ignore_index=True
        )
        dataset = (
            cls.from_pandas(data)
            .rename_column("human", "label")
            .class_encode_column("label", names=annotations.config.labels)
        )
        return cls.from_dataset(dataset)

    def train_test_split(self, *args: Any, **kwargs: Any) -> Self:
        seed = kwargs.pop("seed", None)
        if "key" in self.column_names:
            seed = seed or 0
            seed += inthash(tuple(self["key"]))
        return super().train_test_split(*args, seed=seed, **kwargs)

    @singledispatchmethod
    def make_splits(
        self,
        split: Mapping[str, float | int],
        *,
        seed: int | None = None,
        default_split_name: str = "train",
        **kwargs: Any,
    ) -> DatasetDict[str, Self]:
        """Split dataset."""
        split = dict(split)
        n_examples = len(self)

        for k, v in split.items():
            if isinstance(v, float):
                split[k] = int(ceil(v * n_examples))

        n_in_split = sum(split.values())
        if n_in_split > n_examples:
            errmsg = "cannot define splits with more examples than the size of the dataset"
            raise ValueError(errmsg)
        if n_in_split < n_examples:
            if default_split_name in split:
                errmsg = f"default split name '{default_split_name}' is already defined"
                raise ValueError(errmsg)
            split[default_split_name] = n_examples - n_in_split

        seed = seed or 0
        seed += inthash(tuple(self["key"]))
        kwargs = {"seed": seed, "keep_in_memory": True, **kwargs}
        data = self.shuffle(**kwargs)
        dset = {}
        start = 0
        for name, n in split.items():
            dset[name] = data.select(range(start, start + n), keep_in_memory=True)
            start += n
        return DatasetDict(dset)

    @make_splits.register
    def _(
        self, split: tuple, *splits: tuple[str, float | int], **kwargs: Any
    ) -> DatasetDict[str, Self]:
        dct = dict([split, *splits])
        return self.split(dct, **kwargs)

    def concat(self, *others: Self, **kwargs: Any) -> Self:
        dataset = datasets.concatenate_datasets([self, *others], **kwargs)
        return self.from_dataset(dataset)

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
        return cls(
            dataset.data,
            dataset.info,
            dataset.split,
            dataset._indices,
            dataset._fingerprint,
        )

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

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().filter(*args, **kwargs))

    def select(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().select(*args, **kwargs))

    def map(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().map(*args, **kwargs))

    def select_columns(self, *args: Any, **kwargs: Any) -> Self:
        dataset = super().select_columns(*args, **kwargs)
        return self.from_dataset(dataset)

    def rename_column(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().rename_column(*args, **kwargs))

    def rename_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().rename_columns(*args, **kwargs))

    def class_encode_column(
        self, column: str, names: Iterable[str] | None = None, *args: Any, **kwargs: Any
    ) -> Self:
        dataset = super().class_encode_column(column, *args, **kwargs)
        if names:
            names = list(names)
            if set(names) != set(dataset.features[column].names):
                errmsg = "labels schema is not consistent with observed labels"
                raise ValueError(errmsg)
            map_old = dict(enumerate(dataset.features[column].names))
            map_new = {label: id for id, label in enumerate(names)}

            def remap(example):
                if not (value := example.get(column)):
                    return example
                example = example.copy()
                example[column] = map_new[map_old[value]]
                return example

            dataset = dataset.map(remap)
            dataset.info.features[column].names = names  # type: ignore
        return self.from_dataset(dataset)

    @singledispatchmethod
    def sample(
        self,
        size,
        *,
        seed: int | np.random.Generator | None = None,
        hash_key: str | None = "key",
        **kwargs: Any,
    ) -> Self:
        if size % 1 != 0:
            errmsg = f"'size' must be an integer, not {size}"
            raise ValueError(errmsg)
        size = int(size)
        if size <= 0:
            errmsg = f"size has to be positive, not {size}"
            raise ValueError(errmsg)
        if not isinstance(seed, np.random.Generator):
            if hash_key and hash_key in self.column_names:
                seed = seed or 0
                seed += inthash(tuple(self[hash_key]))
            rng = np.random.default_rng(seed)
        else:
            rng = seed
        idx = np.arange(len(self))
        rng.shuffle(idx)
        idx = idx[:size]
        return self.select(idx, **kwargs)

    @sample.register
    def _(self, size: float, **kwargs: Any) -> Self:
        if size > 1:
            errmsg = "'size' cannot exceed '1.0' fraction of the examples in the dataset"
            raise ValueError(errmsg)
        size = int(ceil(len(self)))
        return self.sample(size, **kwargs)

    def add_balancing_weights(self, *fields: str) -> Self:
        """Add weights balancing examples in groups given by ``*field``."""
        n = self.to_pandas().groupby(list(fields)).size()
        target = len(self) / len(n)
        w = target / n
        df = self.to_pandas()
        if not isinstance(df, pd.DataFrame):
            df = pd.concat(list(df), axis=0, ignore_index=True)
        df["weight"] = w.loc[df[list(fields)].apply(tuple, axis=1)].reset_index(drop=True)
        df["weight"] *= len(df) / df["weight"].sum()
        return self.__class__.from_pandas(df)

    def select_training_columns(
        self, *optional_cols: str, data: str = "text", target: str = "label"
    ) -> Self:
        """Select columns for training."""
        usecols = [data, target]
        usecols.extend(c for c in optional_cols if c in self.column_names)
        return self.select_columns(usecols)


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
