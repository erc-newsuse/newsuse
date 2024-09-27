from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import datasets
from datasets import load_from_disk
from datasets.features import Features

from newsuse.annotations import Annotations
from newsuse.types import PathLike
from newsuse.utils import inthash

__all__ = ("Dataset", "DatasetDict")


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
        ann = annotations.with_labels(**labels)
        dataset = cls.from_pandas(ann)
        for name in ("label", "human", *labels):
            dataset = dataset.class_encode_column(name)
        return cls.from_dataset(dataset)

    def train_test_split(self, *args: Any, **kwargs: Any) -> Self:
        seed = kwargs.pop("seed", None)
        if "key" in self.column_names:
            seed = seed or 0
            seed += inthash(tuple(self["key"]))
        return super().train_test_split(*args, seed=seed, **kwargs)

    def save_to_disk(self, dataset_dict_path: PathLike, *args: Any, **kwargs: Any) -> None:
        try:
            path = Path(dataset_dict_path)
            if path.exists():
                rmtree(path)
        except Exception:  # noqa
            pass
        super().save_to_disk(dataset_dict_path, *args, **kwargs)

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
