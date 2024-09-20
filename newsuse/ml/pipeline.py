from collections.abc import Iterable, Mapping
from functools import singledispatchmethod
from typing import Any

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from newsuse.data import DataFrame


class Pipeline:
    """Pipeline class for applying transformer models to data.

    Attributes
    ----------
    task
        Label of inference task.
    pipeline
        Instance of :class:`transformers.pipelines.Pipeline`.
    """

    def __init__(self, task: str, *args, **kwargs) -> None:
        """Initialization method.

        All arguments are passed to :func:`transormers.pipeline`.
        """
        if self.is_text_classification_task(task):
            kwargs = {"top_k": None, **kwargs}
        else:
            errmsg = f"'{task}' task is not yet supported"
            raise NotImplementedError(errmsg)
        self.pipeline = pipeline(task, *args, **kwargs)

    def __call__(self, data, *args, **kwargs) -> Any:
        return self.apply(data, *args, **kwargs)

    @property
    def task(self) -> str:
        return self.pipeline.task

    @property
    def is_text_classifier(self) -> bool:
        return self.is_text_classification_task(self.task)

    @singledispatchmethod
    def apply(self, data, *args, **kwargs):
        return self.pipeline(data, *args, **kwargs)

    @apply.register
    def _(
        self,
        data: Dataset,
        field: str,
        *,
        key: str | None = None,
        batch_size: int = 8,
        progress: bool | None = None,
        tqdm_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        progress = len(data) >= 100 if progress is None else progress
        tqdm_kwargs = {"disable": not progress, "total": len(data), **(tqdm_kwargs or {})}
        dataset = KeyDataset(data, field)
        kwargs = self.get_kwargs(batch_size=batch_size)
        stream = tqdm(self.pipeline(dataset, **kwargs), **tqdm_kwargs)
        output = DataFrame(list(map(self.postprocess, stream)))
        if key:
            output.insert(0, key, data[key])
        return output

    @apply.register
    def _(
        self, data: pd.DataFrame, field: str, *, key: str | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        cols = [key, field] if key else [field]
        dataset = Dataset.from_pandas(data[cols])
        return self.apply(dataset, field, key=key, **kwargs)

    def get_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        if self.is_text_classifier:
            return {"padding": True, "truncation": True, **kwargs}
        return kwargs

    def postprocess(self, record: Any) -> Any:
        if self.is_text_classifier:
            if isinstance(record, Mapping):
                pass
            elif isinstance(record, Iterable):
                record = {r["label"]: r["score"] for r in record}
            else:
                errmsg = (
                    f"cannot postprocess '{type(record)}' instances "
                    f"for '{self.task}' task"
                )
                raise ValueError(errmsg)
        return record

    @staticmethod
    def is_text_classification_task(task: str) -> bool:
        return task in ("text-classification", "sentiment-analysis")
