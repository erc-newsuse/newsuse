from collections.abc import Iterable
from functools import singledispatchmethod
from typing import Any

import pandas as pd
from datasets import Dataset
from evaluate import CombinedEvaluations, Evaluator, Metric, combine, evaluator
from transformers import Pipeline, TextClassificationPipeline

__all__ = ("Performance",)


class Performance:
    def __init__(
        self,
        pipeline: Pipeline,
        *,
        metrics: str | Iterable[str] | CombinedEvaluations | Metric | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.evaluator = self._get_evaluator(pipeline)
        if isinstance(metrics, Metric | CombinedEvaluations):
            self.metrics = metrics
        else:
            self.metrics = self._get_metrics(pipeline, metrics)

    @singledispatchmethod
    def __call__(self, dataset, **kwargs: Any) -> pd.Series:  # noqa
        self._raise_not_implemented(dataset)

    @__call__.register
    def _(self, dataset: Dataset, **kwargs: Any) -> pd.Series:
        return self._compute(self.pipeline, dataset, **kwargs)

    @__call__.register
    def _(self, dataset: pd.DataFrame, **kwargs: Any) -> pd.Series:
        return self(Dataset.from_pandas(dataset), **kwargs)

    @singledispatchmethod
    def _get_default_metrics(self, pipeline) -> list[str]:
        self._raise_not_implemented(pipeline)

    @_get_default_metrics.register
    def _(self, pipeline: TextClassificationPipeline) -> list[str]:  # noqa
        return ["accuracy", "f1", "precision", "recall"]

    @singledispatchmethod
    def _get_evaluator(self, pipeline) -> Evaluator:
        self._raise_not_implemented(pipeline)

    @_get_evaluator.register
    def _(self, pipeline: TextClassificationPipeline) -> Evaluator:  # noqa
        return evaluator("text-classification")

    @staticmethod
    def _raise_not_implemented(obj):
        errmsg = f"evaluation for '{type(obj)}' is not yet implemented"
        raise NotImplementedError(errmsg)

    @singledispatchmethod
    def _compute(self, pipeline, dataset: Dataset, **kwargs: Any) -> pd.Series:  # noqa
        self._raise_not_implemented(pipeline)

    @_compute.register
    def _(
        self, pipeline: TextClassificationPipeline, dataset: Dataset, **kwargs: Any
    ) -> pd.Series:
        return pd.Series(
            self.evaluator.compute(
                metric=self.metrics,
                model_or_pipeline=pipeline,
                data=dataset,
                label_mapping=self.pipeline.model.config.label2id,
                **kwargs,
            )
        )

    @staticmethod
    def _sanitize_metrics(metrics, defaults) -> list[str]:
        if metrics is None:
            metrics = defaults
        if isinstance(metrics, str):
            metrics = [metrics]
        return list(metrics)

    def _get_metrics(self, pipeline, metrics) -> CombinedEvaluations:
        metrics = self._sanitize_metrics(metrics, self._get_default_metrics(pipeline))
        return combine(metrics)
