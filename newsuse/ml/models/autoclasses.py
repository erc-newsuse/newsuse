from collections.abc import Callable, Hashable, Iterable
from functools import singledispatchmethod
from typing import Any, Self

import transformers
from transformers import PretrainedConfig, PreTrainedModel

from newsuse.types import PathLike

from ..trainer import TrainingMixin

__all__ = ("AutoModelForSequenceClassification",)


_LabelsT = int | Iterable[Hashable]


def _make_labels_kwargs(labels: _LabelsT) -> dict[str, Any]:
    if isinstance(labels, int):
        num_labels = labels
        labels = list(range(num_labels))
    else:
        labels = list(labels)
        num_labels = len(labels)
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    kwargs = {
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }
    return kwargs


class AutoModelForSequenceClassification(
    transformers.AutoModelForSequenceClassification, TrainingMixin
):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike,
        *args: Any,
        labels: _LabelsT | None = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        if labels:
            kwargs = {**kwargs, **_make_labels_kwargs(labels)}
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        *args: Any,
        labels: _LabelsT | None = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        if labels:
            label_kwargs = _make_labels_kwargs(labels)
            config.labels = labels
            for name, value in label_kwargs.items():
                setattr(config, name, value)
        return super().from_config(config, *args, **kwargs)

    @singledispatchmethod
    @classmethod
    def factory(cls, name_or_path: str, *args: Any, **kwargs: Any) -> Callable[..., Self]:
        """Get ``model_init()`` factory function."""

        def model_init():
            return cls.from_pretrained(name_or_path, *args, **kwargs)

        return model_init

    @factory.register
    @classmethod
    def _(cls, config: PretrainedConfig, *args: Any, **kwargs: Any) -> Callable[..., Self]:
        def model_init():
            return cls.from_config(config, *args, **kwargs)

        return model_init
