from typing import Any

import transformers
from datasets.features import ClassLabel

from newsuse.types import PathLike


class AutoModelForSequenceClassification(transformers.AutoModelForSequenceClassification):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike,
        *model_args: Any,
        target: ClassLabel | None = None,
        **kwargs: Any,
    ):
        extra_kwargs = {}
        if target is not None:
            extra_kwargs.update(
                num_labels=target.num_classes,
                id2label=dict(enumerate(target.names)),
                label2id={n: i for i, n in enumerate(target.names)},
            )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **extra_kwargs, **kwargs
        )
