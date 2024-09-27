from collections.abc import Iterator, Mapping
from functools import singledispatchmethod, wraps
from typing import Any

import pandas as pd
import torch
import transformers
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.pt_utils import KeyDataset

__all__ = (
    "pipeline",
    "TextClassificationPipeline",
)


_OutputT = dict[str, float]


@wraps(transformers.pipeline)
def pipeline(*args: Any, **kwargs: Any) -> transformers.Pipeline:
    if kwargs.get("device") is None and torch.cuda.is_available():
        kwargs["device"] = "cuda"
    return transformers.pipeline(*args, **kwargs)


class TextClassificationPipeline(transformers.TextClassificationPipeline):
    """:class:`transformers.pipelines.TextClassificationPipeline`
    with automatic batch processing and support for :class:`pandas.DataFrame`s
    and single-dictionary outputs with probabilities for all labels.

    See also
    --------
    transformers.pipelines.TextClassificationPipeline : Parent pipeline class.
    """

    @singledispatchmethod
    def __call__(self, inputs, **kwargs: Any) -> _OutputT | list[_OutputT]:
        return super().__call__(inputs, **kwargs)

    @__call__.register
    def _(
        self,
        inputs: Dataset,
        field: str = "text",
        *,
        progress: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> Iterator[_OutputT]:
        if isinstance(progress, bool):
            tqdm_kwargs = {"disable": not progress}
        else:
            tqdm_kwargs = {"disable": not progress, **progress}
        try:
            tqdm_kwargs["total"] = len(inputs)
        except TypeError:
            pass
        dataset = KeyDataset(inputs, field)
        yield from tqdm(self(dataset, **kwargs), **tqdm_kwargs)

    @__call__.register
    def _(
        self,
        inputs: pd.DataFrame,
        field: str = "text",
        **kwargs: Any,
    ) -> Iterator[_OutputT]:
        dataset = Dataset.from_pandas(inputs[[field]])
        yield from self(dataset, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        outputs = super().postprocess(*args, **kwargs)
        if not isinstance(outputs, Mapping):
            if kwargs["top_k"] == 1:
                outputs = outputs[0]
            else:
                outputs = {o["label"]: o["score"] for o in outputs}
        return outputs

    def _sanitize_parameters(self, *args: Any, **kwargs: Any) -> tuple[dict, dict, dict]:
        preprocess, forward, postprocess = super()._sanitize_parameters(*args, **kwargs)
        preprocess = {"padding": True, "truncation": True}
        postprocess = {"top_k": 1, **postprocess}
        return preprocess, forward, postprocess


# Register pipelines ---------------------------------------------------------------------

PIPELINE_REGISTRY.register_pipeline(
    "text-classification",
    pipeline_class=TextClassificationPipeline,
    pt_model=transformers.AutoModelForSequenceClassification,
)

# ----------------------------------------------------------------------------------------
