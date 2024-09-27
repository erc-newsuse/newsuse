from collections.abc import Iterable, Sequence
from functools import singledispatch
from inspect import get_annotations
from typing import Any, get_args

import pandas as pd
import tiktoken

from newsuse.llm.gpt import GptClassifier

__all__ = ("count_tokens", "estimate_cost")


@singledispatch
def count_tokens(input, *args: Any, **kwargs: Any) -> dict[str, int]:  # noqa
    errmsg = f"cannot count tokens for '{type(input)}'"
    raise NotImplementedError(errmsg)


@count_tokens.register
def _(
    input: GptClassifier,
    examples: int | Iterable[str],
) -> dict[str, int]:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    labels = get_args(get_annotations(input.response_format)["label"])
    otoks = sum(len(tokenizer.encode(label)) for label in labels)
    if labels:
        otoks /= len(labels)
    itoks = len(tokenizer.encode(input.prompt))

    if isinstance(examples, int):
        otoks *= examples
        itoks *= examples
    else:
        otoks *= len(examples)
        itoks *= len(examples)
        itoks += sum(len(tokenizer.encode(t)) for t in examples)

    return {"input_tokens": int(itoks), "output_tokens": int(otoks)}


@singledispatch
def estimate_cost(model, corpus: Sequence[str], *args: Any, **kwargs: Any) -> float:  # noqa
    """Estimate cost of processing ``corpus`` with ``model`` by sampling.

    ``*args`` and ``**kwargs`` are passed :meth:`pandas.Series.sample`.
    """
    errmsg = f"cannot estimate cost for '{type(model)}'"
    raise NotImplementedError(errmsg)


@estimate_cost.register
def _(model: GptClassifier, corpus: Sequence[str], *args: Any, **kwargs: Any) -> float:
    corpus = pd.Series(corpus)  # type: ignore
    sample = corpus.sample(*args, **kwargs)
    tokens = count_tokens(model, sample)
    cost = model.calculate_cost(**tokens) / len(sample)
    return cost * len(corpus)
