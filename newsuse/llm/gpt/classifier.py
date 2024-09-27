from collections.abc import Iterable, Iterator
from functools import singledispatchmethod
from typing import Any

from tqdm.auto import tqdm

from .formats import ClassificationResponse
from .gpt import ChatGpt

__all__ = ("GptClassifier",)


class GptClassifier(ChatGpt):
    def __init__(
        self,
        prompt: str,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        response_format: ClassificationResponse,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            temperature=temperature,
            response_format=response_format,
            context=({"role": "system", "content": prompt},),
            **kwargs,
        )

    @property
    def prompt(self) -> str:
        return self.context[0]["content"]

    @singledispatchmethod
    def __call__(self, text, **kwargs: Any) -> str:
        yield from (obj.label for obj in self.apply(text, **kwargs))

    @__call__.register
    def _(self, text: str) -> str:
        return self.apply(text).label

    @singledispatchmethod
    def apply(
        self,
        text,
        **kwargs: Any,  # noqa
    ) -> ClassificationResponse | Iterable[ClassificationResponse]:
        errmsg = f"cannot process '{type(text)}' instances"
        raise NotImplementedError(errmsg)

    @apply.register
    def _(self, text: str) -> ClassificationResponse:
        completion = self.get_completion(text)
        obj = completion.choices[0].message.parsed
        return obj

    @apply.register
    def _(
        self,
        text: Iterable,
        *,
        progress: bool = False,
        **kwargs: Any,
    ) -> Iterator[ClassificationResponse]:
        kwargs = {"disable": not progress or kwargs, **kwargs}
        yield from map(self.apply, tqdm(text, **kwargs))
