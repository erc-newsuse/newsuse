import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.resources.beta.chat.completions import Completions
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from newsuse.config import Config
from newsuse.types import PathLike

__all__ = ("ChatGpt",)


class ChatGpt:
    def __init__(
        self,
        model: str,
        *,
        context: Iterable[dict[str, str]] = (),
        response_format: type[BaseModel] | None = None,
        client: OpenAI | None = None,
        pricing: Config | None = None,
        **params: Any,
    ) -> None:
        self.model = model.strip().lower()
        self.context = list(context)
        self.response_format = response_format
        self.params = params
        self.client = client or OpenAI()
        self.pricing = pricing or Config()["openai.pricing"]
        self._meta = {"model": self.model}
        self._usage = []

    @property
    def completions(self) -> Completions:
        return self.client.beta.chat.completions

    @property
    def meta(self) -> dict[str, Any]:
        return {
            **self._meta,
            "context": self.context,
            **self.params,
            "response_format": self.response_format.model_json_schema(),
            "usage": self.usage,
            "cost": self.cost,
        }

    @property
    def usage(self) -> dict[str, Any]:
        input_toks = 0
        output_toks = 0
        for u in self._usage:
            input_toks += u["prompt_tokens"]
            output_toks += u["completion_tokens"]
        return {
            "input_tokens": input_toks,
            "output_tokens": output_toks,
        }

    @property
    def cost(self) -> float:
        return self.calculate_cost(**self.usage)

    def get_completion_spec(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [*self.context, {"role": "user", "content": text}],
            "response_format": self.response_format,
            **self.params,
        }

    def get_completion(
        self,
        text: str,
        *,
        update_context: bool | None = None,
    ) -> ChatCompletion:
        spec = self.get_completion_spec(text)
        response = self.completions.parse(**spec)

        if not self._usage:
            self._meta["model"] = response.model
            self._meta["system_fingerprint"] = response.system_fingerprint
        self._usage.append(response.usage.to_dict())

        if update_context:
            self.update_context(text, response)
        elif update_context is None:
            try:
                self.update_context(text, response)
            except NotImplementedError:
                pass

        return response

    def update_context(self, text: str, response: type[BaseModel]) -> None:
        raise NotImplementedError

    def save_meta(self, dest: PathLike, *, overwrite: bool = False) -> None:
        dest = Path(dest)
        if overwrite:
            dest.unlink(missing_ok=True)
        mode = "a" if dest.exists() else "w"
        with dest.open(mode) as fh:
            meta = {"timestamp": str(datetime.now(UTC)), **self.meta}
            fh.write(json.dumps(meta).strip() + "\n")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        prices = self.pricing[f"{self.model}.standard"]
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1e6
        return cost
