from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import transformers
from transformers.trainer import TrainOutput

from newsuse.types import PathLike

from .datasets import Dataset, DatasetDict
from .evaluation import Evaluation


class TrainingArguments(transformers.TrainingArguments):
    def __init__(
        self,
        *args: Any,
        train_use_sample_weights: bool | None = False,
        eval_use_sample_weights: bool | None = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_use_sample_weights = train_use_sample_weights
        self.eval_use_sample_weights = eval_use_sample_weights


class Trainer(transformers.Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.compute_metrics, Evaluation):
            self.compute_metrics = self.compute_metrics.get_evaluation_function()

    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
        outdir = Path(self.args.output_dir)
        if (key := "resume_from_checkpoint") in kwargs:
            resume_from_checkpoint = kwargs[key]
        elif resume_from_checkpoint := self.args.resume_from_checkpoint:
            kwargs[key] = self.args.resume_from_checkpoint
        if not resume_from_checkpoint and outdir.exists():
            for checkpoint in outdir.glob("checkpoint-*"):
                if checkpoint.is_file():
                    checkpoint.unlink()
                else:
                    rmtree(checkpoint)
        return super().train(*args, **kwargs)

    def save_model(self, output_dir: PathLike, *args: Any, **kwargs: Any) -> None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            for path in output_dir.glob("*"):
                if not path.name.startswith("checkpoint-"):
                    if path.is_file():
                        path.unlink()
                    else:
                        rmtree(path)
        super().save_model(output_dir, *args, **kwargs)

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        if self._train_use_sample_weights or self._eval_use_sample_weights:
            self._signature_columns = [*self._signature_columns, "weight"]

    @property
    def _train_use_sample_weights(self) -> bool:
        return self._should_use_weights("train_use_sample_weights")

    @property
    def _eval_use_sample_weights(self) -> bool:
        return self._should_use_weights("eval_train_sample_weights")

    def _should_use_weights(self, attr: str) -> bool:
        use_weights = getattr(self.args, attr, False)
        try:
            data_has_weights = "weight" in self.train_dataset.column_names
            if use_weights is None:
                use_weights = data_has_weights
        except AttributeError:
            use_weights = use_weights or False
        return use_weights


class TrainingMixin(ABC):
    @classmethod
    @abstractmethod
    def factory(cls, *args: Any, **kwargs: Any) -> Callable[..., Self]:
        """Get ``model_init()`` factory function."""

    @classmethod
    def preprocess_dataset(
        cls,
        dataset: Dataset | DatasetDict,
        tokenizer: transformers.PreTrainedTokenizerBase,
        *args: Any,  # noqa
        **kwargs: Any,  # noqa
    ) -> Dataset:
        """Preprocess ``dataset`` before training."""
        return dataset.tokenize(tokenizer)
