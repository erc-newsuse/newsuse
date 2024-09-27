from collections.abc import Callable, Iterable
from pathlib import Path
from shutil import rmtree
from typing import Any

import evaluate
import numpy as np
import transformers
from transformers import EvalPrediction
from transformers.trainer import TrainOutput

from newsuse.types import PathLike


class Trainer(transformers.Trainer):
    def __init__(
        self,
        *args: Any,
        compute_metrics: str
        | Iterable[str]
        | Callable[[EvalPrediction], dict]
        | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compute_metrics is None or isinstance(compute_metrics, Callable):
            self.compute_metrics = compute_metrics
        else:
            self.compute_metrics = self._get_compute_metrics(compute_metrics)

    def _get_compute_metrics(
        self, metrics: str | Iterable[str]
    ) -> Callable[[EvalPrediction], dict]:
        is_classifier = hasattr(self.model.config, "num_labels")

        if not is_classifier:
            errmsg = (
                "setting 'compute_metrics' using metric names "
                f"is not implemented for '{type(self.model)}'"
            )
            raise NotImplementedError(errmsg)

        if isinstance(metrics, str):
            metrics = [metrics]
        performance = evaluate.combine(list(metrics))

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return performance.compute(predictions=predictions, references=labels)

        return compute_metrics

    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
        outdir = Path(self.args.output_dir)
        if outdir.exists():
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
