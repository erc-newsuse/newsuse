from collections.abc import Callable, Mapping
from typing import Any, Literal, Self

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from newsuse.utils import validate_call

from .heads import SequenceClassificationHead, SequenceClassificationHeadConfig

_ProblemT = Literal[
    "regression", "single_label_classification", "multi_label_classification"
]
_LossFuncT = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


class SequenceClassifierTransformerConfig(PretrainedConfig):
    """Config for :class:`NewsuseSequenceClassifier`.

    Attributes
    ----------
    base
        Name or path of a pretrained base language model.
    base_config
        Config of the base model.
    head_config
        Config of the classification head.
    """

    model_type = "sequence-classifier-transformer"

    @validate_call
    def __init__(
        self,
        base: str | None = None,
        base_config: PretrainedConfig | Mapping | None = None,
        head_config: PretrainedConfig | Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        if not base and isinstance(base_config, Mapping):
            base = base_config.get("_name_or_path")
        if isinstance(base_config, PretrainedConfig):
            if base and base_config._name_or_path != base:
                errmsg = (
                    f"'base={base}' but 'base_config' "
                    f"is for '{base_config._name_or_path}'"
                )
                raise ValueError(errmsg)
        elif base:
            base_config = AutoConfig.from_pretrained(base, **(base_config or {}))
        self.base_config = base_config

        if not isinstance(head_config, SequenceClassificationHeadConfig):
            head_config = SequenceClassificationHeadConfig(**(head_config or {}))
        if isinstance(self.base_config, PretrainedConfig):
            if head_config.dim is None:
                head_config.set_dim(self.base_config.dim)
            if head_config.dropout is None:
                head_config.set_dropout(self.base_config.seq_classif_dropout)
            if head_config.initializer_range is None:
                head_config.initializer_range = self.base_config.initializer_range
        self.head_config = head_config
        super().__init__(**kwargs)
        self.head_config.num_labels = self.num_labels

    @property
    def base(self) -> str:
        return self.base_config._name_or_path


class SequenceClassifierTransformer(PreTrainedModel):
    config_class = SequenceClassifierTransformerConfig

    def __init__(self, config: config_class) -> None:
        super().__init__(config)
        self.base = AutoModel.from_pretrained(self.config.base)
        self.head = SequenceClassificationHead(self.config.head_config)
        self.loss_func = None
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        weight: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        # ruff: noqa: C901
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        weights = weight
        output = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self._get_pooled_output(output)
        logits = self.head(pooled_output)

        loss = None
        if labels is not None:
            self._set_problem_type(labels)
            if self.loss_func is None:
                self.loss_func = self._get_loss_function()
            loss = self.loss_func(logits, labels, weights)

        if not return_dict:
            output = (logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _get_pooled_output(self, output: BaseModelOutput) -> torch.Tensor:
        hidden_states = output[0]
        pooled_output = hidden_states[:, 0]
        return pooled_output

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.config.base)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any], **kwargs: Any) -> Self:
        """Construct from ``config_dict`` with optional additional ``**kwargs``."""
        config = cls.config_class(**config_dict, **kwargs)
        return cls(config)

    def _set_problem_type(self, labels: torch.Tensor) -> _ProblemT:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

    def _get_loss_function(self) -> _LossFuncT:
        if self.config.problem_type == "regression":

            def loss_fct_regression(
                logits: torch.Tensor,
                labels: torch.Tensor,
                weights: torch.Tensor | None = None,
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.MSELoss(reduction=reduction)
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_regression

        if self.config.problem_type == "single_label_classification":

            def loss_fct_single_label(
                logits: torch.Tensor,
                labels: torch.Tensor,
                weights: torch.Tensor | None = None,
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_single_label

        if self.config.problem_type == "multi_label_classification":

            def loss_fct_multi_label(
                logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.BCEWithLogitsLoss(reduction=reduction)
                loss = loss_fct(logits, labels)
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_multi_label

        errmsg = f"cannot define loss function for problem type {self.config.problem_type}"
        raise ValueError(errmsg)


# Register -------------------------------------------------------------------------------

AutoConfig.register(
    SequenceClassifierTransformerConfig.model_type, SequenceClassifierTransformerConfig
)
AutoModelForSequenceClassification.register(
    SequenceClassifierTransformerConfig, SequenceClassifierTransformer
)
