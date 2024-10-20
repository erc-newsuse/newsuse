from collections.abc import Mapping
from typing import Any

import torch
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import get_activation

from newsuse.utils import validate_call

from ..feedforward import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
    _ActivationT,
    _NormStrategyT,
    _NormT,
)


class SequenceClassificationHeadConfig(PretrainedConfig):
    """Config for :class:`SequenceClassificationHead`.

    Attributes
    ----------
    dim
        Dimension of input embeddings.
    num_labels
        Number of output labels.
    ffn_config
        Config of feed forward network for transforming embeddings.
    activation
        Activation function to apply to embeddings.
    dropout
        Dropout to apply to embeddings.
    norm
        Normalization function to apply to embeddings.
    norm_strategy
        Strategy for combining dropout and normalization.
    """

    model_type = "sequence-classification-head"

    @validate_call
    def __init__(
        self,
        dim: PositiveInt | None = None,
        num_labels: PositiveInt | None = None,
        activation: _ActivationT | None = "relu",
        dropout: NonNegativeFloat | None = None,
        norm: _NormT | None = None,
        norm_strategy: _NormStrategyT = "standard",
        ffn_config: FeedForwardNetworkConfig | Mapping | None = None,
        initializer_range: PositiveFloat | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(ffn_config, FeedForwardNetworkConfig):
            ffn_config = FeedForwardNetworkConfig(**(ffn_config or {}))
        self.ffn_config = ffn_config
        self.activation = activation
        self.norm = norm
        self.norm_strategy = norm_strategy
        if num_labels is not None:
            kwargs = {"num_labels": num_labels, **kwargs}
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout = dropout
        self.initializer_range = initializer_range

    def set_dim(self, dim: int) -> None:
        self.dim = dim
        if self.ffn_config.dim is None or self.ffn_config.num_layers == 0:
            self.ffn_config.dim = self.dim

    def set_dropout(self, dropout: float) -> None:
        self.dropout = dropout
        if self.ffn_config.dropout is None:
            self.ffn_config.dropout = self.dropout


class SequenceClassificationHead(PreTrainedModel):
    """Head model for sequence classification based on embeddings."""

    config_class = SequenceClassificationHeadConfig

    def __init__(self, config: SequenceClassificationHeadConfig) -> None:
        super().__init__(config)
        self.norm = torch.nn.LayerNorm(self.config.dim) if self.config.norm else None
        dropoout = torch.nn.Dropout(self.config.dropout) if self.config.dropout else None
        if self.config.norm_strategy == "ic":
            self.dropout = dropoout
        self.preclassifier = torch.nn.Linear(self.config.dim, self.config.ffn_config.dim)
        self.activation = get_activation(self.config.activation)
        if self.config.norm_strategy != "ic":
            self.dropout = dropoout
        self.ffn = FeedForwardNetwork(self.config.ffn_config)
        self.classifier = torch.nn.Linear(self.config.ffn_config.dim, self.num_labels)
        # Initialize weights and apply final processing
        if not self.config.initializer_range:
            self.config.initializer_range = 0.02
        self.post_init()

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        output = embeddings
        if self.config.norm_strategy == "ic":
            if self.norm:
                output = self.norm(output)
            if self.dropout and self.dropout.p:
                output = self.dropout(output)
        else:
            if self.norm:
                output = self.norm(output)
        output = self.preclassifier(output)
        if self.activation:
            output = self.activation(output)
        if self.config.norm_strategy != "ic" and self.dropout and self.dropout.p:
            output = self.dropout(output)
        output = self.ffn(output)
        logits = self.classifier(output)
        return logits

    def _init_weights(self, module: torch.nn.Module) -> None:
        # Follows 'DistilBertPreTrainedModel._init_weights()'
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
