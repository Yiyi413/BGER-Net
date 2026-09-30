"""Parsed output schema definitions.

This module defines structured outputs produced after raw model outputs and
criterion outputs have been parsed. Runtime implementations should consume
these parsed structures instead of parsing raw dictionaries directly.
"""

from dataclasses import dataclass, field

from torch import Tensor

from .types import ParsedMetricOutputs, ParsedTensorOutputs, TensorOutputValue


@dataclass(slots=True)
class ParsedModelOutput:
    """Structured model output after parsing.

    Attributes:
        loss_output:
            Tensor payload parsed from the fixed ``loss`` key.

        eval_output:
            Tensor payload parsed from the fixed ``eval`` key.

        visual_outputs:
            Named tensor payloads parsed from ``visual:*`` keys.

        metric_outputs:
            Scalar metric outputs parsed from ``metric:*`` keys.
    """

    loss_output: TensorOutputValue | None = None
    eval_output: TensorOutputValue | None = None
    visual_outputs: ParsedTensorOutputs = field(default_factory=dict)
    metric_outputs: ParsedMetricOutputs = field(default_factory=dict)


@dataclass(slots=True)
class ParsedCriterionOutput:
    """Structured criterion output after parsing.

    Attributes:
        loss:
            Tensor used as the backward target during TrainRuntime.

        metric_outputs:
            Scalar metric outputs parsed from ``metric:*`` keys.
    """

    loss: Tensor
    metric_outputs: ParsedMetricOutputs = field(default_factory=dict)