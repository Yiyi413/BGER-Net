"""Criterion output schema definitions.

This module defines the raw output schema returned by criterion forward calls.
Criterion outputs use a fixed loss key for the backward target and prefixed
metric keys for optional logging or monitoring values.
"""

from typing import TypeAlias

from .types import CriterionOutputValue


CriterionOutput: TypeAlias = dict[str, CriterionOutputValue]
"""Raw output dictionary returned by a criterion forward call.

Required keys:
    loss:
        Tensor used as the backward target during TrainRuntime.

Optional keys:
    metric:*:
        Scalar values consumed by logging, monitoring, or event systems.

Examples:
    A criterion output with the main training loss and loss metrics:

        {
            "loss": loss,
            "metric:loss_bce": 0.32,
            "metric:loss_dice": 0.18,
        }

Notes:
    Criterion outputs do not support ``loss:*`` keys. The ``loss`` key is a
    fixed field because TrainRuntime consumes a single backward target.

    This type alias only describes the raw dictionary shape. Required field
    validation, key validation, and value validation are handled by parser or
    validator modules.
"""