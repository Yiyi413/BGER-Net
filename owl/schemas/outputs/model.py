"""Model output schema definitions.

This module defines the raw output schema returned by model forward calls.
Model outputs use fixed runtime keys for loss and evaluation payloads, and
prefixed keys for visualization outputs and scalar metrics.

Users are not required to instantiate or return any owl-specific output class.
Any plain dictionary matching this schema can be consumed by the runtime
parser.
"""

from typing import TypeAlias

from .types import ModelOutputValue


ModelOutput: TypeAlias = dict[str, ModelOutputValue]
"""Raw output dictionary returned by a model forward call.

Supported keys:
    loss:
        Tensor payload consumed by criterion during TrainRuntime.

    eval:
        Tensor payload consumed by evaluate functions during EvaluateRuntime.

    visual:*:
        Named tensor payloads consumed by visual functions during VisualRuntime.

    metric:*:
        Named scalar values consumed by logging, monitoring, or event systems.

Examples:
    A model output for training, evaluation, visualization, and logging:

        {
            "loss": (logits, aux_logits),
            "eval": logits,
            "visual:mask": pred_mask,
            "visual:heatmap": heatmap,
            "metric:score": 0.95,
        }

Notes:
    ``loss`` and ``eval`` are fixed keys. They do not support ``loss:*`` or
    ``eval:*`` forms because TrainRuntime and EvaluateRuntime consume a single
    tensor payload from each channel.

    ``visual:*`` and ``metric:*`` are prefixed keys because visual outputs and
    metrics can contain multiple named values.

    This type alias only describes the raw dictionary shape. Runtime users can
    return a plain dictionary directly; they do not need to import or construct
    ``ModelOutput``.

    Key validation, prefix validation, and value validation are handled by
    parser or validator modules.
"""