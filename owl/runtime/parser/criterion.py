"""Criterion output parsing utilities.

This module validates raw criterion outputs and converts them into structured
runtime outputs. Criterion output schema definitions live in
``owl.schemas.outputs``; this module implements the corresponding runtime
validation and parsing behavior.

Example:
    A criterion can return a plain dictionary without constructing any
    owl-specific output class:

        raw_output = {
            "loss": loss,
            "metric:loss_bce": 0.32,
            "metric:loss_dice": 0.18,
        }

    After parsing:

        parsed_output = parse_criterion_output(raw_output)

    The resulting ``ParsedCriterionOutput`` is equivalent to:

        ParsedCriterionOutput(
            loss=loss,
            metric_outputs={
                "loss_bce": 0.32,
                "loss_dice": 0.18,
            }
        )

    The loss tensor is preserved by reference. Parsing does not copy tensor
    data or move tensors between devices.
"""

from typing import Any

from torch import Tensor

from ...schemas.outputs.criterion import CriterionOutput
from ...schemas.outputs.parsed import ParsedCriterionOutput
from ...schemas.outputs.types import MetricValue

_METRIC_PREFIX = "metric:"


def parse_criterion_output(output: CriterionOutput) -> ParsedCriterionOutput:
    """Validate and parse a raw criterion output dictionary.

    Criterion outputs contain one required backward target and optional scalar
    metrics:

    - ``loss`` is stored in ``ParsedCriterionOutput.loss``.
    - ``metric:<name>`` is stored in
      ``ParsedCriterionOutput.metric_outputs[name]``.

    The ``loss`` value must be a single ``Tensor``. Unlike model tensor
    outputs, criterion loss does not accept a list or tuple because
    ``TrainRuntime`` consumes one backward target.

    Metric values must be Python scalar values:

        int
        float
        str
        bool

    Example:
        Given the following criterion output:

            output = {
                "loss": total_loss,
                "metric:bce": 0.32,
                "metric:dice": 0.18,
            }

        Parsing it:

            parsed = parse_criterion_output(output)

        Produces a structure that can be consumed as:

            parsed.loss
            parsed.metric_outputs["bce"]
            parsed.metric_outputs["dice"]

    Args:
        output:
            Raw dictionary returned by a criterion forward call. The dictionary
            must contain the fixed ``loss`` key and may contain ``metric:*``
            keys.

    Returns:
        Structured criterion output containing the backward loss tensor and
        named scalar metrics.

    Raises:
        TypeError:
            If ``output`` is not a dictionary, an output key is not a string,
            the ``loss`` value is not a tensor, or a metric value is not a
            supported Python scalar.
        ValueError:
            If the required ``loss`` key is missing, an output key is
            unsupported, or a metric name is empty or contains leading or
            trailing whitespace.
    """
    if not isinstance(output, dict):
        raise TypeError(
            "criterion output must be a dictionary, "
            f"got {type(output).__name__}"
        )

    if "loss" not in output:
        raise ValueError("criterion output must contain the required 'loss' key")

    loss = _validate_loss_output(output["loss"])
    metric_outputs: dict[str, MetricValue] = {}

    for key, value in output.items():
        if not isinstance(key, str):
            raise TypeError(
                "criterion output keys must be strings, "
                f"got {type(key).__name__}"
            )

        if key == "loss":
            continue

        if key.startswith(_METRIC_PREFIX):
            name = _parse_metric_name(key)
            metric_outputs[name] = _validate_metric_output(
                key=key,
                value=value,
            )
            continue

        raise ValueError(f"unsupported criterion output key: {key!r}")

    return ParsedCriterionOutput(
        loss=loss,
        metric_outputs=metric_outputs,
    )


def _parse_metric_name(key: str) -> str:
    """Extract and validate the name following the ``metric:`` prefix.

    Example:
        ``metric:loss_bce`` produces ``loss_bce``.

    Args:
        key:
            Complete raw criterion output key.

    Returns:
        Metric name after removing the ``metric:`` prefix.

    Raises:
        ValueError:
            If the metric name is empty or contains leading or trailing
            whitespace.
    """
    name = key[len(_METRIC_PREFIX):]

    if not name.strip():
        raise ValueError(
            f"criterion output key {key!r} must contain a name after "
            f"{_METRIC_PREFIX!r}"
        )

    if name != name.strip():
        raise ValueError(
            f"criterion output name in {key!r} must not contain "
            "leading or trailing whitespace"
        )

    return name


def _validate_loss_output(value: Any) -> Tensor:
    """Validate the criterion backward target.

    The criterion must return exactly one tensor under the fixed ``loss`` key.

    Examples:
        Valid value:

            total_loss

        Invalid values:

            [loss_bce, loss_dice]
            (loss_bce, loss_dice)
            0.5

        Multiple loss terms should be combined by the criterion before being
        returned:

            {
                "loss": loss_bce + loss_dice,
                "metric:loss_bce": loss_bce.item(),
                "metric:loss_dice": loss_dice.item(),
            }

    Args:
        value:
            Raw value associated with the ``loss`` key.

    Returns:
        The original loss tensor without copying or transformation.

    Raises:
        TypeError:
            If the value is not a ``Tensor``.
    """
    if isinstance(value, Tensor):
        return value

    raise TypeError(
        "criterion output 'loss' must be a Tensor, "
        f"got {type(value).__name__}"
    )


def _validate_metric_output(
    *,
    key: str,
    value: Any,
) -> MetricValue:
    """Validate a scalar criterion metric value.

    Metric values must already be converted to Python scalar values. Tensor
    metrics are intentionally rejected because converting a CUDA tensor with
    ``Tensor.item()`` can synchronize the device and should remain an explicit
    user decision.

    Examples:
        Valid values:

            0.32
            10
            True
            "stable"

        Invalid value:

            torch.tensor(0.32)

        Tensor metrics should be converted before being returned:

            {
                "loss": total_loss,
                "metric:loss_bce": loss_bce.item(),
            }

    Args:
        key:
            Raw criterion output key used in validation error messages.
        value:
            Raw metric value associated with the output key.

    Returns:
        The validated Python scalar value.

    Raises:
        TypeError:
            If the metric value is not an ``int``, ``float``, ``str``, or
            ``bool``.
    """
    if isinstance(value, (int, float, str, bool)):
        return value

    raise TypeError(
        f"criterion output {key!r} must be an int, float, str, or bool, "
        f"got {type(value).__name__}"
    )