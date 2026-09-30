"""Model output parsing utilities.

This module validates raw model outputs and converts them into structured
runtime outputs. Model output schema definitions live in
``owl.schemas.outputs``; this module implements the corresponding runtime
validation and parsing behavior.

Example:
    A model can return a plain dictionary without constructing any owl-specific
    output class:

        raw_output = {
            "loss": (logits, auxiliary_logits),
            "eval": logits,
            "visual:mask": predicted_mask,
            "visual:heatmap": attention_heatmap,
            "metric:score": 0.95,
        }

    After parsing:

        parsed_output = parse_model_output(raw_output)

    The resulting ``ParsedModelOutput`` is equivalent to:

        ParsedModelOutput(
            loss_output=(logits, auxiliary_logits),
            eval_output=logits,
            visual_outputs={
                "mask": predicted_mask,
                "heatmap": attention_heatmap,
            },
            metric_outputs={
                "score": 0.95,
            }
        )

    Tensor payloads are preserved by reference. Parsing does not copy tensor
    data or move tensors between devices.
"""

from typing import Any

from torch import Tensor

from ...schemas.outputs.model import ModelOutput
from ...schemas.outputs.parsed import ParsedModelOutput
from ...schemas.outputs.types import MetricValue, TensorOutputValue

_VISUAL_PREFIX = "visual:"
_METRIC_PREFIX = "metric:"


def parse_model_output(output: ModelOutput) -> ParsedModelOutput:
    """Validate and parse a raw model output dictionary.

    The parser separates raw model outputs according to their runtime
    consumption channel:

    - ``loss`` is stored in ``ParsedModelOutput.loss_output``.
    - ``eval`` is stored in ``ParsedModelOutput.eval_output``.
    - ``visual:<name>`` is stored in
      ``ParsedModelOutput.visual_outputs[name]``.
    - ``metric:<name>`` is stored in
      ``ParsedModelOutput.metric_outputs[name]``.

    The ``loss``, ``eval``, and ``visual:*`` channels accept tensor-oriented
    payloads:

        Tensor
        list[Tensor]
        tuple[Tensor, ...]

    The ``metric:*`` channel accepts Python scalar values:

        int
        float
        str
        bool

    Example:
        Given the following model output:

            output = {
                "loss": logits,
                "eval": logits.sigmoid(),
                "visual:mask": predicted_mask,
                "metric:confidence": 0.87,
            }

        Parsing it:

            parsed = parse_model_output(output)

        Produces a structure that can be consumed as:

            parsed.loss_output
            parsed.eval_output
            parsed.visual_outputs["mask"]
            parsed.metric_outputs["confidence"]

        Missing channels are represented by ``None`` or an empty dictionary.
        For example:

            parse_model_output({"eval": logits})

        produces:

            ParsedModelOutput(
                loss_output=None,
                eval_output=logits,
                visual_outputs={},
                metric_outputs={},
            )

    Args:
        output:
            Raw dictionary returned by a model forward call. The dictionary
            must use only the supported fixed keys and prefixed keys.

    Returns:
        Structured model output grouped by runtime consumption channel.

    Raises:
        TypeError:
            If ``output`` is not a dictionary, an output key is not a string,
            or an output value does not match the type required by its key.
        ValueError:
            If an output key is unsupported, a prefixed output name is empty,
            or a prefixed output name contains leading or trailing whitespace.
    """
    if not isinstance(output, dict):
        raise TypeError(
            "model output must be a dictionary, "
            f"got {type(output).__name__}"
        )

    parsed = ParsedModelOutput()

    for key, value in output.items():
        if not isinstance(key, str):
            raise TypeError(
                "model output keys must be strings, "
                f"got {type(key).__name__}"
            )

        if key == "loss":
            parsed.loss_output = _validate_tensor_output(
                key=key,
                value=value,
            )
            continue

        if key == "eval":
            parsed.eval_output = _validate_tensor_output(
                key=key,
                value=value,
            )
            continue

        if key.startswith(_VISUAL_PREFIX):
            name = _parse_prefixed_name(
                key=key,
                prefix=_VISUAL_PREFIX,
            )
            parsed.visual_outputs[name] = _validate_tensor_output(
                key=key,
                value=value,
            )
            continue

        if key.startswith(_METRIC_PREFIX):
            name = _parse_prefixed_name(
                key=key,
                prefix=_METRIC_PREFIX,
            )
            parsed.metric_outputs[name] = _validate_metric_output(
                key=key,
                value=value,
            )
            continue

        raise ValueError(f"unsupported model output key: {key!r}")

    return parsed


def _parse_prefixed_name(key: str, prefix: str) -> str:
    """Extract and validate the name following an output prefix.

    Example:
        ``visual:mask`` with the ``visual:`` prefix produces ``mask``.

    Args:
        key:
            Complete raw output key.
        prefix:
            Prefix expected at the beginning of the key.

    Returns:
        Output name after removing the prefix.

    Raises:
        ValueError:
            If the output name is empty or contains leading or trailing
            whitespace.
    """
    name = key[len(prefix):]

    if not name.strip():
        raise ValueError(
            f"model output key {key!r} must contain a name after {prefix!r}"
        )

    if name != name.strip():
        raise ValueError(
            f"model output name in {key!r} must not contain "
            "leading or trailing whitespace"
        )

    return name


def _validate_tensor_output(
    *,
    key: str,
    value: Any,
) -> TensorOutputValue:
    """Validate a tensor-oriented model output value.

    Tensor-oriented channels include ``loss``, ``eval``, and ``visual:*``.
    A value can be a single tensor or a list or tuple containing only tensors.

    Examples:
        Valid values:

            tensor
            [tensor_a, tensor_b]
            (tensor_a, tensor_b)

        Invalid values:

            0.95
            {"logits": tensor}
            [tensor, 0.95]

    Args:
        key:
            Raw model output key used in validation error messages.
        value:
            Raw value associated with the output key.

    Returns:
        The original tensor payload without copying or transformation.

    Raises:
        TypeError:
            If the value is not a supported tensor payload.
    """
    if isinstance(value, Tensor):
        return value

    if isinstance(value, list):
        if all(isinstance(item, Tensor) for item in value):
            return value

    if isinstance(value, tuple):
        if all(isinstance(item, Tensor) for item in value):
            return value

    raise TypeError(
        f"model output {key!r} must be a Tensor, list[Tensor], "
        f"or tuple[Tensor, ...], got {type(value).__name__}"
    )


def _validate_metric_output(
    *,
    key: str,
    value: Any,
) -> MetricValue:
    """Validate a scalar model metric value.

    Metric values must already be converted to Python scalar values. Tensor
    metrics are intentionally rejected because converting a CUDA tensor with
    ``Tensor.item()`` can synchronize the device and should remain an explicit
    user decision.

    Examples:
        Valid values:

            0.95
            10
            True
            "completed"

        Invalid value:

            torch.tensor(0.95)

        Tensor metrics should be converted before being returned:

            {
                "metric:score": score_tensor.item(),
            }

    Args:
        key:
            Raw model output key used in validation error messages.
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
        f"model output {key!r} must be an int, float, str, or bool, "
        f"got {type(value).__name__}"
    )