"""Owl checkpoint v1 schema definitions.

This module defines the first version of the checkpoint dictionary written and
consumed by owl training workflows.

An Owl checkpoint v1 contains the state of the model, optimizer, and scheduler
at the end of one completed epoch:

    {
        "format_version": 1,
        "epoch": 4,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

The ``epoch`` field is the zero-based index of the completed epoch represented
by the stored component states. A resumed training invocation derives its
starting epoch as:

    start_epoch = checkpoint["epoch"] + 1

Checkpoint v1 supports epoch-boundary recovery only. It does not preserve
partial progress within an epoch, batch indexes, or global step counters.

The schema describes a plain dictionary suitable for ``torch.save()``.
``validate_checkpoint_v1()`` validates only the required top-level structure.
The contents of individual state dictionaries are validated by the respective
PyTorch ``load_state_dict()`` implementations.
"""

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast


class OwlCheckpointV1(TypedDict):
    """Dictionary schema for an Owl checkpoint v1.

    Attributes:
        format_version:
            Checkpoint schema version. Version 1 checkpoints contain the
            literal value ``1``.

        epoch:
            Zero-based index of the last fully completed epoch represented by
            the stored states.

        model:
            State dictionary returned by ``torch.nn.Module.state_dict()``.

        optimizer:
            State dictionary returned by
            ``torch.optim.Optimizer.state_dict()``.

        scheduler:
            State dictionary returned by
            ``torch.optim.lr_scheduler.LRScheduler.state_dict()``.

    Notes:
        All fields are required.

        ``TypedDict`` provides static typing only. Runtime validation of the
        required top-level fields is performed by
        ``validate_checkpoint_v1()``.
    """

    format_version: Literal[1]
    epoch: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


def validate_checkpoint_v1(value: Any) -> OwlCheckpointV1:
    """Validate the top-level structure of an Owl checkpoint v1.

    Validation is intentionally limited to the checkpoint envelope. The
    internal contents of model, optimizer, and scheduler state dictionaries
    are delegated to their corresponding PyTorch ``load_state_dict()``
    implementations.

    Args:
        value:
            Object deserialized from a checkpoint file.

    Returns:
        The validated checkpoint dictionary.

    Raises:
        TypeError:
            If the checkpoint is not a mapping or a required field has an
            invalid top-level type.

        ValueError:
            If a required field is missing or the checkpoint version is not
            version 1.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            "Owl checkpoint v1 must contain a mapping"
        )

    required_fields = (
        "format_version",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in value
    ]

    if missing_fields:
        missing = ", ".join(missing_fields)

        raise ValueError(
            f"Owl checkpoint v1 is missing required fields: {missing}"
        )

    if value["format_version"] != 1:
        raise ValueError(
            "Owl checkpoint format_version must be 1"
        )

    if not isinstance(value["epoch"], int):
        raise TypeError(
            "Owl checkpoint epoch must be an integer"
        )

    for field in ("model", "optimizer", "scheduler"):
        if not isinstance(value[field], Mapping):
            raise TypeError(
                f"Owl checkpoint {field} state must be a mapping"
            )

    return cast(OwlCheckpointV1, value)


__all__ = [
    "OwlCheckpointV1",
    "validate_checkpoint_v1",
]