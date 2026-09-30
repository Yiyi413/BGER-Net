"""Checkpoint loading declarations and resolution.

This module defines how an owl invocation requests checkpoint restoration and
applies serialized state to resolved runtime components.

Checkpoint loading and checkpoint saving are separate concerns.

``CheckpointLoad`` describes which checkpoint should be loaded and whether owl
should restore only model parameters or the complete training state.

Model-only loading is suitable for pretrained weights, transfer learning, and
inference:

    checkpoint=CheckpointLoad(
        path="checkpoints/pretrained.pth",
        model_only=True,
    )

It accepts either an Owl checkpoint v1 dictionary or a bare model state
dictionary.

Full checkpoint loading restores the model, optimizer, scheduler, and completed
epoch stored in an Owl checkpoint v1:

    checkpoint=CheckpointLoad(
        path="checkpoints/latest.pth",
        model_only=False,
    )

The resolver also accepts an explicit ``model_only`` override. This allows a
calling workflow to restrict restoration regardless of the policy stored in
the declaration. For example, inference can force model-only restoration:

    resolve_checkpoint(
        checkpoint,
        model=model,
        model_only=True,
    )

When ``model_only`` is not explicitly supplied, the value stored in
``CheckpointLoad.model_only`` is used.

For full checkpoint restoration, the returned epoch is the zero-based completed
epoch represented by the checkpoint. The training orchestration layer derives
the resume position as:

    start_epoch = checkpoint_epoch + 1

Checkpoint values are deserialized onto CPU by default. Callers may provide a
different ``map_location`` when resolving the declaration.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...schemas.checkpoint import validate_checkpoint_v1
from ..data.types import PathLike


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointLoad:
    """Configuration describing how a checkpoint should be restored.

    This object records the checkpoint source and default loading policy. File
    access and component state restoration are deferred until
    ``resolve_checkpoint()`` is called.

    Attributes:
        path:
            Path to one checkpoint file.

        model_only:
            Default checkpoint restoration policy.

            When ``True``, either an Owl checkpoint v1 or a bare model state
            dictionary is accepted. Only model parameters are restored.

            When ``False``, the file must contain an Owl checkpoint v1. Model,
            optimizer, scheduler, and completed epoch state are restored.

            A caller may explicitly override this value through the
            ``model_only`` argument of ``resolve_checkpoint()``.

        strict:
            Whether model state keys must exactly match the current model.

            This value is forwarded to
            ``torch.nn.Module.load_state_dict(..., strict=strict)``. It affects
            model restoration only.
    """

    path: PathLike
    model_only: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        """Normalize the checkpoint path without accessing the file system."""

        object.__setattr__(
            self,
            "path",
            Path(self.path),
        )


def resolve_checkpoint(
    declaration: CheckpointLoad | None,
    *,
    model: Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    model_only: bool | None = None,
    map_location: str | torch.device = "cpu",
) -> int | None:
    """Resolve and apply a checkpoint loading declaration.

    ``None`` leaves all supplied components unchanged.

    The effective restoration policy is determined by ``model_only``. When that
    argument is ``None``, the policy stored in ``declaration.model_only`` is
    used. An explicitly supplied value takes precedence over the declaration.

    Model-only restoration accepts either an Owl checkpoint v1 or a bare model
    state dictionary. It restores model parameters and returns ``None``.

    Full restoration requires an Owl checkpoint v1 and resolved optimizer and
    scheduler instances. It restores all three component states and returns the
    zero-based completed epoch stored in the checkpoint.

    Deserialization errors, checkpoint schema errors, and native PyTorch state
    restoration errors are allowed to propagate unchanged.

    Args:
        declaration:
            Checkpoint loading configuration, or ``None`` when no checkpoint
            should be restored.

        model:
            Resolved model that receives the checkpoint model state.

        optimizer:
            Resolved optimizer that receives the checkpoint optimizer state.
            Required for full checkpoint restoration.

        scheduler:
            Resolved scheduler that receives the checkpoint scheduler state.
            Required for full checkpoint restoration.

        model_only:
            Optional restoration-policy override.

            ``None`` uses ``declaration.model_only``. ``True`` forces model-only
            restoration. ``False`` forces full checkpoint restoration.

        map_location:
            Device onto which checkpoint tensors are deserialized.

    Returns:
        The zero-based completed epoch stored in a fully restored Owl checkpoint
        v1, or ``None`` when no checkpoint is loaded or only model state is
        restored.

    Raises:
        TypeError:
            If full checkpoint restoration is requested without an optimizer or
            scheduler.

        ValueError:
            If full restoration receives data that does not match the Owl
            checkpoint v1 schema.
    """
    if declaration is None:
        return None

    loaded = torch.load(
        declaration.path,
        map_location=map_location,
        weights_only=True,
    )

    effective_model_only = (
        declaration.model_only
        if model_only is None
        else model_only
    )

    if effective_model_only:
        model.load_state_dict(
            _resolve_model_state(loaded),
            strict=declaration.strict,
        )

        return None

    if optimizer is None:
        raise TypeError(
            "optimizer is required when loading a full checkpoint"
        )

    if scheduler is None:
        raise TypeError(
            "scheduler is required when loading a full checkpoint"
        )

    checkpoint = validate_checkpoint_v1(loaded)

    model.load_state_dict(
        checkpoint["model"],
        strict=declaration.strict,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["epoch"]


def _resolve_model_state(
    loaded: Any,
) -> Mapping[str, Any]:
    """Extract model state from an Owl checkpoint or bare state dictionary.

    A mapping identified as an Owl checkpoint v1 is validated and its ``model``
    field is returned. Any other mapping is treated as a bare model state
    dictionary.

    Args:
        loaded:
            Object returned by ``torch.load()``.

    Returns:
        Mapping passed to ``torch.nn.Module.load_state_dict()``.

    Raises:
        TypeError:
            If the deserialized object is not a mapping.

        ValueError:
            If the object identifies itself as an Owl checkpoint v1 but does
            not satisfy that schema.
    """
    if not isinstance(loaded, Mapping):
        raise TypeError(
            "model checkpoint must contain a mapping"
        )

    if loaded.get("format_version") == 1:
        return validate_checkpoint_v1(loaded)["model"]

    return loaded


__all__ = [
    "CheckpointLoad",
    "resolve_checkpoint",
]