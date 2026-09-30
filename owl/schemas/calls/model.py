"""Model call schema definitions.

This module defines the training and inference call conventions for model
components used by owl runtimes. Model implementations receive a standard owl
dataset batch, while training calls additionally receive immutable runtime
progress information.
"""

from typing import Protocol

from ...data.dataset import DatasetBatch
from ..outputs.model import ModelOutput
from .context import TrainCallContext


class TrainModelCall(Protocol):
    """Protocol for model calls performed during training.

    The batch follows the standard owl dataset batch schema and contains the
    input image, ground-truth mask, image-level label, and edge-supervision
    mask.

    Args:
        batch:
            Batch produced by an owl training dataloader.
        context:
            Immutable progress information for the current training batch.

    Returns:
        Raw model output consumed by the owl output parser.
    """

    def __call__(
        self,
        batch: DatasetBatch,
        context: TrainCallContext,
    ) -> ModelOutput:
        ...


class InferModelCall(Protocol):
    """Protocol for model calls performed during inference.

    Inference calls receive only the current owl dataset batch. The model may
    consume the fields required by its task and ignore the remaining fields.

    Args:
        batch:
            Batch produced by an owl inference dataloader.

    Returns:
        Raw model output consumed by the owl output parser.
    """

    def __call__(
        self,
        batch: DatasetBatch,
    ) -> ModelOutput:
        ...
