"""Criterion call schema definitions.

This module defines the call convention for criterion components used by owl
training runtimes. Criterion implementations receive the model payload selected
from the fixed ``loss`` output, the current owl dataset batch, and an immutable
training call context.
"""

from typing import Protocol

from ...data.dataset import DatasetBatch
from ..outputs.criterion import CriterionOutput
from ..outputs.types import TensorOutputValue
from .context import TrainCallContext


class CriterionCall(Protocol):
    """Protocol implemented by callable criterion components.

    The criterion receives only the model payload associated with the fixed
    ``loss`` output key. Evaluation, visualization, and metric outputs are
    excluded before the criterion is invoked.

    The batch follows the standard owl dataset batch schema and contains the
    input image, ground-truth mask, image-level label, and edge-supervision
    mask.

    Args:
        loss_output:
            Tensor payload parsed from the model's fixed ``loss`` key. The
            payload may be a tensor, a list of tensors, or a tuple of tensors.
            The criterion implementation is responsible for interpreting the
            payload structure and ordering.
        batch:
            Batch produced by an owl training dataloader.
        context:
            Immutable progress information for the current training batch.

    Returns:
        Raw criterion output consumed by the owl output parser.
    """

    def __call__(
        self,
        loss_output: TensorOutputValue,
        batch: DatasetBatch,
        context: TrainCallContext,
    ) -> CriterionOutput:
        ...
