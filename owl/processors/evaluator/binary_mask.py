"""Default evaluator for binary mask predictions.

This module provides a streaming evaluator for binary segmentation or
localization outputs represented as model logits.
"""

import math
from typing import cast

import torch
from torch import Tensor

from ...data.dataset import DatasetBatch
from ...schemas.outputs.types import (
    ParsedMetricOutputs,
    TensorOutputValue,
)
from ...utils.metric import (
    binarize_by_threshold,
    binary_f1_score,
    binary_roc_auc,
)


class BinaryMaskEvaluator:
    """Evaluate binary mask logits using pixel-level F1 and ROC AUC.

    The evaluator expects the model's ``eval`` output to be a single tensor of
    logits with shape ``[B, C, H, W]``. Ground-truth binary masks are read from
    ``batch["gt"]`` and must have the same shape.

    During each update, logits are converted to probabilities using sigmoid.
    The probabilities are then thresholded to obtain binary predictions for
    pixel-level F1 calculation. Pixel-level ROC AUC is calculated directly
    from the continuous probabilities.

    Metrics are accumulated as running sums and counts. Predictions and
    targets from previous batches are not retained.

    Pixel F1 is averaged over every batch item. Samples with empty target masks
    therefore follow the behavior of :func:`binary_f1_score` and produce an F1
    score of ``0.0``.

    Pixel ROC AUC is undefined when a target contains only one class. Such
    samples produce ``NaN`` values and are excluded from AUC aggregation. If
    no valid AUC value is observed over the complete dataloader,
    ``compute()`` returns ``NaN`` for ``pixel_auc``.

    Args:
        threshold:
            Probability threshold used to convert sigmoid probabilities into
            binary predictions. Values greater than or equal to the threshold
            are assigned to the positive class.

    Examples:
        Evaluate a batch and compute dataset-level metrics:

        >>> evaluator = BinaryMaskEvaluator(threshold=0.5)
        >>> evaluator.reset()
        >>>
        >>> evaluator.update(
        ...     eval_output=logits,
        ...     batch={"gt": target},
        ... )
        >>> metrics = evaluator.compute()
        >>> metrics.keys()
        dict_keys(['pixel_f1', 'pixel_auc'])

    Notes:
        This evaluator intentionally supports only a single tensor output.
        Models producing multiple evaluation tensors must use a custom
        evaluator that defines how each output should be interpreted.

        The evaluator assumes that ``eval_output`` contains logits. Supplying
        probabilities will apply sigmoid a second time and produce incorrect
        threshold semantics.
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold

        self._f1_sum: float
        self._f1_count: int
        self._auc_sum: float
        self._auc_count: int

        self.reset()

    def reset(self) -> None:
        """Clear all metrics accumulated from previous batches."""
        self._f1_sum = 0.0
        self._f1_count = 0
        self._auc_sum = 0.0
        self._auc_count = 0

    def update(
        self,
        *,
        eval_output: TensorOutputValue,
        batch: DatasetBatch,
    ) -> None:
        """Accumulate pixel-level metrics from one inference batch.

        One F1 score and one ROC AUC value are calculated independently for
        every batch item.

        All F1 values are included in aggregation. Samples with empty target
        masks therefore contribute the ``0.0`` value returned by
        :func:`binary_f1_score`.

        Undefined ROC AUC values are excluded from aggregation. An AUC value
        is undefined and represented as ``NaN`` when the corresponding target
        mask contains only one class. Consequently, only samples containing
        both positive and negative target pixels contribute to
        ``pixel_auc``.

        Args:
            eval_output:
                Model logits with shape ``[B, C, H, W]``. The evaluator
                expects exactly one tensor output.
            batch:
                Dataset batch containing a binary ground-truth mask under the
                ``"gt"`` key.
        """
        logits = cast(Tensor, eval_output)
        target = batch["gt"]

        probability = torch.sigmoid(logits)
        prediction = binarize_by_threshold(
            probability,
            threshold=self.threshold,
        )

        batch_f1 = binary_f1_score(
            prediction,
            target,
        )
        batch_auc = binary_roc_auc(
            target,
            probability,
        )

        self._f1_sum += batch_f1.sum().item()
        self._f1_count += batch_f1.numel()

        valid_auc = batch_auc[~torch.isnan(batch_auc)]
        self._auc_sum += valid_auc.sum().item()
        self._auc_count += valid_auc.numel()

    def compute(self) -> ParsedMetricOutputs:
        """Compute dataset-level averages from the accumulated metrics.

        ``pixel_f1`` is averaged over all accumulated samples, including
        samples with empty target masks.

        ``pixel_auc`` is averaged only over samples whose target masks contain
        both positive and negative pixels. Undefined ``NaN`` AUC values are
        excluded from aggregation.

        Returns:
            A dictionary containing:

            - ``pixel_f1``: mean image-level pixel F1 over all samples;
            - ``pixel_auc``: mean image-level pixel ROC AUC over valid samples.

            A metric is returned as ``NaN`` when no corresponding valid sample
            has been accumulated.
        """
        pixel_f1 = (
            self._f1_sum / self._f1_count
            if self._f1_count > 0
            else math.nan
        )
        pixel_auc = (
            self._auc_sum / self._auc_count
            if self._auc_count > 0
            else math.nan
        )

        return {
            "pixel_f1": pixel_f1,
            "pixel_auc": pixel_auc,
        }
