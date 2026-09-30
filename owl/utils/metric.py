"""Metric utilities for binary prediction tasks.

This module provides stateless tensor operations for threshold-based
binarization and image-level binary mask evaluation.
"""

from torch import Tensor
import torch

def binarize_by_threshold(
    values: Tensor,
    threshold: float,
) -> Tensor:
    """Binarize a tensor using an element-wise threshold.

    Values greater than or equal to ``threshold`` are converted to ``1``.
    Values below ``threshold`` are converted to ``0``.

    The function does not assume that ``values`` contains probabilities. It
    may be used with probabilities, logits, grayscale values, or other numeric
    scores.

    The returned tensor preserves the shape, dtype, and device of ``values``.
    The input tensor is not modified.

    Args:
        values:
            Numeric tensor of any shape.
        threshold:
            Threshold separating negative and positive values. Values equal to
            the threshold are assigned to the positive class.

    Returns:
        A new tensor containing only ``0`` and ``1`` values.

    Examples:
        Binarize probabilities:

        >>> values = torch.tensor([0.2, 0.5, 0.8])
        >>> binarize_by_threshold(values, threshold=0.5)
        tensor([0., 1., 1.])

        Binarize grayscale values:

        >>> values = torch.tensor([0, 127, 128, 255], dtype=torch.uint8)
        >>> binarize_by_threshold(values, threshold=128)
        tensor([0, 0, 1, 1], dtype=torch.uint8)

        Binarize logits around zero:

        >>> values = torch.tensor([-1.2, 0.0, 2.4])
        >>> binarize_by_threshold(values, threshold=0.0)
        tensor([0., 1., 1.])
    """
    return (values >= threshold).to(dtype=values.dtype)


def binary_f1_score(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Compute an image-level F1 score for each item in a batch.

    ``prediction`` and ``target`` must be binary tensors containing only
    ``0`` and ``1`` values. Both tensors must have the same
    ``[B, C, H, W]`` shape.

    The function treats all pixels across the channel and spatial dimensions
    as binary classification results for one image. It independently computes
    one F1 score for every batch item:

    ``F1 = 2 * TP / (2 * TP + FP + FN)``

    When the denominator is zero, meaning that both prediction and target
    contain no positive pixels, the corresponding F1 score is defined as
    ``0.0``.

    This function does not apply sigmoid or thresholding. Continuous
    probabilities or logits must be converted to binary values before they
    are passed to this function. The binary value constraint is not checked
    at runtime to avoid an additional full-tensor traversal.

    Args:
        prediction:
            Predicted binary masks with shape ``[B, C, H, W]``. Every element
            must be either ``0`` or ``1``.
        target:
            Ground-truth binary masks with the same shape as ``prediction``.
            Every element must be either ``0`` or ``1``.

    Returns:
        A floating-point tensor with shape ``[B]``. Each value is the
        image-level F1 score of the corresponding batch item.

    Raises:
        ValueError:
            If either input is not four-dimensional, or if their shapes do
            not match.

    Examples:
        Compute the F1 score for two binary masks:

        >>> prediction = torch.tensor(
        ...     [
        ...         [[[1, 1], [0, 0]]],
        ...         [[[1, 0], [1, 0]]],
        ...     ],
        ... )
        >>> target = torch.tensor(
        ...     [
        ...         [[[1, 1], [0, 0]]],
        ...         [[[1, 1], [0, 0]]],
        ...     ],
        ... )
        >>> binary_f1_score(prediction, target)
        tensor([1.0000, 0.5000])

        The result contains one score for each batch item:

        >>> prediction = torch.zeros(4, 1, 32, 32)
        >>> target = torch.zeros(4, 1, 32, 32)
        >>> binary_f1_score(prediction, target).shape
        torch.Size([4])
    """
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "prediction and target must have shape [B, C, H, W], "
            f"got prediction={tuple(prediction.shape)} and "
            f"target={tuple(target.shape)}"
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, "
            f"got prediction={tuple(prediction.shape)} and "
            f"target={tuple(target.shape)}"
        )

    prediction = prediction.bool()
    target = target.bool()

    reduce_dims = (1, 2, 3)

    true_positive = (prediction & target).sum(dim=reduce_dims)
    false_positive = (prediction & ~target).sum(dim=reduce_dims)
    false_negative = (~prediction & target).sum(dim=reduce_dims)

    numerator = 2 * true_positive
    denominator = numerator + false_positive + false_negative

    return numerator / denominator.clamp_min(1)


def binary_roc_auc(
    target: Tensor,
    prediction: Tensor,
) -> Tensor:
    """Compute an image-level pixel ROC AUC for each item in a batch.

    The implementation follows the pixel-level ROC AUC calculation used by
    IMDLBenCo. For each batch item, all values across the channel and spatial
    dimensions are flattened and treated as independent binary classification
    samples.

    Prediction scores are sorted in descending order. The corresponding
    binary targets are then used to construct cumulative true-positive and
    false-positive rates, and the area under the ROC curve is calculated using
    the trapezoidal rule.

    ``prediction`` may contain probabilities, logits, or other continuous
    scores because ROC AUC depends on their relative ordering rather than their
    absolute range.

    ``target`` must be a binary tensor containing only ``0`` and ``1`` values.
    This constraint is not checked at runtime to avoid an additional
    full-tensor traversal.

    If one batch item contains only a single target class, either all ``0`` or
    all ``1``, its ROC AUC is undefined. The corresponding output value is
    therefore ``NaN``. The caller is responsible for filtering or otherwise
    handling undefined values.

    Args:
        target:
            Ground-truth binary masks with shape ``[B, C, H, W]``. Every
            element must be either ``0`` or ``1``.
        prediction:
            Continuous prediction scores with the same ``[B, C, H, W]`` shape
            as ``target``.

    Returns:
        A floating-point tensor with shape ``[B]``. Each value is the
        image-level pixel ROC AUC of the corresponding batch item. Samples
        containing only one target class produce ``NaN``.

    Raises:
        ValueError:
            If either input is not four-dimensional, or if their shapes do
            not match.

    Examples:
        Compute ROC AUC for two images:

        >>> target = torch.tensor(
        ...     [
        ...         [[[0, 0], [1, 1]]],
        ...         [[[0, 1], [0, 1]]],
        ...     ],
        ... )
        >>> prediction = torch.tensor(
        ...     [
        ...         [[[0.1, 0.2], [0.8, 0.9]]],
        ...         [[[0.1, 0.9], [0.2, 0.8]]],
        ...     ],
        ... )
        >>> binary_roc_auc(target, prediction)
        tensor([1., 1.])

        A sample containing only one target class produces ``NaN``:

        >>> target = torch.zeros(1, 1, 2, 2)
        >>> prediction = torch.rand(1, 1, 2, 2)
        >>> binary_roc_auc(target, prediction)
        tensor([nan])

        Undefined values can be filtered before metric aggregation:

        >>> auc = binary_roc_auc(target, prediction)
        >>> valid_auc = auc[~torch.isnan(auc)]
        >>> valid_auc.numel()
        0
    """
    if target.ndim != 4 or prediction.ndim != 4:
        raise ValueError(
            "target and prediction must have shape [B, C, H, W], "
            f"got target={tuple(target.shape)} and "
            f"prediction={tuple(prediction.shape)}"
        )

    if target.shape != prediction.shape:
        raise ValueError(
            "target and prediction must have the same shape, "
            f"got target={tuple(target.shape)} and "
            f"prediction={tuple(prediction.shape)}"
        )

    batch_auc: list[Tensor] = []

    for batch_index in range(target.shape[0]):
        item_target = target[batch_index].flatten().to(torch.float32)
        item_prediction = (
            prediction[batch_index].flatten().to(torch.float32)
        )

        positive_count = item_target.sum()
        negative_count = item_target.numel() - positive_count

        if positive_count == 0 or negative_count == 0:
            batch_auc.append(
                torch.tensor(
                    float("nan"),
                    device=prediction.device,
                    dtype=torch.float32,
                )
            )
            continue

        sorted_indices = torch.argsort(
            item_prediction,
            descending=True,
        )
        sorted_target = item_target[sorted_indices]

        true_positives = torch.cumsum(sorted_target, dim=0)
        false_positives = torch.cumsum(1 - sorted_target, dim=0)

        true_positive_rate = true_positives / positive_count
        false_positive_rate = false_positives / negative_count

        batch_auc.append(
            torch.trapz(
                true_positive_rate,
                false_positive_rate,
            )
        )

    return torch.stack(batch_auc)
