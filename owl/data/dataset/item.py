"""Dataset item schema definitions.

This module defines the mapping returned by owl datasets. Dataset items are
ordinary dictionaries at runtime, while ``TypedDict`` provides a stable schema
for static type checking and editor support.
"""

from typing import TypedDict

from torch import Tensor


class DatasetItem(TypedDict):
    """Representation of one dataset sample.

    Example with edge supervision::

        {
            "tp_name": "Tp_D_NRN_001",
            "tp": Tensor[3, H, W],
            "gt": Tensor[1, H, W],
            "label": Tensor[],
            "edge": Tensor[1, H, W],
        }

    Example without edge supervision::

        {
            "tp_name": "Tp_D_NRN_001",
            "tp": Tensor[3, H, W],
            "gt": Tensor[1, H, W],
            "label": Tensor[],
            "edge": Tensor[1, 1, 1],
        }

    All owl datasets return the same keys. Edge supervision follows a
    dataset-level all-or-none protocol: a dataset must provide edge resources
    for every sample or omit them for every sample.

    Attributes:
        tp_name:
            File stem of the input image. The name excludes the original file
            extension and is used to associate inference visualizations with
            their source images.
        tp:
            Input image tensor with shape ``[3, H, W]`` and dtype
            ``torch.float32``. Values preserve the numeric scale produced by
            the dataset transform.
        gt:
            Binary ground-truth mask with shape ``[1, H, W]`` and dtype
            ``torch.float32``. A missing ground-truth resource is represented
            by a full-resolution zero tensor.
        label:
            Image-level class index represented by a scalar ``torch.int64``
            tensor. A missing label is represented by ``-1``.
        edge:
            Binary edge-supervision tensor with dtype ``torch.float32``.

            Its shape is ``[1, H, W]`` when edge supervision is available.
            Its shape is ``[1, 1, 1]`` when the dataset does not provide edge
            supervision and the sample hook does not generate one.
    """

    tp_name: str
    tp: Tensor
    gt: Tensor
    label: Tensor
    edge: Tensor
