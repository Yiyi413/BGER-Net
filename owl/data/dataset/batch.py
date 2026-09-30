"""Dataset batch schema definitions.

This module defines the mapping produced when PyTorch collates standard owl
dataset items into a batch.
"""

from typing import TypedDict

from torch import Tensor


class DatasetBatch(TypedDict):
    """Batched mapping produced by an owl dataloader.

    Batch with edge supervision::

        {
            "tp_name": ["Tp_D_NRN_001", "Tp_D_NRN_002"],
            "tp": Tensor[B, 3, H, W],
            "gt": Tensor[B, 1, H, W],
            "label": Tensor[B],
            "edge": Tensor[B, 1, H, W],
        }

    Batch without edge supervision::

        {
            "tp_name": ["Tp_D_NRN_001", "Tp_D_NRN_002"],
            "tp": Tensor[B, 3, H, W],
            "gt": Tensor[B, 1, H, W],
            "label": Tensor[B],
            "edge": Tensor[B, 1, 1, 1],
        }

    PyTorch's default collate function stacks tensor fields from
    ``DatasetItem`` objects along a new leading batch dimension and collects
    string fields into lists.

    Datasets that provide full-resolution edge supervision must not be mixed
    with datasets that use the compact missing-edge placeholder in the same
    dataloader.

    Attributes:
        tp_name:
            File stems of the input images. The list length is ``B`` and each
            name corresponds to the sample at the same batch index.
        tp:
            Input image tensor with shape ``[B, 3, H, W]`` and dtype
            ``torch.float32``.
        gt:
            Binary ground-truth tensor with shape ``[B, 1, H, W]`` and dtype
            ``torch.float32``.
        label:
            Image-level class indices with shape ``[B]`` and dtype
            ``torch.int64``. Missing labels use ``-1``.
        edge:
            Binary edge-supervision tensor with dtype ``torch.float32``.

            Its shape is ``[B, 1, H, W]`` when edge supervision is available.
            Its shape is ``[B, 1, 1, 1]`` when edge supervision is absent.
    """

    tp_name: list[str]
    tp: Tensor
    gt: Tensor
    label: Tensor
    edge: Tensor
