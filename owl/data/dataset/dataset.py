"""Standard owl dataset implementation.

This module materializes source-level ``SampleEntry`` objects as stable
mappings that can be consumed by PyTorch dataloaders.

The processing pipeline is:

    SampleEntry
    -> derive the input image name
    -> load disk resources as NumPy arrays
    -> apply optional Albumentations transform
    -> apply optional sample hook
    -> normalize binary masks
    -> convert NumPy arrays to tensors
    -> DatasetItem

Ground-truth resources may be missing on individual samples. A missing ground
truth is represented by a full-resolution zero mask because it participates in
the same spatial pipeline as the input image.

Edge supervision follows a dataset-level all-or-none protocol. An entry source
must either provide edge paths for every sample or omit edge paths for every
sample. When edge supervision is absent, every item contains a compact
``[1, 1, 1]`` zero edge tensor.

Datasets with different edge-supervision modes must not be combined in the same
dataloader when using PyTorch's default collate function.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..augment import AugmentSample, SampleHook
from ..entry import EntrySource
from .item import DatasetItem


class OwlDataset(Dataset[DatasetItem]):
    """Materialize source entries as standard owl dataset items.

    Every returned item contains the fields ``tp_name``, ``tp``, ``gt``,
    ``label``, and ``edge``.

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

    The ``tp_name`` field is derived from the input image path stem and is
    preserved as sample metadata. Inference runtimes may use it to associate
    generated visualizations with their source images.

    Edge supervision uses a dataset-level all-or-none protocol:

    - all source entries provide an edge path; or
    - all source entries omit an edge path.

    Mixed edge availability inside one source is rejected. This keeps the edge
    tensor shape stable for PyTorch's default collate function.

    Missing ground-truth resources are represented by full-resolution zero
    masks. Missing labels are represented by ``-1``.

    Albumentations receives the input through the ``image`` target. Ground truth
    and available edge supervision are passed through the ``masks`` target so
    spatial transforms remain synchronized.

    When edge supervision is absent, the ``[1, 1]`` edge placeholder does not
    enter the Albumentations pipeline. A configured sample hook may replace this
    placeholder with a generated full-resolution edge mask.

    Args:
        source:
            Indexed source that provides ``SampleEntry`` objects.
        transform:
            Optional Albumentations pipeline operating on NumPy arrays.
            Normalization is allowed, but tensor conversion is not.
        hook:
            Optional post-transform hook receiving and returning an
            ``AugmentSample``.

    Raises:
        ValueError:
            If the source is empty or mixes entries with and without edge paths.
    """

    def __init__(
        self,
        source: EntrySource,
        transform: A.Compose | None = None,
        hook: SampleHook | None = None,
    ) -> None:
        if len(source) == 0:
            raise ValueError("dataset entry source is empty")

        self.source = source
        self.transform = transform
        self.hook = hook
        self._has_edge = self._resolve_edge_availability(source)

    @property
    def has_edge(self) -> bool:
        """Whether every source entry provides an edge resource.

        This property describes edge resources supplied by the ``EntrySource``.
        It does not describe edge masks that may be generated dynamically by a
        sample hook.

        A future Client may use this property to prevent combining datasets with
        incompatible edge tensor shapes.
        """
        return self._has_edge

    def __len__(self) -> int:
        """Return the number of entries exposed by the dataset."""
        return len(self.source)

    def __getitem__(self, index: int) -> DatasetItem:
        """Load, process, and materialize one dataset item."""
        entry = self.source[index]
        tp_name = entry.tp.stem

        tp_array = self._load_array(
            path=entry.tp,
            mode="RGB",
        )
        image_size = tp_array.shape[:2]

        gt_array = (
            self._load_array(
                path=entry.gt,
                mode="L",
            )
            if entry.gt is not None
            else np.zeros(image_size, dtype=np.uint8)
        )

        if self._has_edge:
            if entry.edge is None:
                raise RuntimeError(
                    "dataset edge protocol changed after initialization"
                )

            edge_array = self._load_array(
                path=entry.edge,
                mode="L",
            )
        else:
            edge_array = self._missing_edge()

        if self.transform is not None:
            masks = [gt_array]

            if self._has_edge:
                masks.append(edge_array)

            transformed = self.transform(
                image=tp_array,
                masks=np.stack(masks, axis=0),
            )

            tp_array = transformed["image"]
            gt_array = transformed["masks"][0]

            if self._has_edge:
                edge_array = transformed["masks"][1]

        label = entry.label if entry.label is not None else -1

        # The intermediate mapping only exists when a sample-level extension is
        # configured. The common path keeps using local variables directly.
        if self.hook is not None:
            sample = self.hook(
                AugmentSample(
                    tp=tp_array,
                    gt=gt_array,
                    label=label,
                    edge=edge_array,
                )
            )

            tp_array = sample["tp"]
            gt_array = sample["gt"]
            label = sample["label"]
            edge_array = sample["edge"]

            self._validate_hook_output(
                tp=tp_array,
                gt=gt_array,
                edge=edge_array,
            )

        gt_array = self._normalize_binary_mask(gt_array)
        edge_array = self._normalize_binary_mask(edge_array)

        # NumPy slicing and user hooks may produce arrays with negative strides
        # or non-contiguous layouts. torch.from_numpy requires a supported
        # contiguous layout, so this adaptation remains a Dataset responsibility.
        tp_array = np.ascontiguousarray(tp_array)
        gt_array = np.ascontiguousarray(gt_array)
        edge_array = np.ascontiguousarray(edge_array)

        # NumPy images use HWC layout. DatasetItem images use CHW layout.
        tp_tensor = (
            torch.from_numpy(tp_array)
            .to(torch.float32)
            .permute(2, 0, 1)
        )

        # NumPy masks use HW layout. DatasetItem masks use 1HW layout.
        gt_tensor = (
            torch.from_numpy(gt_array)
            .to(torch.float32)
            .unsqueeze(0)
        )
        edge_tensor = (
            torch.from_numpy(edge_array)
            .to(torch.float32)
            .unsqueeze(0)
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.int64,
        )

        return DatasetItem(
            tp_name=tp_name,
            tp=tp_tensor,
            gt=gt_tensor,
            label=label_tensor,
            edge=edge_tensor,
        )

    @staticmethod
    def _resolve_edge_availability(source: EntrySource) -> bool:
        """Resolve the dataset-level edge-supervision mode.

        Returns:
            ``True`` when every entry provides an edge path.
            ``False`` when every entry omits an edge path.

        Raises:
            ValueError:
                If the source mixes entries with and without edge paths.
        """
        first_has_edge = source[0].edge is not None

        for index in range(1, len(source)):
            has_edge = source[index].edge is not None

            if has_edge != first_has_edge:
                raise ValueError(
                    "entry source must either provide edge paths for every "
                    "sample or omit edge paths for every sample"
                )

        return first_has_edge

    @staticmethod
    def _load_array(
        *,
        path: Path,
        mode: str,
    ) -> np.ndarray:
        """Load an image resource as an independent NumPy array.

        Conversion occurs while the file is open, so the returned array does not
        depend on Pillow's file object or retain an open file descriptor.

        Args:
            path:
                Path to the image resource.
            mode:
                Pillow conversion mode, such as ``"RGB"`` or ``"L"``.
        """
        try:
            with Image.open(path) as image:
                return np.array(image.convert(mode))
        except FileNotFoundError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"failed to load image {path}: {error}"
            ) from error

    @staticmethod
    def _missing_edge() -> np.ndarray:
        """Return the compact placeholder for absent edge supervision."""
        return np.zeros((1, 1), dtype=np.uint8)

    @staticmethod
    def _validate_hook_output(
        *,
        tp: np.ndarray,
        gt: np.ndarray,
        edge: np.ndarray,
    ) -> None:
        """Validate the spatial layouts returned by a sample hook.

        A hook may keep the ``[1, 1]`` missing-edge placeholder or replace it
        with a full-resolution edge mask matching ``tp``.
        """
        if tp.ndim != 3 or tp.shape[2] != 3:
            raise ValueError(
                f"hook output tp must have shape [H, W, 3], got {tp.shape}"
            )

        if gt.ndim != 2 or gt.shape != tp.shape[:2]:
            raise ValueError(
                "hook output gt must have shape [H, W] matching tp, "
                f"got tp={tp.shape[:2]} and gt={gt.shape}"
            )

        if edge.ndim != 2:
            raise ValueError(
                "hook output edge must have shape [H, W] or [1, 1], "
                f"got {edge.shape}"
            )

        if edge.shape != (1, 1) and edge.shape != tp.shape[:2]:
            raise ValueError(
                "hook output edge must match tp spatial size or remain the "
                f"[1, 1] placeholder, got tp={tp.shape[:2]} and "
                f"edge={edge.shape}"
            )

    @staticmethod
    def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Convert a mask into a float32 array containing zero and one."""
        threshold = 0.5 if mask.max() <= 1 else 127.5
        return (mask > threshold).astype(np.float32)
