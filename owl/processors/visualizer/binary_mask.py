"""Default visualizer for binary mask predictions.

This module converts binary mask logits into grayscale probability maps or
binary prediction masks and saves them to runtime-managed destinations.
"""

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor

from ...schemas.outputs.types import ParsedTensorOutputs
from ...schemas.processors.visualizer import VisualizationOutputs
from ...utils.metric import binarize_by_threshold


class BinaryMaskVisualizer:
    """Visualize binary mask logits as grayscale images.

    The visualizer reads the ``"prediction"`` entry from the parsed model
    visualization outputs. Its value must be one logits tensor with shape
    ``[B, 1, H, W]``.

    The expected raw model output therefore contains a key such as::

        {
            "visual:prediction": logits,
        }

    After model-output parsing, the visualizer receives::

        {
            "prediction": logits,
        }

    The logits are converted to probabilities using sigmoid. By default,
    probabilities are thresholded at ``0.5`` and returned as binary prediction
    masks. Passing ``threshold=None`` preserves the continuous probabilities
    and produces grayscale probability maps instead.

    The inference runtime later validates the returned image batch, removes the
    batch dimension, moves each image to the CPU, constructs its destination
    path, and invokes :meth:`save`.

    Args:
        threshold:
            Optional probability threshold used to produce binary masks.

            Values greater than or equal to the threshold are assigned to the
            positive class. The default value ``0.5`` produces binary masks.
            Passing ``None`` preserves continuous probability maps in the
            ``[0, 1]`` range.

    Examples:
        Produce binary prediction masks using the default threshold:

        >>> visualizer = BinaryMaskVisualizer()
        >>> outputs = visualizer.visualize(
        ...     visual_outputs={"prediction": logits},
        ... )
        >>> outputs["prediction"].shape
        torch.Size([2, 1, 512, 512])
        >>> torch.unique(outputs["prediction"])
        tensor([0., 1.])

        Produce continuous grayscale probability maps:

        >>> visualizer = BinaryMaskVisualizer(threshold=None)
        >>> outputs = visualizer.visualize(
        ...     visual_outputs={"prediction": logits},
        ... )
        >>> outputs["prediction"].min() >= 0
        tensor(True)
        >>> outputs["prediction"].max() <= 1
        tensor(True)

    Notes:
        This default implementation reads only the ``"prediction"`` entry.
        Its value must be a single tensor containing binary mask logits.
        Additional visualization entries are ignored.

        The input tensor must contain logits rather than probabilities.
        Supplying probabilities applies sigmoid a second time and produces
        incorrect visualization values and threshold semantics.

        Models exposing progressive predictions, tensor sequences, overlays,
        differently named visualization materials, or multiple output types
        should use a custom visualizer.

        Returned image tensors remain in the ``[0, 1]`` range. The
        :meth:`save` method converts each image into an 8-bit grayscale image
        in the ``[0, 255]`` range.
    """

    def __init__(
        self,
        threshold: float | None = 0.5,
    ) -> None:
        self.threshold = threshold

    def visualize(
        self,
        *,
        visual_outputs: ParsedTensorOutputs,
    ) -> VisualizationOutputs:
        """Convert prediction logits into a visualization image batch.

        The ``"prediction"`` entry is interpreted as one binary mask logits
        tensor with shape ``[B, 1, H, W]``. Sigmoid is applied before optional
        thresholding.

        Args:
            visual_outputs:
                Parsed model visualization outputs containing a
                ``"prediction"`` logits tensor.

                Additional entries are ignored by this default implementation.

        Returns:
            A mapping containing one ``"prediction"`` image batch.

            When ``threshold`` is a float, the returned tensor contains binary
            ``0`` and ``1`` values. When ``threshold`` is ``None``, it contains
            continuous probabilities in the ``[0, 1]`` range.
        """
        logits = cast(Tensor, visual_outputs["prediction"])
        probability = torch.sigmoid(logits)

        if self.threshold is None:
            prediction = probability
        else:
            prediction = binarize_by_threshold(
                probability,
                threshold=self.threshold,
            )

        return {
            "prediction": prediction,
        }

    def save(
        self,
        *,
        image: Tensor,
        path: Path,
    ) -> None:
        """Save one binary mask visualization as an 8-bit grayscale image.

        The inference runtime supplies one detached CPU image tensor with shape
        ``[1, H, W]`` and a complete destination path.

        Values are clamped to ``[0, 1]``, scaled to ``[0, 255]``, rounded,
        converted to unsigned 8-bit integers, and encoded as a grayscale image.

        Args:
            image:
                One probability map or binary mask with shape ``[1, H, W]``.
            path:
                Complete destination path generated and managed by the
                inference runtime.
        """
        array = (
            image.clamp(0.0, 1.0)
            .mul(255)
            .round()
            .to(torch.uint8)
            .squeeze(0)
            .numpy()
        )

        Image.fromarray(array).save(path)
