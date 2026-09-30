"""Dataset visualization helpers.

This module provides a small interactive viewer for inspecting the tensor items
returned by an owl dataset. It is intended for debugging dataset loading,
augmentation behavior, and final tensor shapes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from .dataset import OwlDataset


def visualize_dataset(dataset: OwlDataset) -> None:
    """Display dataset items in an interactive Matplotlib window.

    The viewer displays the input image, ground-truth mask, and edge mask in
    three columns. The tensor shape of each field is shown below its image.

    Example with edge supervision::

        TP             GT             Edge
        [3, H, W]      [1, H, W]      [1, H, W]

    Example without edge supervision::

        TP             GT             Edge
        [3, H, W]      [1, H, W]      [1, 1, 1]

    The current sample index, image-level label, and label tensor shape are
    shown in the figure title.

    Use the ``Next`` button to advance to the next item. Use the ``Exit`` button
    or close the window to stop visualization.

    The dataset is accessed directly instead of through a dataloader because
    this helper is intended to inspect the output of ``Dataset.__getitem__``,
    including configured transforms and sample hooks.

    Args:
        dataset:
            Dataset returning the standard owl ``DatasetItem`` schema.

    Raises:
        ValueError:
            If the dataset contains no items.
    """
    if len(dataset) == 0:
        raise ValueError("cannot visualize an empty dataset")

    index = 0

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    figure.subplots_adjust(bottom=0.22, top=0.82)

    next_axis = figure.add_axes((0.72, 0.05, 0.1, 0.075))
    exit_axis = figure.add_axes((0.84, 0.05, 0.1, 0.075))

    next_button = Button(next_axis, "Next")
    exit_button = Button(exit_axis, "Exit")

    def draw_item() -> None:
        item = dataset[index]

        tp_tensor = item["tp"]
        gt_tensor = item["gt"]
        label_tensor = item["label"]
        edge_tensor = item["edge"]

        tp = (
            tp_tensor
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        gt = (
            gt_tensor
            .detach()
            .cpu()
            .squeeze(0)
            .numpy()
        )
        edge = (
            edge_tensor
            .detach()
            .cpu()
            .squeeze(0)
            .numpy()
        )
        label = int(label_tensor.item())

        tp = np.clip(tp, 0, 255).astype(np.uint8)

        for axis in axes:
            axis.clear()
            axis.set_xticks([])
            axis.set_yticks([])

        axes[0].imshow(tp)
        axes[0].set_title("TP")
        axes[0].set_xlabel(
            f"Tensor shape: {tuple(tp_tensor.shape)}"
        )

        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[1].set_xlabel(
            f"Tensor shape: {tuple(gt_tensor.shape)}"
        )

        axes[2].imshow(edge, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Edge")
        axes[2].set_xlabel(
            f"Tensor shape: {tuple(edge_tensor.shape)}"
        )

        figure.suptitle(
            f"Sample {index + 1}/{len(dataset)} | "
            f"Label: {label} | "
            f"Label shape: {tuple(label_tensor.shape)}"
        )

        next_button.set_active(index < len(dataset) - 1)
        figure.canvas.draw_idle()

    def show_next(_event: object) -> None:
        nonlocal index

        if index >= len(dataset) - 1:
            return

        index += 1
        draw_item()

    def exit_viewer(_event: object) -> None:
        plt.close(figure)

    next_button.on_clicked(show_next)
    exit_button.on_clicked(exit_viewer)

    draw_item()
    plt.show()