"""Dataset entry protocol definitions.

This module defines the source-level representation of dataset samples and the
structural protocol for objects that provide those entries.

Entries describe facts supplied by the underlying dataset. They contain paths
and labels, but do not contain loaded resources, tensors, transformed data, or
derived supervision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SampleEntry:
    """Source-level description of one dataset sample.

    A sample entry records the resources and labels supplied by the underlying
    dataset. Dataset implementations consume entries to load resources, apply
    transforms, and construct the items returned to a dataloader.

    Entry sources must preserve the semantics of the underlying dataset. They
    must not infer labels, generate masks, or derive edge supervision.

    Attributes:
        tp:
            Path to the input image.
        gt:
            Optional path to the pixel-level ground-truth mask.
        label:
            Optional image-level class index defined by the dataset.
        edge:
            Optional path to a precomputed edge supervision mask.
    """

    tp: Path
    gt: Path | None = None
    label: int | None = None
    edge: Path | None = None


class EntrySource(Protocol):
    """Structural protocol for indexed collections of sample entries.

    Implementations may obtain entries from manifests, directory layouts, or
    any other user-defined storage convention. They do not need to inherit from
    this protocol; implementing the required methods is sufficient.

    Entry order must remain stable for the lifetime of the source.
    """

    def __len__(self) -> int:
        """Return the number of available sample entries."""
        ...

    def __getitem__(self, index: int) -> SampleEntry:
        """Return the sample entry at the given index."""
        ...