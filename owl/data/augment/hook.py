"""Sample hook protocol for owl dataset augmentation.

Sample hooks run after the Albumentations transform pipeline and before the
dataset converts NumPy arrays into tensors. They provide an extension point for
operations that need access to the complete sample, such as copy-move
augmentation, label updates, or edge reconstruction.
"""

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class AugmentSample(TypedDict):
    """Intermediate sample passed through dataset augmentation hooks.

    Attributes:
        tp:
            RGB image represented as a NumPy array with shape ``[H, W, 3]``.
        gt:
            Ground-truth mask represented as a NumPy array with shape
            ``[H, W]``.
        label:
            Image-level label. Missing labels are represented by ``-1``.
        edge:
            Edge mask represented as a NumPy array with shape ``[H, W]``.
    """

    tp: NDArray[np.generic]
    gt: NDArray[np.generic]
    label: int
    edge: NDArray[np.generic]


class SampleHook(Protocol):
    """Protocol implemented by post-transform sample hooks.

    Hooks may modify the supplied sample in place or return a newly created
    sample mapping. Implementations must always return an ``AugmentSample`` and
    must keep image and mask values as NumPy arrays.
    """

    def __call__(self, sample: AugmentSample) -> AugmentSample:
        """Transform and return a complete augmentation sample."""
        ...