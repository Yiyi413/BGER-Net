"""Runtime call context schema definitions.

This module defines immutable context objects passed to runtime components.
Call contexts contain execution progress managed by owl and must not expose
mutable runtime or session objects to user components.
"""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class TrainCallContext:
    """Runtime context for a single training batch.

    A training runtime creates this context for each batch and passes the training
    context to the model and criterion. Component implementations may inspect
    the context when their behavior depends on the current training progress.

    All indexes are zero-based. The context is immutable so model and criterion
    implementations cannot modify runtime progress.

    Attributes:
        current_epoch: Index of the epoch currently being executed.
        current_batch: Index of the batch within the current epoch.
        total_epochs: Total number of epochs configured for the training run.
        total_batches: Total number of batches in one training epoch.

    Raises:
        ValueError: If an index is negative, a total is not positive, or an
            index falls outside its configured range.
    """

    current_epoch: int
    current_batch: int
    total_epochs: int
    total_batches: int

    def __post_init__(self) -> None:
        """Validate the training progress represented by this context."""
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")

        if self.total_batches <= 0:
            raise ValueError("total_batches must be greater than zero")

        if not 0 <= self.current_epoch < self.total_epochs:
            raise ValueError(
                "current_epoch must be in "
                f"[0, {self.total_epochs}), got {self.current_epoch}"
            )

        if not 0 <= self.current_batch < self.total_batches:
            raise ValueError(
                "current_batch must be in "
                f"[0, {self.total_batches}), got {self.current_batch}"
            )

ModelCallContext: TypeAlias = TrainCallContext | None
"""Context accepted by model forward calls.

Training calls receive a ``TrainCallContext``. Inference calls omit the
context argument, so model implementations should provide ``None`` as the
default value.
"""