"""Training history schema definitions.

This module defines records appended to ``history/train.jsonl``. Each record
stores the loss and learning rates associated with one completed training
batch.

Unlike ``snapshot/train.json``, history records contain only the one-based
epoch and batch positions required to identify the training point. Total epoch
and batch counts are omitted to avoid repeating static progress information in
every JSONL record.
"""

from typing import TypedDict


class TrainHistoryRecord(TypedDict):
    """Training values recorded for one completed batch.

    Epoch and batch positions use one-based indexing for consistency with
    workspace snapshots. Runtime implementations that use zero-based indexes
    must convert them before publishing the record.

    Attributes:
        epoch:
            One-based epoch position associated with the training values.

        batch:
            One-based batch position within the active epoch.

        loss:
            Scalar backward loss produced for the batch.

        learning_rates:
            Learning rate for each optimizer parameter group. List order
            matches the order of ``optimizer.param_groups``. A single
            parameter group is still represented by a one-element list.

    Example:
        Training values produced by batch 120 of epoch 3:

        >>> record: TrainHistoryRecord = {
        ...     "epoch": 3,
        ...     "batch": 120,
        ...     "loss": 0.312,
        ...     "learning_rates": [0.00001, 0.0001],
        ... }
    """

    epoch: int
    batch: int
    loss: float
    learning_rates: list[float]