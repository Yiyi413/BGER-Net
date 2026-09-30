"""Training snapshot schema definitions.

This module defines the current training snapshot stored in
``snapshot/train.json`` for each workspace. The snapshot describes the latest
published training position together with the runtime values associated with
that position.

Unlike history artifacts, this file contains only the latest coherent snapshot
and should be replaced atomically whenever a new snapshot is published.
"""

from typing import TypedDict


class Progress(TypedDict):
    """Progress of a bounded training dimension.

    ``current`` uses one-based indexing so the value can be displayed directly
    to users. ``total`` represents the complete number of positions in the
    dimension and is therefore a count rather than an index.

    Runtime implementations that use zero-based indexes must convert them
    before publishing the snapshot.

    Attributes:
        current:
            One-based position currently being processed. The valid range is
            from ``1`` through ``total``, inclusive.

        total:
            Total number of positions in this dimension.

    Example:
        Progress while processing the third epoch out of 100:

        >>> progress: Progress = {
        ...     "current": 3,
        ...     "total": 100,
        ... }
    """

    current: int
    total: int


class TrainSnapshot(TypedDict):
    """Current coherent snapshot of a training runtime.

    The epoch, batch, loss, and learning rates in one snapshot describe the
    same training position. Writers should publish the complete structure
    atomically so readers never observe progress and runtime values from
    different batches.

    Epoch and batch progress use one-based positions intended for external
    observation. For example, epoch ``1`` and batch ``1`` identify the first
    batch of the first epoch, even when the underlying runtime uses zero-based
    indexes internally.

    Attributes:
        epoch:
            Current one-based epoch position and the total epoch count.

        batch:
            Current one-based batch position within the active epoch and the
            total batch count for that epoch.

        loss:
            Scalar backward loss produced for the current batch.

        learning_rates:
            Current learning rate for each optimizer parameter group. The list
            order matches the optimizer parameter-group order.

    Example:
        Snapshot while processing batch 120 of epoch 3:

        >>> snapshot: TrainSnapshot = {
        ...     "epoch": {
        ...         "current": 3,
        ...         "total": 100,
        ...     },
        ...     "batch": {
        ...         "current": 120,
        ...         "total": 900,
        ...     },
        ...     "loss": 0.312,
        ...     "learning_rates": [0.0001],
        ... }
    """

    epoch: Progress
    batch: Progress
    loss: float
    learning_rates: list[float]