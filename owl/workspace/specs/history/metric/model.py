"""Model metric history schema definitions.

This module defines records appended to
``history/metric/model.jsonl``. Each record contains all model metrics
published for one training batch.
"""

from typing import TypeAlias, TypedDict

from owl.schemas.outputs.types import ParsedMetricOutputs


class MetricHistoryRecord(TypedDict):
    """Metric values recorded for one training batch.

    Epoch and batch positions use one-based indexing for consistency with
    workspace snapshots. Runtime implementations that use zero-based indexes
    must convert them before publishing the record.

    Attributes:
        epoch:
            One-based epoch position associated with the metric values.

        batch:
            One-based batch position within the active epoch.

        metrics:
            Parsed metric values published for the batch. Metric names are
            preserved as dictionary keys, and their values use the normalized
            metric value types produced by the output parser.

    Example:
        Model metrics produced by batch 120 of epoch 3:

        >>> record: MetricHistoryRecord = {
        ...     "epoch": 3,
        ...     "batch": 120,
        ...     "metrics": {
        ...         "feature_norm": 0.23,
        ...         "confidence": 0.91,
        ...     },
        ... }
    """

    epoch: int
    batch: int
    metrics: ParsedMetricOutputs


ModelMetricHistoryRecord: TypeAlias = MetricHistoryRecord
"""Record appended to ``history/metric/model.jsonl``."""