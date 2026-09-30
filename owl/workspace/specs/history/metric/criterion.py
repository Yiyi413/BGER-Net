"""Criterion metric history schema definitions.

This module defines records appended to
``history/metric/criterion.jsonl``. Criterion metric records share the same
structure as model metric records but are stored separately to preserve their
component source.
"""

from typing import TypeAlias

from .model import MetricHistoryRecord


CriterionMetricHistoryRecord: TypeAlias = MetricHistoryRecord
"""Record appended to ``history/metric/criterion.jsonl``.

Example:
    Criterion metrics produced by batch 120 of epoch 3:

    >>> record: CriterionMetricHistoryRecord = {
    ...     "epoch": 3,
    ...     "batch": 120,
    ...     "metrics": {
    ...         "loss_bce": 0.12,
    ...         "loss_dice": 0.08,
    ...     },
    ... }
"""