"""Evaluation history schema definitions.

This module defines records appended to ``history/evaluation.jsonl``. Each
record stores the complete evaluation results returned for all named
dataloaders during one evaluation run.

Evaluation records produced during training include the one-based epoch
position associated with the evaluation. Standalone inference records omit the
epoch field.
"""

from typing import NotRequired, TypeAlias, TypedDict

from owl.schemas.outputs.types import ParsedMetricOutputs


EvaluationResults: TypeAlias = dict[str, ParsedMetricOutputs]
"""Evaluation metrics grouped by dataloader name."""


class EvaluationHistoryRecord(TypedDict):
    """Results recorded for one completed evaluation run.

    All named dataloader results from the same evaluation are stored in one
    JSONL record. This preserves the evaluation as one coherent unit and avoids
    requiring readers to reconstruct it from several lines.

    Attributes:
        epoch:
            Optional one-based training epoch position associated with the
            evaluation.

            This field is present when evaluation is performed during a
            training invocation and omitted for standalone inference.

        results:
            Parsed evaluation metrics grouped by dataloader name. Each key is
            the name assigned to a dataloader, and each value contains the
            metrics computed for that dataloader.

    Example:
        Evaluation performed after epoch 3 of a training invocation:

        >>> record: EvaluationHistoryRecord = {
        ...     "epoch": 3,
        ...     "results": {
        ...         "casia": {
        ...             "f1": 0.82,
        ...             "auc": 0.91,
        ...         },
        ...         "nist16": {
        ...             "f1": 0.75,
        ...             "auc": 0.86,
        ...         },
        ...     },
        ... }

        Evaluation performed by a standalone inference invocation:

        >>> record = {
        ...     "results": {
        ...         "casia": {
        ...             "f1": 0.82,
        ...             "auc": 0.91,
        ...         },
        ...     },
        ... }
    """

    epoch: NotRequired[int]
    results: EvaluationResults