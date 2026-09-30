"""Workspace history persistence implementations.

This module manages append-only JSON Lines artifacts stored under a workspace
``history/`` directory.

Each history artifact owns an independent throttled writer. Writers open their
files lazily on the first appended record, remain open for repeated writes, and
flush their Python text buffers according to a dynamically configured time
interval.
"""

from pathlib import Path

from owl.schemas.outputs.parsed import ParsedMetricOutputs

from .specs.history.evaluation import (
    EvaluationHistoryRecord,
    EvaluationResults,
)
from .specs.history.metric.criterion import (
    CriterionMetricHistoryRecord,
)
from .specs.history.metric.model import (
    ModelMetricHistoryRecord,
)
from .specs.history.train import TrainHistoryRecord
from .writer.jsonl import ThrottledJsonlWriter


_HISTORY_FLUSH_INTERVAL_ENVIRONMENT_VARIABLE = (
    "OWL_WORKSPACE_HISTORY_FLUSH_INTERVAL_MS"
)
_DEFAULT_HISTORY_FLUSH_INTERVAL_MS = 1000


class _HistoryStore:
    """Manage append-only workspace history artifacts.

    Each artifact is assigned its own throttled JSON Lines writer. Writer
    objects are created during store construction, but their directories,
    files, and file handles are created lazily when the first corresponding
    record is appended.

    Args:
        root:
            Root directory of the active workspace.

    Notes:
        Appending a record writes the serialized JSON line into the Python text
        buffer immediately. The buffer is flushed automatically according to
        the configured interval, or explicitly through ``flush`` and ``close``.

        Each writer tracks its own flush interval. Activity in one history
        artifact therefore does not cause unrelated history files to flush.

        This class does not call ``os.fsync``. Flushing makes records visible
        outside the Python file buffer without forcing synchronous physical
        storage writes into the training loop.
    """

    def __init__(
        self,
        *,
        root: Path,
    ) -> None:
        history_dir = root / "history"
        metric_dir = history_dir / "metric"

        self._train_writer = ThrottledJsonlWriter(
            path=history_dir / "train.jsonl",
            interval_environment_variable=(
                _HISTORY_FLUSH_INTERVAL_ENVIRONMENT_VARIABLE
            ),
            default_interval_ms=_DEFAULT_HISTORY_FLUSH_INTERVAL_MS,
        )
        self._evaluation_writer = ThrottledJsonlWriter(
            path=history_dir / "evaluation.jsonl",
            interval_environment_variable=(
                _HISTORY_FLUSH_INTERVAL_ENVIRONMENT_VARIABLE
            ),
            default_interval_ms=_DEFAULT_HISTORY_FLUSH_INTERVAL_MS,
        )
        self._model_metric_writer = ThrottledJsonlWriter(
            path=metric_dir / "model.jsonl",
            interval_environment_variable=(
                _HISTORY_FLUSH_INTERVAL_ENVIRONMENT_VARIABLE
            ),
            default_interval_ms=_DEFAULT_HISTORY_FLUSH_INTERVAL_MS,
        )
        self._criterion_metric_writer = ThrottledJsonlWriter(
            path=metric_dir / "criterion.jsonl",
            interval_environment_variable=(
                _HISTORY_FLUSH_INTERVAL_ENVIRONMENT_VARIABLE
            ),
            default_interval_ms=_DEFAULT_HISTORY_FLUSH_INTERVAL_MS,
        )

    def append_train(
        self,
        *,
        epoch: int,
        batch: int,
        loss: float,
        learning_rates: list[float],
    ) -> None:
        """Append values produced by one completed training batch.

        Args:
            epoch:
                One-based epoch position associated with the batch.

            batch:
                One-based batch position within the active epoch.

            loss:
                Scalar backward loss produced for the batch.

            learning_rates:
                Learning rates ordered according to optimizer parameter groups.

        Raises:
            RuntimeError:
                If the training history writer has already been closed.
        """
        record: TrainHistoryRecord = {
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "learning_rates": list(learning_rates),
        }

        self._train_writer.append(record)

    def append_evaluation(
        self,
        *,
        results: EvaluationResults,
        epoch: int | None = None,
    ) -> None:
        """Append the complete results of one evaluation run.

        One record contains the results of every named dataloader involved in
        the evaluation.

        Args:
            results:
                Parsed evaluation metrics grouped by dataloader name.

            epoch:
                Optional one-based training epoch associated with the
                evaluation. Standalone inference omits this field.

        Raises:
            RuntimeError:
                If the evaluation history writer has already been closed.
        """
        copied_results: EvaluationResults = {
            dataloader_name: dict(metrics)
            for dataloader_name, metrics in results.items()
        }

        record: EvaluationHistoryRecord = {
            "results": copied_results,
        }

        if epoch is not None:
            record["epoch"] = epoch

        self._evaluation_writer.append(record)

    def append_model_metric(
        self,
        *,
        epoch: int,
        batch: int,
        metrics: ParsedMetricOutputs,
    ) -> None:
        """Append model metrics produced for one training batch.

        Args:
            epoch:
                One-based epoch position associated with the metrics.

            batch:
                One-based batch position within the active epoch.

            metrics:
                Complete parsed model metric output for the batch.

        Raises:
            RuntimeError:
                If the model metric history writer has already been closed.
        """
        record: ModelMetricHistoryRecord = {
            "epoch": epoch,
            "batch": batch,
            "metrics": dict(metrics),
        }

        self._model_metric_writer.append(record)

    def append_criterion_metric(
        self,
        *,
        epoch: int,
        batch: int,
        metrics: ParsedMetricOutputs,
    ) -> None:
        """Append criterion metrics produced for one training batch.

        Args:
            epoch:
                One-based epoch position associated with the metrics.

            batch:
                One-based batch position within the active epoch.

            metrics:
                Complete parsed criterion metric output for the batch.

        Raises:
            RuntimeError:
                If the criterion metric history writer has already been closed.
        """
        record: CriterionMetricHistoryRecord = {
            "epoch": epoch,
            "batch": batch,
            "metrics": dict(metrics),
        }

        self._criterion_metric_writer.append(record)

    def flush(self) -> None:
        """Force flushing of every history writer.

        Writers whose files have not been opened ignore the operation. Errors
        propagate immediately to the caller.

        Raises:
            RuntimeError:
                If any history writer has already been closed.
        """
        self._train_writer.flush()
        self._evaluation_writer.flush()
        self._model_metric_writer.flush()
        self._criterion_metric_writer.flush()

    def close(self) -> None:
        """Flush and close every history writer.

        Every writer is given a chance to close even when an earlier writer
        raises an exception. The first exception is re-raised after all close
        operations have been attempted.

        Closing an already closed store has no effect because individual
        writers implement idempotent close operations.
        """
        first_error: BaseException | None = None

        for writer in (
            self._train_writer,
            self._evaluation_writer,
            self._model_metric_writer,
            self._criterion_metric_writer,
        ):
            try:
                writer.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise first_error
