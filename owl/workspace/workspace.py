"""Workspace lifecycle and artifact persistence implementations.

This module defines the public workspace facade used by Owl invocations.

A workspace owns its static metadata, mutable snapshots, append-only history
artifacts, checkpoint directory, and visual output directory. The workspace
context manager coordinates artifact initialization, normal completion,
interruption reporting, and writer cleanup.
"""

from datetime import datetime
from pathlib import Path
from types import TracebackType
try:
    from typing import Self
except ImportError:  # Python 3.10 on some AutoDL images
    from typing_extensions import Self

from owl.__about__ import __version__
from owl.schemas.outputs.parsed import ParsedMetricOutputs

from .history import _HistoryStore
from .snapshot import _SnapshotStore
from .specs.history.evaluation import EvaluationResults
from .specs.metadata import WorkspaceMetadata, WorkspaceMode
from .specs.snapshot.lifecycle import WorkspaceLifecycleStage
from .writer.atomic import AtomicJsonWriter


_METADATA_SCHEMA_VERSION = 1


class Workspace:
    """Manage the artifacts associated with one Owl invocation.

    A workspace instance may be entered only once. Entering the context creates
    the workspace root, publishes static metadata, initializes the snapshot and
    history stores, and starts the running lifecycle before invocation
    execution begins.

    Args:
        parent:
            Parent directory under which the timestamped workspace directory is
            created.

        mode:
            Invocation mode associated with the workspace.

    Notes:
        Checkpoint and visual directories are created lazily when requested.

        Runtime exceptions always retain propagation priority over persistence
        and cleanup exceptions. When invocation execution succeeds but cleanup
        fails, the first cleanup exception is propagated.
    """

    def __init__(
        self,
        *,
        parent: Path,
        mode: WorkspaceMode,
    ) -> None:
        self._parent = Path(parent)
        self._mode = mode

        self._workspace_id: str | None = None
        self._root: Path | None = None

        self._snapshot_store: _SnapshotStore | None = None
        self._history_store: _HistoryStore | None = None

        self._entered = False
        self._closed = False

    @property
    def mode(self) -> WorkspaceMode:
        """Return the invocation mode associated with the workspace."""
        return self._mode

    @property
    def workspace_id(self) -> str:
        """Return the generated workspace identifier.

        Raises:
            RuntimeError:
                If the workspace has not been entered successfully.
        """
        if self._workspace_id is None:
            raise RuntimeError("workspace has not been entered")

        return self._workspace_id

    @property
    def root(self) -> Path:
        """Return the workspace root directory.

        Raises:
            RuntimeError:
                If the workspace has not been entered successfully.
        """
        if self._root is None:
            raise RuntimeError("workspace has not been entered")

        return self._root

    def __enter__(self) -> Self:
        """Create and initialize the workspace.

        Initialization order:

        1. Generate the workspace identity and root path.
        2. Create the root directory.
        3. Publish ``metadata.json``.
        4. Construct snapshot and history stores.
        5. Publish the initial running lifecycle.

        Returns:
            The initialized workspace instance.

        Raises:
            RuntimeError:
                If the workspace instance has already been entered or closed.

            FileExistsError:
                If the generated workspace directory already exists.

            OSError:
                If directory or artifact initialization fails.
        """
        if self._entered:
            raise RuntimeError("workspace has already been entered")

        if self._closed:
            raise RuntimeError("workspace has already been closed")

        created_at = datetime.now().astimezone()
        workspace_id = _create_workspace_id(created_at)
        root = self._parent / workspace_id

        self._workspace_id = workspace_id
        self._root = root

        try:
            root.mkdir(parents=True, exist_ok=False)

            metadata: WorkspaceMetadata = {
                "schema_version": _METADATA_SCHEMA_VERSION,
                "workspace_id": workspace_id,
                "created_at": created_at.isoformat(timespec="milliseconds"),
                "mode": self._mode,
                "owl_version": __version__,
            }

            AtomicJsonWriter(
                path=root / "metadata.json",
            ).write(metadata)

            snapshot_store = _SnapshotStore(
                root=root,
                mode=self._mode,
            )
            history_store = _HistoryStore(
                root=root,
            )

            snapshot_store.start()
        except BaseException:
            self._closed = True
            raise

        self._snapshot_store = snapshot_store
        self._history_store = history_store
        self._entered = True

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Finalize the workspace and release history writer resources.

        When invocation execution succeeds, pending snapshots are flushed and
        the lifecycle is marked completed before history writers are closed.

        When invocation execution raises, the lifecycle is marked interrupted,
        history writers are closed, and the original exception is allowed to
        propagate.

        Cleanup failures do not replace an existing invocation exception. If
        execution succeeded, the first cleanup failure is propagated after all
        required cleanup attempts have completed.

        Returns:
            Always ``False`` when an invocation exception exists so the
            original exception continues propagating.
        """
        del exc_type, traceback

        snapshot_store, history_store = self._require_active_stores()

        if exc_value is not None:
            return self._exit_with_runtime_error(
                runtime_error=exc_value,
                snapshot_store=snapshot_store,
                history_store=history_store,
            )

        self._exit_normally(
            snapshot_store=snapshot_store,
            history_store=history_store,
        )
        return False

    def set_stage(
        self,
        stage: WorkspaceLifecycleStage,
    ) -> None:
        """Set and immediately publish the active runtime stage."""
        snapshot_store, _ = self._require_active_stores()
        snapshot_store.set_stage(stage)

    def update_train_snapshot(
        self,
        *,
        current_epoch: int,
        total_epoch: int,
        current_batch: int,
        total_batch: int,
        loss: float,
        learning_rates: list[float],
    ) -> None:
        """Update the current throttled training snapshot."""
        snapshot_store, _ = self._require_active_stores()

        snapshot_store.update_train(
            current_epoch=current_epoch,
            total_epoch=total_epoch,
            current_batch=current_batch,
            total_batch=total_batch,
            loss=loss,
            learning_rates=learning_rates,
        )

    def append_train_history(
        self,
        *,
        epoch: int,
        batch: int,
        loss: float,
        learning_rates: list[float],
    ) -> None:
        """Append one completed training batch to train history."""
        _, history_store = self._require_active_stores()

        history_store.append_train(
            epoch=epoch,
            batch=batch,
            loss=loss,
            learning_rates=learning_rates,
        )

    def append_evaluation_history(
        self,
        *,
        results: EvaluationResults,
        epoch: int | None = None,
    ) -> None:
        """Append the complete results of one evaluation run."""
        _, history_store = self._require_active_stores()

        history_store.append_evaluation(
            results=results,
            epoch=epoch,
        )

    def append_model_metric_history(
        self,
        *,
        epoch: int,
        batch: int,
        metrics: ParsedMetricOutputs,
    ) -> None:
        """Append model metrics produced for one training batch."""
        _, history_store = self._require_active_stores()

        history_store.append_model_metric(
            epoch=epoch,
            batch=batch,
            metrics=metrics,
        )

    def append_criterion_metric_history(
        self,
        *,
        epoch: int,
        batch: int,
        metrics: ParsedMetricOutputs,
    ) -> None:
        """Append criterion metrics produced for one training batch."""
        _, history_store = self._require_active_stores()

        history_store.append_criterion_metric(
            epoch=epoch,
            batch=batch,
            metrics=metrics,
        )

    def checkpoint_dir(self) -> Path:
        """Return the lazily created checkpoint directory."""
        self._require_active_stores()

        path = self.root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)

        return path

    def visual_dir(self) -> Path:
        """Return the lazily created visual output directory."""
        self._require_active_stores()

        path = self.root / "visual"
        path.mkdir(parents=True, exist_ok=True)

        return path

    def flush(self) -> None:
        """Force publication of pending snapshot and history buffers.

        Snapshot state is flushed before history buffers. Errors propagate
        immediately to the caller.
        """
        snapshot_store, history_store = self._require_active_stores()

        snapshot_store.flush()
        history_store.flush()

    def _exit_normally(
        self,
        *,
        snapshot_store: _SnapshotStore,
        history_store: _HistoryStore,
    ) -> None:
        """Finalize a workspace whose invocation returned normally."""
        first_error: BaseException | None = None

        try:
            snapshot_store.complete()
        except BaseException as error:
            first_error = error

        try:
            history_store.close()
        except BaseException as error:
            if first_error is None:
                first_error = error

        if first_error is not None:
            try:
                snapshot_store.publish_interrupted(
                    _exception_message(first_error)
                )
            except BaseException:
                pass

        self._closed = True

        if first_error is not None:
            raise first_error

    def _exit_with_runtime_error(
        self,
        *,
        runtime_error: BaseException,
        snapshot_store: _SnapshotStore,
        history_store: _HistoryStore,
    ) -> bool:
        """Finalize a workspace after invocation execution raises."""
        cleanup_errors: list[BaseException] = []
        interrupted_published = False
        message = _exception_message(runtime_error)

        try:
            snapshot_store.interrupt(message)
        except BaseException as error:
            cleanup_errors.append(error)
        else:
            interrupted_published = True

        try:
            history_store.close()
        except BaseException as error:
            cleanup_errors.append(error)

        if not interrupted_published:
            try:
                snapshot_store.publish_interrupted(message)
            except BaseException as error:
                cleanup_errors.append(error)

        self._closed = True

        for cleanup_error in cleanup_errors:
            runtime_error.add_note(
                "Workspace cleanup also failed: "
                f"{_exception_message(cleanup_error)}"
            )

        return False

    def _require_active_stores(
        self,
    ) -> tuple[_SnapshotStore, _HistoryStore]:
        """Return active stores after validating workspace state."""
        if not self._entered:
            raise RuntimeError("workspace has not been entered")

        if self._closed:
            raise RuntimeError("workspace has already been closed")

        if self._snapshot_store is None or self._history_store is None:
            raise RuntimeError("workspace stores have not been initialized")

        return self._snapshot_store, self._history_store


def _create_workspace_id(
    created_at: datetime,
) -> str:
    """Create a timestamp-based workspace identifier.

    The identifier uses the local date and time represented by ``created_at``
    and includes milliseconds to reduce the chance of collisions between
    workspaces created within the same second.

    Format:
        ``workspace-YYYYMMDDHHMMSSmmm``

    Args:
        created_at:
            Time associated with workspace creation. The datetime is expected
            to already contain the desired local timezone information.

    Returns:
        Workspace identifier containing the date, time, and three-digit
        millisecond component.

    Example:
        A workspace created at ``2026-07-19 14:23:08.456`` produces:

        ``workspace-20260719142308456``
    """
    milliseconds = created_at.microsecond // 1000

    return (
        f"workspace-{created_at:%Y%m%d%H%M%S}"
        f"{milliseconds:03d}"
    )



def _exception_message(
    error: BaseException,
) -> str:
    """Return a stable human-readable exception description."""
    error_type = type(error).__name__
    message = str(error)

    if not message:
        return error_type

    return f"{error_type}: {message}"
