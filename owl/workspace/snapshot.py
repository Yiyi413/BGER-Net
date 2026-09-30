"""Workspace snapshot persistence implementations.

This module manages mutable snapshot artifacts stored under a workspace
``snapshot/`` directory.

Lifecycle snapshots are published immediately whenever their state changes.
Training snapshots are submitted for every batch and published atomically
according to a dynamically configured time interval.
"""

from pathlib import Path

from .specs.metadata import WorkspaceMode
from .specs.snapshot.lifecycle import (
    WorkspaceLifecycle,
    WorkspaceLifecycleStage,
)
from .specs.snapshot.train import TrainSnapshot


from .writer.atomic import (
    AtomicJsonWriter,
    ThrottledAtomicJsonWriter,
)


_SNAPSHOT_INTERVAL_ENVIRONMENT_VARIABLE = (
    "OWL_WORKSPACE_SNAPSHOT_INTERVAL_MS"
)
_DEFAULT_SNAPSHOT_INTERVAL_MS = 1000


class _SnapshotStore:
    """Manage current workspace snapshots and their persistence policy.

    Lifecycle changes are low-frequency and operationally important, so every
    change is published immediately. Training snapshots may be updated once per
    batch and are therefore published using a time-based throttle.

    Args:
        root:
            Root directory of the active workspace.

        mode:
            Invocation mode associated with the workspace. Training workspaces
            begin in the ``train`` stage, while inference workspaces begin in
            the ``infer`` stage.

    Notes:
        Construction has no filesystem side effects. ``start`` must be called
        after the workspace root and metadata artifact have been created and
        before the invocation begins.

        Snapshot files are replaced atomically. Readers therefore observe
        either the previous complete snapshot or the new complete snapshot,
        never a partially written JSON document.
    """

    def __init__(
        self,
        *,
        root: Path,
        mode: WorkspaceMode,
    ) -> None:
        self._mode = mode
        self._lifecycle: WorkspaceLifecycle | None = None

        snapshot_dir = root / "snapshot"

        self._lifecycle_writer = AtomicJsonWriter(
            path=snapshot_dir / "lifecycle.json",
        )
        self._train_writer = ThrottledAtomicJsonWriter(
            path=snapshot_dir / "train.json",
            interval_environment_variable=(
                _SNAPSHOT_INTERVAL_ENVIRONMENT_VARIABLE
            ),
            default_interval_ms=_DEFAULT_SNAPSHOT_INTERVAL_MS,
        )

    def start(self) -> None:
        """Initialize and publish the running lifecycle snapshot.

        The initial execution stage is derived from the workspace mode.
        Calling this method more than once is not supported.

        Raises:
            RuntimeError:
                If the snapshot store has already been started.
        """
        if self._lifecycle is not None:
            raise RuntimeError("snapshot store has already been started")

        initial_stage: WorkspaceLifecycleStage
        if self._mode == "train":
            initial_stage = "train"
        else:
            initial_stage = "infer"

        lifecycle: WorkspaceLifecycle = {
            "status": "running",
            "stage": initial_stage,
            "message": "",
        }

        self._lifecycle_writer.write(lifecycle)
        self._lifecycle = lifecycle

    def set_stage(
        self,
        stage: WorkspaceLifecycleStage,
    ) -> None:
        """Set and immediately publish the active execution stage.

        Training invocations may alternate between ``train`` and ``infer`` when
        validation is executed between epochs. Standalone inference normally
        remains in the ``infer`` stage.

        Repeating the currently active stage has no effect.

        Args:
            stage:
                Execution stage currently active in the workspace.

        Raises:
            RuntimeError:
                If the store has not been started or its lifecycle has already
                reached a terminal status.
        """
        lifecycle = self._require_running_lifecycle()

        if lifecycle["stage"] == stage:
            return

        updated: WorkspaceLifecycle = {
            "status": "running",
            "stage": stage,
            "message": "",
        }

        self._lifecycle_writer.write(updated)
        self._lifecycle = updated

    def update_train(
        self,
        *,
        current_epoch: int,
        total_epoch: int,
        current_batch: int,
        total_batch: int,
        loss: float,
        learning_rates: list[float],
    ) -> None:
        """Update the current training snapshot.

        A complete snapshot is submitted to the throttled atomic writer on
        every call. The first update is published immediately. Later updates
        replace the pending value and are published when the configured
        interval elapses or an explicit flush is requested.

        Args:
            current_epoch:
                One-based current epoch position.

            total_epoch:
                Total number of training epochs.

            current_batch:
                One-based current batch position within the active epoch.

            total_batch:
                Total number of batches in the active epoch.

            loss:
                Scalar backward loss associated with the current batch.

            learning_rates:
                Learning rates ordered according to optimizer parameter groups.

        Raises:
            RuntimeError:
                If the store has not been started, its lifecycle has already
                reached a terminal status, or the workspace is not a training
                workspace.
        """
        self._require_running_lifecycle()

        if self._mode != "train":
            raise RuntimeError(
                "training snapshots are not supported by inference workspaces"
            )

        snapshot: TrainSnapshot = {
            "epoch": {
                "current": current_epoch,
                "total": total_epoch,
            },
            "batch": {
                "current": current_batch,
                "total": total_batch,
            },
            "loss": loss,
            "learning_rates": list(learning_rates),
        }

        self._train_writer.update(snapshot)

    def flush(self) -> None:
        """Force publication of the latest pending training snapshot.

        This method has no effect when no training snapshot has been submitted
        or when the latest snapshot has already been published.

        Raises:
            RuntimeError:
                If the snapshot store has not been started.
        """
        self._require_started()
        self._train_writer.flush()

    def complete(
        self,
        message: str = "",
    ) -> None:
        """Flush pending training state and publish a completed lifecycle.

        If training snapshot publication fails, the completed lifecycle is not
        written and the error propagates to the orchestration layer.

        Args:
            message:
                Optional human-readable information associated with successful
                completion.

        Raises:
            RuntimeError:
                If the store has not been started or its lifecycle has already
                reached a terminal status.
        """
        lifecycle = self._require_running_lifecycle()

        self._train_writer.flush()

        completed: WorkspaceLifecycle = {
            "status": "completed",
            "stage": lifecycle["stage"],
            "message": message,
        }

        self._lifecycle_writer.write(completed)
        self._lifecycle = completed

    def interrupt(
        self,
        message: str,
    ) -> None:
        """Flush pending training state and publish an interrupted lifecycle.

        This method represents the normal interruption path. If training
        snapshot publication fails, the error propagates and the interrupted
        lifecycle is not written. The orchestration layer may then call
        ``publish_interrupted`` to bypass the failed flush operation.

        The message may describe a runtime exception, user interruption,
        checkpoint failure, history failure, or another component-level reason.

        Args:
            message:
                Human-readable reason why execution ended before completion.

        Raises:
            RuntimeError:
                If the store has not been started or its lifecycle has already
                reached a terminal status.
        """
        self._require_running_lifecycle()

        self._train_writer.flush()
        self.publish_interrupted(message)

    def publish_interrupted(
        self,
        message: str,
    ) -> None:
        """Immediately publish an interrupted lifecycle without flushing.

        This method is the lifecycle fallback used by the workspace
        orchestration layer after another persistence or cleanup operation has
        failed. It does not retry training snapshot publication.

        Unlike the normal ``interrupt`` method, this method may replace either
        a running or completed lifecycle. This permits a workspace that was
        initially marked completed to be corrected when a later cleanup
        operation fails.

        Args:
            message:
                Human-readable interruption information supplied by the
                orchestration layer or another workspace component.

        Raises:
            RuntimeError:
                If the snapshot store has not been started.
        """
        lifecycle = self._require_started()

        interrupted: WorkspaceLifecycle = {
            "status": "interrupted",
            "stage": lifecycle["stage"],
            "message": message,
        }

        self._lifecycle_writer.write(interrupted)
        self._lifecycle = interrupted

    def _require_started(self) -> WorkspaceLifecycle:
        """Return the lifecycle after verifying store initialization."""
        if self._lifecycle is None:
            raise RuntimeError("snapshot store has not been started")

        return self._lifecycle

    def _require_running_lifecycle(self) -> WorkspaceLifecycle:
        """Return the lifecycle after verifying that execution is active."""
        lifecycle = self._require_started()

        if lifecycle["status"] != "running":
            raise RuntimeError(
                "snapshot store lifecycle has already finished"
            )

        return lifecycle