"""Workspace lifecycle snapshot schema definitions.

This module defines the lifecycle snapshot stored in
``snapshot/lifecycle.json`` for each workspace. The lifecycle snapshot
describes whether an invocation is running, completed normally, or was
interrupted before completion, together with its latest execution stage.
"""

from typing import Literal, TypedDict


WorkspaceLifecycleStatus = Literal[
    "running",
    "completed",
    "interrupted",
]
"""Lifecycle statuses supported by workspace snapshot artifacts."""


WorkspaceLifecycleStage = Literal[
    "train",
    "infer",
]
"""Execution stages supported by workspace lifecycle snapshots."""


class WorkspaceLifecycle(TypedDict):
    """Lifecycle snapshot of a workspace invocation.

    ``stage`` describes the runtime stage most recently entered by the
    invocation. During a training invocation, this value may alternate between
    ``train`` and ``infer`` when evaluation is executed between training
    epochs.

    For completed or interrupted workspaces, ``stage`` retains the final stage
    entered before execution ended.

    Attributes:
        status:
            Current lifecycle status of the invocation.

            ``running`` indicates that the invocation is still executing.
            ``completed`` indicates that it finished normally.
            ``interrupted`` indicates that execution ended before normal
            completion because of an exception, user interruption, or another
            runtime failure.

        stage:
            Current or most recently entered execution stage.

            ``train`` indicates that the training runtime is active.
            ``infer`` indicates that an inference runtime is active, including
            evaluation performed during a training invocation.

        message:
            Human-readable lifecycle message. This value is normally empty for
            running and completed workspaces. Interrupted workspaces should use
            it to describe the interruption reason.

    Example:
        Lifecycle snapshot while training:

        >>> lifecycle: WorkspaceLifecycle = {
        ...     "status": "running",
        ...     "stage": "train",
        ...     "message": "",
        ... }

        Lifecycle snapshot while evaluating during training:

        >>> lifecycle = {
        ...     "status": "running",
        ...     "stage": "infer",
        ...     "message": "",
        ... }

        Lifecycle snapshot interrupted during inference:

        >>> lifecycle = {
        ...     "status": "interrupted",
        ...     "stage": "infer",
        ...     "message": "evaluation output is missing",
        ... }
    """

    status: WorkspaceLifecycleStatus
    stage: WorkspaceLifecycleStage
    message: str