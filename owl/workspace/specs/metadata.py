"""Workspace metadata schema definitions.

This module defines the static metadata stored in ``metadata.json`` for each
workspace. Metadata describes the identity and format of an invocation
workspace and does not contain mutable runtime snapshot.
"""

from typing import Literal, TypedDict


WorkspaceMode = Literal["train", "infer"]
"""Execution modes supported by workspace metadata."""


class WorkspaceMetadata(TypedDict):
    """Static metadata written when a workspace is created.

    Attributes:
        schema_version:
            Version of the workspace artifact schema. This version is
            independent of the installed Owl package version.

        workspace_id:
            Stable identifier assigned to the workspace. By default, this is
            the generated workspace directory name.

        created_at:
            ISO 8601 timestamp representing when the workspace was created.
            The timestamp includes the local UTC offset and millisecond
            precision.

        mode:
            Invocation execution mode that created the workspace.

        owl_version:
            Version of Owl responsible for creating the workspace.

    Example:
        Metadata for a training workspace:

        >>> metadata: WorkspaceMetadata = {
        ...     "schema_version": 1,
        ...     "workspace_id": "workspace-20260719134530123",
        ...     "created_at": "2026-07-19T13:45:30.123+08:00",
        ...     "mode": "train",
        ...     "owl_version": "0.0.2",
        ... }
    """

    schema_version: int
    workspace_id: str
    created_at: str
    mode: WorkspaceMode
    owl_version: str