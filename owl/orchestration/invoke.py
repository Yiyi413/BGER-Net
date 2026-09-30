"""Invocation orchestration entry point.

This module defines the public orchestration entry point used to dispatch a
complete owl invocation to its corresponding training or inference workflow.

The entry point also owns the workspace context associated with the invocation.
Concrete component resolution, session assembly, runtime execution, and
workspace artifact updates remain implemented by the specialized training and
inference orchestration modules.
"""

from pathlib import Path
from typing import TypeAlias

from ..invocation.infer import InferInvocation
from ..invocation.train import TrainInvocation
from ..schemas.outputs.types import ParsedMetricOutputs
from ..workspace.workspace import Workspace
from .infer import invoke_infer
from .train import invoke_train


Invocation: TypeAlias = TrainInvocation | InferInvocation
"""Complete invocation accepted by owl orchestration."""


InvocationResult: TypeAlias = dict[str, ParsedMetricOutputs] | None
"""Result returned by an owl invocation.

Standalone evaluation returns metrics grouped by dataset name. Training and
visualization workflows return ``None``.
"""


def invoke(
    invocation: Invocation,
) -> InvocationResult:
    """Dispatch and execute one owl invocation inside a workspace.

    A timestamped workspace is created under the parent directory declared by
    the invocation execution configuration. When no parent directory is
    provided, the current working directory is used.

    Training invocations are delegated to ``invoke_train`` and return ``None``.
    Standalone inference invocations are delegated to ``invoke_infer`` and may
    return dataset-level evaluation metrics.

    The workspace context publishes its running lifecycle before specialized
    orchestration begins. Normal completion and uncaught execution exceptions
    are finalized automatically when the context exits.

    Args:
        invocation:
            Complete training or standalone inference declaration.

    Returns:
        Metrics grouped by dataset name for standalone evaluation, otherwise
        ``None``.

    Raises:
        TypeError:
            If the supplied object is not a supported owl invocation.

        OSError:
            If workspace initialization or artifact persistence fails.

        BaseException:
            Any exception raised by the delegated training or inference
            orchestration is propagated after workspace cleanup.
    """
    if isinstance(invocation, TrainInvocation):
        with Workspace(
            parent=_resolve_workspace_parent(
                invocation.execution.workspace
            ),
            mode="train",
        ) as workspace:
            invoke_train(
                invocation,
                workspace=workspace,
            )

        return None

    if isinstance(invocation, InferInvocation):
        with Workspace(
            parent=_resolve_workspace_parent(
                invocation.execution.workspace
            ),
            mode="infer",
        ) as workspace:
            return invoke_infer(
                invocation,
                workspace=workspace,
            )

    raise TypeError(
        "invocation must be a TrainInvocation or InferInvocation"
    )


def _resolve_workspace_parent(
    workspace: Path | None,
) -> Path:
    """Resolve the parent directory used to create an invocation workspace.

    An explicitly configured path is returned unchanged. When the invocation
    does not declare a workspace parent, the process current working directory
    is used.

    Args:
        workspace:
            Optional workspace parent from the invocation execution
            configuration.

    Returns:
        Parent directory under which the timestamped workspace directory is
        created.

    Example:
        When ``workspace`` is ``Path("runs")``, the invocation creates a child
        such as:

        ``runs/workspace-20260719142308456``

        When ``workspace`` is ``None``, the same timestamped child is created
        under ``Path.cwd()``.
    """
    if workspace is None:
        return Path.cwd()

    return workspace


__all__ = [
    "Invocation",
    "InvocationResult",
    "invoke",
]