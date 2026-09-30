"""Standalone inference orchestration.

This module resolves one ``InferInvocation`` into concrete runtime objects,
constructs the corresponding inference session and runtime, executes the
declared workflow, and persists standalone inference artifacts.

Component, data, checkpoint, and process resolution remain implemented by their
owning invocation modules. The active workspace is supplied by the public
invocation orchestration entry point.
"""

from ..invocation.components.checkpoint import resolve_checkpoint
from ..invocation.components.model import resolve_model
from ..invocation.data.infer import resolve_infer_data
from ..invocation.infer import InferInvocation
from ..invocation.process.process import resolve_process
from ..runtime.infer import InferRuntime
from ..runtime.session.infer import InferSession
from ..schemas.outputs.types import ParsedMetricOutputs
from ..workspace.workspace import Workspace


def invoke_infer(
    invocation: InferInvocation,
    *,
    workspace: Workspace,
) -> dict[str, ParsedMetricOutputs] | None:
    """Resolve and execute one standalone inference invocation.

    The active workspace stage is set to inference before component resolution
    begins.

    The model is constructed first and moved to the configured execution
    device. An optional checkpoint is then restored before the inference session
    is created.

    Inference data declarations are resolved into named dataloaders. The process
    declaration is constructed and classified as either an evaluator or a
    visualizer, then injected into the corresponding ``InferSession`` fields.

    Evaluator results are appended to standalone evaluation history before being
    returned. Visualizer workflows receive the workspace visual directory and
    return ``None``.

    Args:
        invocation:
            Complete standalone inference declaration.

        workspace:
            Active workspace owned by the invocation orchestration entry point.

    Returns:
        Metrics grouped by dataset name when the resolved process is an
        evaluator. Returns ``None`` when the resolved process is a visualizer.

    Raises:
        TypeError:
            If one of the component, data, or process declarations is invalid.

        ValueError:
            If the resolved process is ambiguous, inference data is invalid, or
            generated visualization outputs are invalid.

        RuntimeError:
            If the model output does not contain the values required by the
            selected processor.

        OSError:
            If workspace artifact persistence or visualization output fails.
    """
    workspace.set_stage("infer")

    execution = invocation.execution
    components = invocation.components

    model = resolve_model(components.model)
    model.to(execution.device)

    resolve_checkpoint(
        components.checkpoint,
        model=model,
        model_only=True,
    )

    dataloaders = resolve_infer_data(invocation.data)
    process = resolve_process(invocation.process)

    session = InferSession(
        model=model,
        device=execution.device,
        dataloaders=dataloaders,
        evaluator=process.evaluator,
        visualizer=process.visualizer,
    )

    visualization_dir = (
        workspace.visual_dir()
        if process.visualizer is not None
        else None
    )

    runtime = InferRuntime(
        visualization_dir=visualization_dir,
    )

    results = runtime.run(session)

    if results is not None:
        workspace.append_evaluation_history(
            results=results,
        )

    return results


__all__ = [
    "invoke_infer",
]