"""Standalone inference invocation declarations.

This module defines the complete declarative configuration accepted by the owl
inference client.

A standalone inference invocation combines four configuration domains:

- model and optional checkpoint declarations;
- named inference data declarations;
- inference execution options;
- an output processor declaration.

The invocation layer does not instantiate the model, datasets, dataloaders,
processor, session, or runtime. The client later consumes the declaration,
constructs the required objects, restores optional model snapshot, creates the
inference session and runtime, and starts execution.
"""

from dataclasses import dataclass
from typing import cast

from ..processors.evaluator.binary_mask import BinaryMaskEvaluator
from .components.components import InferComponents
from .data.infer import InferData
from .execution.infer import InferExecution
from .process.process import ProcessDeclaration


_DEFAULT_PROCESS = cast(
    ProcessDeclaration,
    BinaryMaskEvaluator,
)
"""Default processor declaration used by standalone inference.

The cast expresses that ``BinaryMaskEvaluator`` structurally satisfies the
evaluator declaration accepted by ``ProcessDeclaration``.

The evaluator class itself is stored as a declaration. No evaluator instance is
created at module import time.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class InferInvocation:
    """Complete declaration for one standalone inference workflow.

    An inference invocation contains every user-facing declaration required by
    the inference client. The client resolves these declarations into concrete
    framework objects and starts the default inference runtime.

    Attributes:
        components:
            Model and optional checkpoint-load declarations used by the
            inference workflow.

        data:
            Named inference dataset and dataloader declarations.

            Each declared source is converted into an independent dataloader.
            The inference runtime processes these dataloaders by name.

        execution:
            Inference execution settings, including the target device.

        process:
            Evaluator or visualizer construction declaration used to consume
            parsed inference outputs.

            When omitted, the built-in ``BinaryMaskEvaluator`` is used.

    Examples:
        Declare inference with the default binary-mask evaluator:

        >>> invocation = InferInvocation(
        ...     components=components,
        ...     data=infer_data,
        ...     execution=execution,
        ... )

        Declare inference with a custom evaluator:

        >>> invocation = InferInvocation(
        ...     components=components,
        ...     data=infer_data,
        ...     execution=execution,
        ...     process=CustomEvaluator,
        ... )

        Configure a custom evaluator through constructor arguments:

        >>> invocation = InferInvocation(
        ...     components=components,
        ...     data=infer_data,
        ...     execution=execution,
        ...     process=(
        ...         CustomEvaluator,
        ...         {
        ...             "threshold": 0.4,
        ...         },
        ...     ),
        ... )

        Use a visualizer instead of an evaluator:

        >>> invocation = InferInvocation(
        ...     components=components,
        ...     data=infer_data,
        ...     execution=execution,
        ...     process=CustomVisualizer,
        ... )

    Notes:
        The process declaration must describe either an evaluator or a
        visualizer. The client constructs the declared processor and determines
        which inference-session role it satisfies.

        The current default workflow executes one model on one device.
        Alternative execution backends may later be provided by specialized
        clients without changing this invocation declaration.
    """

    components: InferComponents
    data: InferData
    execution: InferExecution
    process: ProcessDeclaration = _DEFAULT_PROCESS


__all__ = [
    "InferInvocation",
]
