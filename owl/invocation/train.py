"""Training invocation declarations.

This module defines the complete declarative configuration accepted by owl
training orchestration.

A training invocation combines four independent configuration domains:

- training component declarations;
- training data declarations;
- training execution options;
- optional evaluation declarations executed during training.

The invocation layer does not instantiate models, datasets, optimizers,
schedulers, evaluators, sessions, or runtimes. It only records the user's
configuration. The orchestration layer later consumes this declaration,
constructs the required objects, restores optional checkpoint snapshot, creates
the corresponding sessions and runtimes, and starts execution.

Training-time evaluation is represented by ``TrainInference`` rather than by
placing evaluation fields directly on ``TrainInvocation``. This prevents invalid
partial configurations, such as declaring an evaluator without providing
inference data, while keeping the complete validation workflow optional.

Training-time inference is intentionally restricted to evaluation. Visualization
belongs to standalone inference because training workflows normally consume
dataset-level metrics after each epoch rather than generating image artifacts.
"""

from dataclasses import dataclass
from typing import cast

from ..processors.evaluator.binary_mask import BinaryMaskEvaluator
from .components.components import TrainComponents
from .data.infer import InferData
from .data.train import TrainData
from .execution.train import TrainExecution
from .process.evaluator import EvaluatorDeclaration


_DEFAULT_EVALUATOR = cast(
    EvaluatorDeclaration,
    BinaryMaskEvaluator,
)
"""Default evaluator declaration used by training-time inference.

The cast expresses that ``BinaryMaskEvaluator`` is a callable whose instances
structurally satisfy the evaluator protocol accepted by
``EvaluatorDeclaration``.

The evaluator class itself is stored as a declaration. No evaluator instance is
created at module import time.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainInference:
    """Optional evaluation workflow executed during training.

    ``TrainInference`` groups the inference data and evaluator declaration
    required to construct an inference session that shares the training model
    and device.

    When attached to ``TrainInvocation``, the default training runtime executes
    this evaluation workflow after every completed training epoch.

    The declaration does not contain model, checkpoint, device, session, or
    runtime settings. Those resources are inherited from or assembled around
    the enclosing training invocation by the orchestration layer.

    Attributes:
        data:
            Named inference dataset declarations used for validation.

            Each declared source is resolved into an independent dataloader.
            The evaluator processes every dataloader and returns metrics grouped
            by dataset name.

        evaluator:
            Evaluator construction declaration used to accumulate inference
            outputs and compute dataset-level metrics.

            The declaration may contain either an evaluator constructor or a
            tuple containing the constructor and its keyword arguments.

            When omitted, the built-in ``BinaryMaskEvaluator`` is used.

    Examples:
        Declare validation with the default binary-mask evaluator:

        >>> inference = TrainInference(
        ...     data=validation_data,
        ... )

        Declare validation with a custom evaluator:

        >>> inference = TrainInference(
        ...     data=validation_data,
        ...     evaluator=CustomEvaluator,
        ... )

        Configure a custom evaluator through constructor arguments:

        >>> inference = TrainInference(
        ...     data=validation_data,
        ...     evaluator=(
        ...         CustomEvaluator,
        ...         {
        ...             "threshold": 0.4,
        ...         },
        ...     ),
        ... )

    Notes:
        Training-time inference accepts evaluators only. Visualizers are
        supported by standalone ``InferInvocation`` workflows.

        Evaluator instances are not accepted directly. A fresh evaluator is
        constructed for the training invocation because evaluators may retain
        mutable aggregation snapshot between ``reset``, ``update``, and ``compute``
        calls.
    """

    data: InferData
    evaluator: EvaluatorDeclaration = _DEFAULT_EVALUATOR


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainInvocation:
    """Complete declaration for one training workflow.

    A training invocation contains every user-facing declaration required by
    owl training orchestration. The orchestration layer resolves these
    declarations into concrete framework objects and starts the default
    training runtime.

    Attributes:
        components:
            Model, criterion, optimizer, scheduler, and optional checkpoint-load
            declarations used by the training workflow.

        data:
            Training dataset and dataloader declarations.

        execution:
            Training execution settings, including the total epoch count,
            target device, and checkpoint-saving behavior.

        inference:
            Optional evaluation workflow executed after each completed training
            epoch.

            When ``None``, training runs without validation.

    Examples:
        Declare training without evaluation:

        >>> invocation = TrainInvocation(
        ...     components=components,
        ...     data=train_data,
        ...     execution=execution,
        ... )

        Declare training with validation after every epoch:

        >>> invocation = TrainInvocation(
        ...     components=components,
        ...     data=train_data,
        ...     execution=execution,
        ...     inference=TrainInference(
        ...         data=validation_data,
        ...     ),
        ... )

        Declare training with a custom evaluator:

        >>> invocation = TrainInvocation(
        ...     components=components,
        ...     data=train_data,
        ...     execution=execution,
        ...     inference=TrainInference(
        ...         data=validation_data,
        ...         evaluator=CustomEvaluator,
        ...     ),
        ... )

    Notes:
        This declaration intentionally targets the default single-model,
        single-device training workflow.

        Alternative execution strategies, such as adversarial training or
        distributed training, require different component assembly and control
        flow. They may later be supported through specialized orchestration,
        sessions, runtimes, or internal backend configuration without changing
        the common training invocation API.
    """

    components: TrainComponents
    data: TrainData
    execution: TrainExecution
    inference: TrainInference | None = None


__all__ = [
    "TrainInference",
    "TrainInvocation",
]
