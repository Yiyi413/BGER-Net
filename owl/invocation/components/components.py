"""Component collections used by owl invocations.

This module groups individual component declarations into configurations that
can be consumed by a future owl client.

The individual declaration modules describe how each component is constructed:

- ``ModelDeclaration`` describes model construction.
- ``CriterionDeclaration`` describes criterion construction.
- ``OptimizerDeclaration`` describes optimizer-factory resolution.
- ``SchedulerDeclaration`` describes scheduler-factory resolution.
- ``CheckpointLoad`` describes optional startup-snapshot restoration.

This module does not construct any component. It only records which component
declarations belong to a training or inference invocation.

A future client is responsible for resolving the declarations in dependency
order.

For training, the expected construction order is:

    model declaration
    -> model instance

    criterion declaration
    -> criterion instance

    optimizer declaration
    -> optimizer factory
    -> optimizer instance using the model

    scheduler declaration
    -> scheduler factory
    -> scheduler instance using the optimizer and training plan

    optional checkpoint load
    -> restore model-only or complete training snapshot

For inference, only a model and an optional checkpoint load are required.
Training-only components are deliberately excluded from ``InferComponents``.
"""

from dataclasses import dataclass

from .checkpoint import CheckpointLoad
from .criterion import CriterionDeclaration
from .model import ModelDeclaration
from .optimizer import OptimizerDeclaration
from .scheduler import SchedulerDeclaration


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainComponents:
    """Component declarations required to construct a training session.

    This configuration contains declarations rather than instantiated runtime
    objects. The client resolves every declaration before creating
    ``TrainSession``.

    Attributes:
        model:
            Declaration used to construct the training model.

            The declaration may contain only a ``torch.nn.Module`` class or a
            tuple containing the class and its constructor keyword arguments.

        criterion:
            Declaration used to construct the criterion component.

            Like the model declaration, it may contain only a module class or a
            tuple containing the class and its constructor keyword arguments.

        optimizer:
            Declaration used to obtain an ``OptimizerFactory``.

            The declaration may be an already configured optimizer factory or a
            tuple containing an outer factory builder and its user-defined
            keyword arguments.

            After resolving the declaration, the client injects the
            instantiated model into the resulting factory.

        scheduler:
            Declaration used to obtain a ``SchedulerFactory``.

            The declaration may be an already configured scheduler factory, a
            tuple containing an outer factory builder and its user-defined
            keyword arguments, or ``None``.

            ``None`` requests owl's built-in constant learning-rate scheduler.
            The client resolves this form with
            ``owl.optim.scheduler.constant()`` so the resulting training session
            always receives a concrete scheduler instance.

        checkpoint:
            Optional checkpoint-loading configuration applied after the
            components have been constructed.

            Model-only loading restores model parameters while preserving the
            newly constructed optimizer, scheduler, and configured execution
            progress.

            Full-snapshot loading restores model, optimizer, scheduler, and
            training progress for interrupted-run continuation.

            ``None`` means that the training run starts from newly constructed
            component snapshot.
    """

    model: ModelDeclaration
    criterion: CriterionDeclaration
    optimizer: OptimizerDeclaration
    scheduler: SchedulerDeclaration = None
    checkpoint: CheckpointLoad | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InferComponents:
    """Component declarations required to construct an inference session.

    Inference requires only a model component and optional model weights.
    Criterion, optimizer, and scheduler declarations are intentionally absent
    because they are not consumed by ``InferSession``.

    Attributes:
        model:
            Declaration used to construct the inference model.

            The same declaration type is shared with training, allowing one
            model class and constructor configuration to be reused across both
            invocation types.

        checkpoint:
            Optional checkpoint-loading configuration used to restore model
            parameters before inference begins.

            Inference does not contain optimizer, scheduler, or training
            progress snapshot. A future inference client therefore applies only
            the model snapshot represented by this checkpoint configuration.

            ``None`` means that inference uses the snapshot produced directly by
            the model constructor.
    """

    model: ModelDeclaration
    checkpoint: CheckpointLoad | None = None


__all__ = [
    "InferComponents",
    "TrainComponents",
]