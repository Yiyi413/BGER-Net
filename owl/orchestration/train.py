"""Training workflow orchestration.

This module resolves one ``TrainInvocation`` into concrete framework objects,
constructs the required training and optional inference sessions, creates their
runtimes, and starts execution.

The orchestration layer owns dependency ordering and object assembly. Concrete
declaration resolution remains implemented by the corresponding invocation
modules, while training and inference control flow remains implemented by the
runtime layer.
"""

from torch import device as TorchDevice
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..invocation.components.checkpoint import (
    CheckpointLoad,
    resolve_checkpoint,
)
from ..invocation.components.criterion import resolve_criterion
from ..invocation.components.model import resolve_model
from ..invocation.components.optimizer import resolve_optimizer
from ..invocation.components.scheduler import resolve_scheduler
from ..invocation.data.infer import resolve_infer_data
from ..invocation.data.train import resolve_train_data
from ..invocation.process.evaluator import resolve_evaluator
from ..invocation.train import TrainInference, TrainInvocation
from ..runtime.infer import InferRuntime
from ..runtime.session.infer import InferSession
from ..runtime.session.train import TrainSession
from ..runtime.train import TrainRuntime
from ..workspace.workspace import Workspace


def _restore_training_state(
    declaration: CheckpointLoad | None,
    *,
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> int:
    """Restore optional training snapshot and return the starting epoch.

    Model-only checkpoint loading restores model parameters and leaves training
    progress unchanged, so training starts from epoch zero.

    Full-snapshot checkpoint loading restores the model, optimizer, scheduler, and
    completed epoch. Training resumes from the following epoch.

    Args:
        declaration:
            Optional checkpoint loading declaration.

        model:
            Resolved training model.

        optimizer:
            Resolved optimizer associated with the training model.

        scheduler:
            Resolved learning-rate scheduler associated with the optimizer.

    Returns:
        Zero-based epoch index from which training should begin.
    """

    checkpoint_epoch = resolve_checkpoint(
        declaration,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if checkpoint_epoch is None:
        return 0

    return checkpoint_epoch + 1


def _build_infer_session(
    declaration: TrainInference,
    *,
    model: Module,
    device: TorchDevice,
) -> InferSession:
    """Construct the optional evaluation session used during training.

    The inference session shares the exact model instance and device used by the
    training session. This guarantees that validation observes the parameters
    updated by the training optimizer.

    Args:
        declaration:
            Training-time inference declaration containing validation data and
            evaluator construction configuration.

        model:
            Resolved model shared with the training session.

        device:
            Device shared with the training session.

    Returns:
        Inference session configured for dataset-level evaluation.
    """

    dataloaders = resolve_infer_data(declaration.data)
    evaluator = resolve_evaluator(declaration.evaluator)

    return InferSession(
        model=model,
        device=device,
        dataloaders=dataloaders,
        evaluator=evaluator,
    )


def invoke_train(
    invocation: TrainInvocation,
    *,
    workspace: Workspace,
) -> None:
    """Resolve and execute one training invocation.

    The active workspace is supplied by the public invocation entry point.
    Training components and sessions are resolved here, while epoch, batch,
    validation, artifact persistence, and checkpoint-saving control flow remain
    owned by ``TrainRuntime``.

    Args:
        invocation:
            Complete training declaration.

        workspace:
            Active workspace owned by the invocation orchestration entry point.
    """
    workspace.set_stage("train")

    components = invocation.components
    execution = invocation.execution

    # Resolve the training data before constructing components whose
    # configuration depends on the complete training plan.
    train_dataloader = resolve_train_data(invocation.data)

    # Resolve and prepare the core trainable components.
    model = resolve_model(components.model)
    model.to(execution.device)

    criterion = resolve_criterion(components.criterion)
    criterion.to(execution.device)

    # Stateful training components must be constructed in dependency order:
    # the optimizer depends on the model, and the scheduler depends on the
    # optimizer together with the complete training plan.
    optimizer = resolve_optimizer(
        components.optimizer,
        model=model,
    )

    total_steps = execution.total_epochs * len(train_dataloader)

    scheduler = resolve_scheduler(
        components.scheduler,
        optimizer=optimizer,
        total_epochs=execution.total_epochs,
        total_steps=total_steps,
    )

    # Restore optional checkpoint snapshot before constructing the session.
    # Scheduler factories always receive the complete training plan, while the
    # restored epoch only determines where runtime execution resumes.
    start_epoch = _restore_training_state(
        components.checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Assemble the stateful training boundary consumed by TrainRuntime.
    train_session = TrainSession(
        model=model,
        device=execution.device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        total_epochs=execution.total_epochs,
        start_epoch=start_epoch,
    )

    # Training-time evaluation shares the exact model instance used by the
    # training session so validation observes the latest parameters.
    infer_session: InferSession | None = None

    if invocation.inference is not None:
        infer_session = _build_infer_session(
            invocation.inference,
            model=model,
            device=execution.device,
        )

    # Runtime owns training, validation, persistence, and checkpoint control
    # flow. Orchestration only resolves dependencies and assembles objects.
    infer_runtime = InferRuntime()

    train_runtime = TrainRuntime(
        infer_runtime=infer_runtime,
        workspace=workspace,
        checkpoint_save=execution.checkpoint,
    )

    train_runtime.run(
        train_session=train_session,
        infer_session=infer_session,
    )


__all__ = [
    "invoke_train",
]