"""Scheduler component declaration and resolution types.

This module defines the learning-rate scheduler forms accepted by owl
invocations and provides the resolver that constructs the final scheduler
instance.

Scheduler construction is delayed until the optimizer and complete training
plan have been resolved. Owl therefore receives a scheduler constructor rather
than an already instantiated scheduler.

A scheduler constructor is a callable that accepts the resolved optimizer and
training-plan values:

    scheduler = scheduler_constructor(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

An invocation may supply an already configured constructor directly:

    scheduler_constructor = create_scheduler(
        warmup_steps=1000,
        min_lr_ratio=0.01,
    )

    invocation = TrainInvocation(
        scheduler=scheduler_constructor,
        ...
    )

An outer configuration callable may instead be paired with the keyword
arguments used to create the constructor:

    invocation = TrainInvocation(
        scheduler=(
            create_scheduler,
            {
                "warmup_steps": 1000,
                "min_lr_ratio": 0.01,
            },
        ),
        ...
    )

During resolution, owl first creates the configured constructor:

    scheduler_constructor = create_scheduler(
        warmup_steps=1000,
        min_lr_ratio=0.01,
    )

Owl then injects the resolved optimizer and training-plan values:

    scheduler = scheduler_constructor(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

The absence of an explicit scheduling strategy may be expressed with ``None``:

    invocation = TrainInvocation(
        scheduler=None,
        ...
    )

In that form, owl uses its built-in constant learning-rate constructor.

Concrete ``LRScheduler`` instances are deliberately excluded because a
scheduler must be associated with the final optimizer selected by the
invocation.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...optim.scheduler import constant
from ...schemas.optim import SchedulerConstructor


SchedulerArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to a scheduler configuration callable.

The mapping is expanded as keyword arguments when resolving a configured
scheduler declaration:

    scheduler_constructor = configure_scheduler(
        **dict(scheduler_args),
    )

For example:

    scheduler=(
        create_scheduler,
        {
            "warmup_steps": 1000,
            "min_lr_ratio": 0.01,
        },
    )

is resolved as:

    scheduler_constructor = create_scheduler(
        warmup_steps=1000,
        min_lr_ratio=0.01,
    )

A generic mapping is used because scheduler configuration callables may expose
different options such as warmup duration, polynomial power, minimum
learning-rate ratio, milestones, and decay factors.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored declaration.
"""


SchedulerDeclaration: TypeAlias = (
    SchedulerConstructor
    | tuple[
        Callable[..., SchedulerConstructor],
        SchedulerArgs,
    ]
    | None
)
"""Scheduler construction specification accepted by an owl invocation.

Three declaration forms are supported.

An already configured ``SchedulerConstructor`` may be supplied directly:

    scheduler_constructor = poly(
        power=0.9,
    )

    invocation = TrainInvocation(
        scheduler=scheduler_constructor,
        ...
    )

A scheduler configuration callable may be paired with the keyword arguments
used to create the constructor:

    invocation = TrainInvocation(
        scheduler=(
            poly,
            {
                "power": 0.9,
            },
        ),
        ...
    )

That form is resolved in two stages:

    scheduler_constructor = poly(
        power=0.9,
    )

    scheduler = scheduler_constructor(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

Finally, ``None`` requests owl's built-in constant learning-rate strategy:

    invocation = TrainInvocation(
        scheduler=None,
        ...
    )

This is equivalent to obtaining the built-in constructor first:

    scheduler_constructor = constant()

and then invoking it through the same constructor protocol:

    scheduler = scheduler_constructor(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

Concrete ``LRScheduler`` instances are not accepted. Scheduler construction
depends on the final optimizer and resolved training plan, so it must occur
after those resources are available.
"""


def resolve_scheduler(
    declaration: SchedulerDeclaration,
    *,
    optimizer: Optimizer,
    total_epochs: int,
    total_steps: int,
) -> LRScheduler:
    """Resolve a scheduler declaration into a scheduler instance.

    An already configured scheduler constructor is invoked directly with the
    supplied optimizer and training-plan values.

    A configured declaration first invokes its outer configuration callable
    with the stored keyword arguments:

        scheduler_constructor = configure_scheduler(
            **scheduler_args,
        )

    The resulting constructor is then invoked with resources resolved by owl:

        scheduler = scheduler_constructor(
            optimizer=optimizer,
            total_epochs=total_epochs,
            total_steps=total_steps,
        )

    When ``declaration`` is ``None``, owl obtains its built-in constant
    scheduler constructor and invokes it through the same path.

    Exceptions raised by user-defined configuration callables and constructors
    are allowed to propagate unchanged so callers retain the original exception
    type and traceback.

    Args:
        declaration:
            Configured scheduler constructor, a scheduler configuration callable
            paired with its keyword arguments, or ``None`` for the built-in
            constant scheduler.
        optimizer:
            Resolved optimizer whose learning rate will be scheduled.
        total_epochs:
            Total number of epochs in the training plan.
        total_steps:
            Total number of planned optimizer updates.

    Returns:
        Learning-rate scheduler associated with the supplied optimizer.

    Raises:
        TypeError:
            If the declaration does not match a supported scheduler form, if
            the configuration callable does not return a callable constructor,
            or if the constructor does not return an ``LRScheduler`` instance.
    """
    if declaration is None:
        scheduler_constructor = constant()

    elif isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "scheduler constructor declaration must contain exactly "
                "a configuration callable and its keyword arguments"
            )

        configure_scheduler, scheduler_args = declaration

        if (
            not callable(configure_scheduler)
            or not isinstance(scheduler_args, Mapping)
        ):
            raise TypeError(
                "scheduler constructor declaration must contain a callable "
                "and a mapping of keyword arguments"
            )

        scheduler_constructor = configure_scheduler(
            **dict(scheduler_args),
        )

    else:
        scheduler_constructor = declaration

    if not callable(scheduler_constructor):
        raise TypeError(
            "scheduler declaration must resolve to a callable "
            "SchedulerConstructor"
        )

    scheduler = scheduler_constructor(
        optimizer=optimizer,
        total_epochs=total_epochs,
        total_steps=total_steps,
    )

    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            "scheduler constructor must return a "
            "torch.optim.lr_scheduler.LRScheduler instance"
        )

    return scheduler


__all__ = [
    "SchedulerArgs",
    "SchedulerDeclaration",
    "resolve_scheduler",
]