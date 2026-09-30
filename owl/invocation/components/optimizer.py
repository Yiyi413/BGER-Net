"""Optimizer component declaration and resolution types.

This module defines the optimizer forms accepted by owl invocations and
provides the resolver that constructs the final optimizer instance.

Optimizer construction is delayed until the invocation model has been
resolved because the optimizer must reference that model's parameters.

An invocation may receive an already configured ``OptimizerConstructor``:

    optimizer_constructor = create_optimizer(
        lr=1e-4,
        weight_decay=1e-2,
    )

    invocation = TrainInvocation(
        optimizer=optimizer_constructor,
        ...
    )

The constructor receives the resolved model and creates the optimizer:

    optimizer = optimizer_constructor(
        model=model,
    )

Alternatively, an invocation may contain an outer configuration callable
together with the keyword arguments used to create the constructor:

    optimizer=(
        create_optimizer,
        {
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
    )

During ``owl.invoke()``, ``resolve_optimizer()`` converts either form into a
``torch.optim.Optimizer`` instance.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from torch.nn import Module
from torch.optim import Optimizer

from ...schemas.optim import OptimizerConstructor


OptimizerArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to an optimizer configuration callable.

The mapping is expanded when ``resolve_optimizer()`` resolves a configured
constructor declaration:

    optimizer_constructor = configure_optimizer(
        **dict(optimizer_args),
    )

A generic mapping is used because optimizer configuration callables may expose
different options such as learning rates, weight decay, momentum, epsilon, and
parameter-group settings.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored declaration.
"""


OptimizerDeclaration: TypeAlias = (
    OptimizerConstructor
    | tuple[
        Callable[..., OptimizerConstructor],
        OptimizerArgs,
    ]
)
"""Optimizer construction specification accepted by an owl invocation.

The direct form contains an already configured ``OptimizerConstructor``:

    optimizer_constructor

This form is typically produced by calling an outer configuration function:

    optimizer_constructor = adamw(
        lr=1e-4,
        weight_decay=1e-2,
    )

The configured form contains that outer callable and the keyword arguments used
to produce the constructor:

    (
        adamw,
        {
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
    )

The configured form is resolved in two stages:

    optimizer_constructor = adamw(
        **optimizer_args,
    )

    optimizer = optimizer_constructor(
        model=model,
    )

Concrete ``Optimizer`` instances are deliberately excluded. The optimizer must
be constructed after the final model instance has been resolved so its
parameter groups reference the model used by the training session.
"""


def resolve_optimizer(
    declaration: OptimizerDeclaration,
    *,
    model: Module,
) -> Optimizer:
    """Resolve an optimizer declaration for a model.

    An already configured optimizer constructor is invoked directly with the
    supplied model. A configured declaration first invokes its outer callable
    with the stored keyword arguments and then invokes the resulting
    constructor with the resolved model.

    Exceptions raised by user-defined configuration callables and constructors
    are allowed to propagate unchanged so callers retain the original exception
    type and traceback.

    Args:
        declaration:
            Configured optimizer constructor or an outer configuration callable
            paired with its keyword arguments.
        model:
            Resolved model whose parameters will be optimized.

    Returns:
        Optimizer constructed for the supplied model.

    Raises:
        TypeError:
            If the declaration does not match a supported optimizer form, if
            the outer callable does not return a callable constructor, or if
            the constructor does not return a ``torch.optim.Optimizer``
            instance.
    """
    if isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "optimizer constructor declaration must contain exactly "
                "a configuration callable and its keyword arguments"
            )

        configure_optimizer, optimizer_args = declaration

        if (
            not callable(configure_optimizer)
            or not isinstance(optimizer_args, Mapping)
        ):
            raise TypeError(
                "optimizer constructor declaration must contain a callable "
                "and a mapping of keyword arguments"
            )

        optimizer_constructor = configure_optimizer(
            **dict(optimizer_args),
        )
    else:
        optimizer_constructor = declaration

    if not callable(optimizer_constructor):
        raise TypeError(
            "optimizer declaration must resolve to a callable "
            "OptimizerConstructor"
        )

    optimizer = optimizer_constructor(
        model=model,
    )

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            "optimizer constructor must return a "
            "torch.optim.Optimizer instance"
        )

    return optimizer


__all__ = [
    "OptimizerArgs",
    "OptimizerDeclaration",
    "resolve_optimizer",
]