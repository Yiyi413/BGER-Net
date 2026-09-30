"""Criterion component specification types.

This module defines the criterion forms accepted by owl invocations.

A criterion may be supplied as an existing ``torch.nn.Module`` instance when
the caller needs to control its construction or initialization before
invocation:

    criterion = MyCriterion(
        edge_weight=2.0,
        label_weight=1.0,
    )

    invocation = TrainInvocation(
        criterion=criterion,
        ...
    )

A criterion class may be supplied directly when its constructor does not
require arguments:

    criterion=MyCriterion

A criterion class may also be paired with keyword arguments:

    criterion=(
        MyCriterion,
        {
            "edge_weight": 2.0,
            "label_weight": 1.0,
        },
    )

During ``owl.invoke()``, an existing criterion instance is used as supplied.
Class-based specifications are instantiated by ``resolve_criterion()``:

    criterion = criterion_type(**criterion_kwargs)

Accepting instances allows callers to assemble compound losses, initialize
internal modules, or configure task-specific supervision before handing the
criterion to owl. Class-based forms remain useful for concise and
configuration-oriented invocations.

Regardless of how the criterion is supplied, the resolved module is expected
to follow the criterion call convention defined in ``owl.schemas.calls``.
"""

from typing import Any, Mapping, TypeAlias

from torch.nn import Module


CriterionType: TypeAlias = type[Module]
"""Class used to construct an owl criterion component.

The class must derive from ``torch.nn.Module``. Its constructor signature is
left unrestricted because criterion implementations may expose different
task-specific initialization options.

The ``resolve_criterion()`` function instantiates this type when resolving the
training invocation.
"""


CriterionArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to a criterion constructor.

The mapping is expanded as keyword arguments when resolving a configured
criterion specification:

    criterion = criterion_type(**dict(criterion_args))

A generic mapping is used because criterion implementations may expose
task-specific options that cannot be represented by one fixed owl schema.

Typical values include loss weights, reduction modes, class-balancing
parameters, and supervision-specific configuration.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored specification.
"""


CriterionDeclaration: TypeAlias = (
    Module
    | CriterionType
    | tuple[CriterionType, CriterionArgs]
)
"""Criterion specification accepted by an owl invocation.

An existing module instance may be supplied when construction has already been
completed by the caller:

    MyCriterion(...)

The compact declarative form contains only the criterion class:

    MyCriterion

The configured declarative form contains the criterion class and its keyword
arguments:

    (
        MyCriterion,
        {
            "edge_weight": 2.0,
            "reduction": "mean",
        },
    )

Existing instances are used without reconstruction. Declarative forms are
resolved into a new instance during ``owl.invoke()``.

The configured form supports keyword arguments only. Its second tuple element
is therefore expanded with ``**kwargs`` rather than positional expansion.
"""


def resolve_criterion(declaration: CriterionDeclaration) -> Module:
    """Resolve a criterion declaration into a module instance.

    Existing module instances are returned without reconstruction. A module
    class is instantiated without arguments, while a configured declaration is
    instantiated with its associated keyword arguments.

    Constructor errors are allowed to propagate unchanged so callers retain the
    original exception type and traceback produced by the criterion
    implementation.

    Args:
        declaration: Criterion instance or declarative construction
            specification.

    Returns:
        The criterion module used by the training invocation.

    Raises:
        TypeError: If the declaration does not match a supported criterion
            form.
    """
    if isinstance(declaration, Module):
        return declaration

    if isinstance(declaration, type) and issubclass(declaration, Module):
        return declaration()

    if isinstance(declaration, tuple) and len(declaration) == 2:
        criterion_type, criterion_args = declaration

        if (
            isinstance(criterion_type, type)
            and issubclass(criterion_type, Module)
            and isinstance(criterion_args, Mapping)
        ):
            return criterion_type(**dict(criterion_args))

    raise TypeError(
        "criterion declaration must be a torch.nn.Module instance, "
        "a torch.nn.Module subclass, or a tuple containing a module "
        "subclass and its keyword arguments"
    )


__all__ = [
    "CriterionArgs",
    "CriterionDeclaration",
    "CriterionType",
    "resolve_criterion",
]
