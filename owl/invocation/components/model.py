"""Model component specification types.

This module defines the model forms accepted by owl invocations.

A model may be supplied as an existing ``torch.nn.Module`` instance when the
caller needs to control its construction or initialization before invocation:

    model = MyModel(
        backbone="res2net50",
        pretrained=True,
    )

    invocation = TrainInvocation(
        model=model,
        ...
    )

A model class may be supplied directly when its constructor does not require
arguments:

    model=MyModel

A model class may also be paired with keyword arguments:

    model=(
        MyModel,
        {
            "backbone": "res2net50",
            "pretrained": True,
        },
    )

During ``owl.invoke()``, an existing model instance is used as supplied.
Class-based specifications are instantiated by the invocation resolver:

    model = model_type(**model_kwargs)

Accepting instances allows callers to perform initialization, load external
weights, freeze parameters, or replace submodules before handing the model to
owl. Class-based forms remain useful for concise and configuration-oriented
invocations.

Regardless of how the model is supplied, the resolved module is expected to
follow the model call conventions defined in ``owl.schemas.calls``.
"""

from typing import Any, Mapping, TypeAlias

from torch.nn import Module


ModelType: TypeAlias = type[Module]
"""Class used to construct an owl model component.

The class must derive from ``torch.nn.Module``. Its constructor signature is
left unrestricted because model implementations may require different
initialization options.

The invocation resolver instantiates this type after the complete invocation
has been validated.
"""


ModelArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to a model constructor.

The mapping is expanded as keyword arguments when resolving a configured model
specification:

    model = model_type(**dict(model_args))

A generic mapping is used because model constructors are user-defined and owl
does not prescribe a fixed constructor schema.

Invocation objects may copy this mapping during initialization so later
mutations to caller-owned configuration do not alter the stored specification.
"""


ModelDeclaration: TypeAlias = (
    Module
    | ModelType
    | tuple[ModelType, ModelArgs]
)
"""Model specification accepted by an owl invocation.

An existing module instance may be supplied when construction has already been
completed by the caller:

    MyModel(...)

The compact declarative form contains only the model class:

    MyModel

The configured declarative form contains the model class and its keyword
arguments:

    (
        MyModel,
        {
            "channels": 64,
            "pretrained": True,
        },
    )

Existing instances are used without reconstruction. Declarative forms are
resolved into a new instance during ``owl.invoke()``.

The configured form supports keyword arguments only. Its second tuple element
is therefore expanded with ``**kwargs`` rather than positional expansion.
"""


def resolve_model(declaration: ModelDeclaration) -> Module:
    """Resolve a model declaration into a module instance.

    Existing module instances are returned without reconstruction. A module
    class is instantiated without arguments, while a configured declaration is
    instantiated with its associated keyword arguments.

    Constructor errors are allowed to propagate unchanged so callers retain the
    original exception type and traceback produced by the model implementation.

    Args:
        declaration: Model instance or declarative construction specification.

    Returns:
        The module instance used by the invocation.

    Raises:
        TypeError: If the declaration does not match a supported model form.
    """
    if isinstance(declaration, Module):
        return declaration

    if isinstance(declaration, type) and issubclass(declaration, Module):
        return declaration()

    if isinstance(declaration, tuple) and len(declaration) == 2:
        model_type, model_args = declaration

        if (
            isinstance(model_type, type)
            and issubclass(model_type, Module)
            and isinstance(model_args, Mapping)
        ):
            return model_type(**dict(model_args))

    raise TypeError(
        "model declaration must be a torch.nn.Module instance, "
        "a torch.nn.Module subclass, or a tuple containing a module "
        "subclass and its keyword arguments"
    )

__all__ = [
    "ModelArgs",
    "ModelDeclaration",
    "ModelType",
    "resolve_model",
]