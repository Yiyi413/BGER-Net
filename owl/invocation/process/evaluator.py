"""Evaluator process declarations and resolution.

This module defines the declarative forms accepted by owl invocations for
inference evaluator construction and provides the resolver that constructs a
fresh evaluator instance.

An evaluator is a stateful inference processor that accumulates model outputs
over one complete dataloader and computes dataset-level metrics. Its runtime
protocol is defined by ``owl.schemas.processors.Evaluator``.

Evaluator implementations are declared as constructors rather than instantiated
objects. The invocation orchestration layer constructs a fresh evaluator before
creating an inference session.

An evaluator declaration may contain either:

- an evaluator constructor;
- a tuple containing the evaluator constructor and its keyword arguments.

An evaluator without additional constructor arguments may be supplied directly:

    evaluator=BinaryMaskEvaluator

An evaluator that requires configuration may be paired with a mapping:

    evaluator=(
        BinaryMaskEvaluator,
        {
            "threshold": 0.4,
        },
    )

Custom evaluators follow the same declaration format:

    evaluator=(
        ProgressiveEvaluator,
        {
            "stages": 3,
            "threshold": 0.5,
        },
    )
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from ...schemas.processors import Evaluator


EvaluatorType: TypeAlias = Callable[..., Evaluator]
"""Callable used to construct an owl evaluator.

The declaration stores an evaluator constructor rather than an already
constructed evaluator instance.

A class implementing the evaluator protocol is callable and therefore satisfies
this type:

    evaluator_type = BinaryMaskEvaluator
    evaluator = evaluator_type()

Using constructors ensures that orchestration can create a fresh evaluator for
every resolved inference configuration. This is important because evaluators may
hold mutable aggregation snapshot between ``reset``, ``update``, and ``compute``
calls.

The constructor signature remains unrestricted because concrete evaluator
implementations may require different initialization options.
"""


EvaluatorArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to an evaluator constructor.

A configured evaluator declaration is resolved by expanding the mapping:

    evaluator = evaluator_type(
        **dict(evaluator_args),
    )

A generic mapping is used because constructor options belong to the concrete
evaluator implementation rather than to owl's evaluator protocol.
"""


EvaluatorDeclaration: TypeAlias = (
    EvaluatorType
    | tuple[EvaluatorType, EvaluatorArgs]
)
"""Declarative evaluator construction specification.

The direct form contains an evaluator constructor that can be invoked without
additional arguments:

    BinaryMaskEvaluator

The configured form contains the evaluator constructor and its keyword
arguments:

    (
        BinaryMaskEvaluator,
        {
            "threshold": 0.4,
        },
    )

Evaluator instances are deliberately excluded. Evaluators may hold mutable
dataset-level aggregation snapshot, so every inference session should receive a
newly constructed evaluator.
"""


def resolve_evaluator(
    declaration: EvaluatorDeclaration,
) -> Evaluator:
    """Resolve an evaluator declaration into a fresh evaluator instance.

    A direct constructor declaration is invoked without arguments. A configured
    declaration is expanded into constructor keyword arguments.

    The resolved object is structurally validated against the evaluator protocol
    by checking its required runtime methods. The protocol itself is not marked
    as runtime-checkable, so ``isinstance`` cannot be used for this validation.

    Constructor errors are allowed to propagate unchanged so callers retain the
    original exception type and traceback produced by the evaluator
    implementation.

    Args:
        declaration:
            Evaluator constructor or a tuple containing the constructor and its
            keyword arguments.

    Returns:
        Newly constructed evaluator instance.

    Raises:
        TypeError:
            If the declaration is malformed, its constructor is not callable,
            its constructor arguments are not a mapping, or the constructed
            object does not implement the evaluator protocol.
    """

    if isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "evaluator declaration must contain exactly "
                "(evaluator_type, evaluator_args)"
            )

        evaluator_type, evaluator_args = declaration

        if not callable(evaluator_type):
            raise TypeError(
                "evaluator type must be callable"
            )

        if not isinstance(evaluator_args, Mapping):
            raise TypeError(
                "evaluator constructor arguments must be a mapping"
            )

        evaluator = evaluator_type(
            **dict(evaluator_args),
        )
    else:
        if not callable(declaration):
            raise TypeError(
                "evaluator declaration must be a callable evaluator type or "
                "(evaluator_type, evaluator_args) tuple"
            )

        evaluator = declaration()

    required_methods = (
        "reset",
        "update",
        "compute",
    )

    if not all(
        callable(getattr(evaluator, method, None))
        for method in required_methods
    ):
        raise TypeError(
            "resolved evaluator must implement reset, update, and compute"
        )

    return evaluator


__all__ = [
    "EvaluatorArgs",
    "EvaluatorDeclaration",
    "EvaluatorType",
    "resolve_evaluator",
]