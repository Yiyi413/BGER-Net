"""Inference process declaration and resolution.

This module defines the unified declaration accepted by inference workflows and
the resolver that constructs and classifies the declared processor.

An inference process may be either:

- an evaluator that accumulates dataset-level metrics; or
- a visualizer that generates and saves visualization images.

The resolver constructs one fresh processor instance and returns a tagged result
that can be passed directly to ``InferSession``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from ...schemas.processors import Evaluator, Visualizer
from .evaluator import EvaluatorDeclaration
from .visualizer import VisualizerDeclaration


ProcessDeclaration: TypeAlias = (
    EvaluatorDeclaration
    | VisualizerDeclaration
)
"""Declarative inference processor construction specification.

The declaration may describe either:

- an evaluator constructor;
- a configured evaluator constructor;
- a visualizer constructor; or
- a configured visualizer constructor.

The direct form contains only the processor constructor:

    BinaryMaskEvaluator

The configured form additionally contains constructor keyword arguments:

    (
        BinaryMaskEvaluator,
        {
            "threshold": 0.4,
        },
    )
"""


@dataclass(frozen=True, slots=True)
class ResolvedProcess:
    """Resolved and classified inference processor.

    Exactly one of ``evaluator`` and ``visualizer`` is populated.

    The result can be expanded directly when constructing an ``InferSession``:

        process = resolve_process(declaration)

        session = InferSession(
            ...,
            evaluator=process.evaluator,
            visualizer=process.visualizer,
        )
    """

    evaluator: Evaluator | None = None
    visualizer: Visualizer | None = None

    def __post_init__(self) -> None:
        """Require exactly one resolved processor category."""

        if (self.evaluator is None) == (self.visualizer is None):
            raise ValueError(
                "exactly one of evaluator or visualizer must be resolved"
            )


def _construct_process(
    declaration: ProcessDeclaration,
) -> object:
    """Construct the processor described by one declaration."""

    if isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "configured process declaration must contain exactly "
                "(process_type, process_args)"
            )

        process_type, process_args = declaration

        if not callable(process_type):
            raise TypeError(
                "process type must be callable"
            )

        if not isinstance(process_args, Mapping):
            raise TypeError(
                "process constructor arguments must be a mapping"
            )

        constructor = cast(Callable[..., object], process_type)

        return constructor(
            **dict(process_args),
        )

    if not callable(declaration):
        raise TypeError(
            "process declaration must be a callable processor type or "
            "(process_type, process_args) tuple"
        )

    constructor = cast(Callable[..., object], declaration)
    return constructor()


def _is_evaluator(process: object) -> bool:
    """Return whether an object structurally implements Evaluator."""

    return all(
        callable(getattr(process, method, None))
        for method in (
            "reset",
            "update",
            "compute",
        )
    )


def _is_visualizer(process: object) -> bool:
    """Return whether an object structurally implements Visualizer."""

    return all(
        callable(getattr(process, method, None))
        for method in (
            "visualize",
            "save",
        )
    )


def resolve_process(
    declaration: ProcessDeclaration,
) -> ResolvedProcess:
    """Construct and classify one inference processor declaration.

    Runtime classification follows the structural processor protocols:

    - evaluators provide ``reset``, ``update``, and ``compute``;
    - visualizers provide ``visualize`` and ``save``.

    Args:
        declaration:
            Evaluator or visualizer construction declaration.

    Returns:
        Classified processor containing exactly one evaluator or visualizer.

    Raises:
        TypeError:
            If the declaration is malformed, the declared constructor is not
            callable, constructor arguments are not a mapping, or the resolved
            object implements neither processor protocol.
        ValueError:
            If the resolved object implements both processor protocols and is
            therefore ambiguous.
    """

    process = _construct_process(declaration)

    is_evaluator = _is_evaluator(process)
    is_visualizer = _is_visualizer(process)

    if is_evaluator and is_visualizer:
        raise ValueError(
            "resolved process is ambiguous because it implements both "
            "evaluator and visualizer protocols"
        )

    if is_evaluator:
        return ResolvedProcess(
            evaluator=cast(Evaluator, process),
        )

    if is_visualizer:
        return ResolvedProcess(
            visualizer=cast(Visualizer, process),
        )

    raise TypeError(
        "resolved process must implement either the evaluator protocol "
        "(reset, update, compute) or the visualizer protocol "
        "(visualize, save)"
    )


__all__ = [
    "ProcessDeclaration",
    "ResolvedProcess",
    "resolve_process",
]