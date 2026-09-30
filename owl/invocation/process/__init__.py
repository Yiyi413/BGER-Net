"""Inference processor declarations and resolvers."""

from .evaluator import (
    EvaluatorArgs,
    EvaluatorDeclaration,
    EvaluatorType,
    resolve_evaluator,
)
from .process import (
    ProcessDeclaration,
    ResolvedProcess,
    resolve_process,
)
from .visualizer import (
    VisualizerArgs,
    VisualizerDeclaration,
    VisualizerType,
)


__all__ = [
    "EvaluatorArgs",
    "EvaluatorDeclaration",
    "EvaluatorType",
    "ProcessDeclaration",
    "ResolvedProcess",
    "VisualizerArgs",
    "VisualizerDeclaration",
    "VisualizerType",
    "resolve_evaluator",
    "resolve_process",
]