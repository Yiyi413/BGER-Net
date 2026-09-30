"""Top-level invocation declarations.

This package exposes the primary user-facing invocation entry points.

Detailed declarations remain organized by semantic domain in the
``components``, ``data``, ``execution``, and ``process`` subpackages.
"""

from .infer import InferInvocation
from .train import TrainInference, TrainInvocation


__all__ = [
    "InferInvocation",
    "TrainInference",
    "TrainInvocation",
]
