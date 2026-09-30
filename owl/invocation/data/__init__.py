"""Public data invocation API.

This package exposes the declarative objects, resolver functions, and type
aliases used to describe and construct training and inference data pipelines.

The public configuration classes are:

- ``TrainData`` for an ordered collection of training dataset declarations;
- ``InferData`` for a named mapping of inference dataset declarations.

The public resolver functions are:

- ``resolve_train_data`` for constructing one training ``DataLoader``;
- ``resolve_infer_data`` for constructing named inference dataloaders.

The public type aliases describe the values accepted by the configuration
objects. They are exported primarily for user-defined annotations and extension
code.

Internal implementation details, including ``_DataConfig`` and the shared
dataset-construction helpers, are intentionally not exported.
"""

from .infer import InferData, resolve_infer_data
from .train import TrainData, resolve_train_data
from .types import (
    DataDeclaration,
    EntrySourceConstructor,
    LoaderOptions,
    PathLike,
)


__all__ = [
    "DataDeclaration",
    "EntrySourceConstructor",
    "InferData",
    "LoaderOptions",
    "PathLike",
    "TrainData",
    "resolve_infer_data",
    "resolve_train_data",
]