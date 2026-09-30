"""Inference data invocation configuration and resolver.

This module defines the declarative configuration used to construct inference
data pipelines and the resolver that materializes those declarations.

Inference sources remain independent because each source represents one named
evaluation or visualization dataset:

    named data declarations
    -> one OwlDataset per name
    -> one DataLoader per name
    -> dict[str, DataLoader]

Dataset names are preserved throughout resolution and can be injected directly
into ``InferSession``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from torch.utils.data import DataLoader

from .base import _DataConfig, _build_dataset
from .types import DataDeclaration


@dataclass(frozen=True, slots=True, kw_only=True)
class InferData(_DataConfig):
    """Configuration describing named inference data pipelines.

    Each entry in ``sources`` represents one logical inference dataset:

        {
            "casia_v1": "datasets/casia_v1",
            "columbia": "datasets/columbia",
            "custom": (
                "datasets/custom_test",
                CustomEntrySource,
            ),
        }

    Compact declarations use the inherited ``entry`` field as their default
    entry-source constructor. Declarations containing a local entry-source
    constructor use that constructor instead.

    Unlike training datasets, inference datasets are never combined. Resolution
    creates one independent dataloader for every declared name.

    Attributes:
        sources:
            Mapping from inference dataset names to data declarations.

            The supplied mapping is copied during initialization so later
            mutations to the caller-owned mapping do not affect the stored
            configuration. Dictionary insertion order is preserved.
    """

    sources: Mapping[str, DataDeclaration] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy shared options and detach the source mapping."""

        _DataConfig.__post_init__(self)

        object.__setattr__(
            self,
            "sources",
            dict(self.sources),
        )


def resolve_infer_data(config: InferData) -> dict[str, DataLoader]:
    """Resolve an inference data configuration into named dataloaders.

    Every declared source is independently resolved into an ``OwlDataset`` and
    wrapped in its own ``DataLoader``. Dataset names and declaration order are
    preserved in the returned dictionary.

    Args:
        config:
            Declarative inference data configuration.

    Returns:
        Mapping from inference dataset names to resolved dataloaders.

    Raises:
        ValueError:
            If no inference sources are declared or a dataset name is blank.
        TypeError:
            If a dataset name is not a string.
    """

    if not config.sources:
        raise ValueError(
            "inference data must declare at least one dataset source"
        )

    dataloaders: dict[str, DataLoader] = {}

    for name, declaration in config.sources.items():
        if not isinstance(name, str):
            raise TypeError(
                "inference dataset names must be strings"
            )

        if not name.strip():
            raise ValueError(
                "inference dataset names must not be blank"
            )

        dataset = _build_dataset(
            declaration,
            default_entry=config.default_entry_source,
            augment=config.augment,
            hook=config.hook,
        )

        dataloaders[name] = DataLoader(
            dataset,
            **dict(config.loader),
        )

    return dataloaders


__all__ = [
    "InferData",
    "resolve_infer_data",
]