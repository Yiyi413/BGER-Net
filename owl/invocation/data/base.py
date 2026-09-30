"""Shared data invocation configuration and construction utilities.

This module defines the configuration fields shared by training and inference
data declarations together with the internal construction utilities used by
their resolvers.

The shared data construction pipeline is:

    DataDeclaration
    -> EntrySource
    -> OwlDataset

Training and inference resolvers extend this pipeline differently:

- training datasets are combined and wrapped in one ``DataLoader``;
- inference datasets remain independent and produce a named mapping of
  ``DataLoader`` objects.

Keeping the single-dataset construction logic in this module ensures that
training and inference declarations interpret roots, entry-source overrides,
augmentation pipelines, and sample hooks consistently.
"""

from dataclasses import dataclass, field
from pathlib import Path

import albumentations as A

from ...data.augment import SampleHook
from ...data.entry.owl_v1 import OwlV1EntrySource
from ...data.dataset import OwlDataset
from ...data.entry import EntrySource
from .types import (
    DataDeclaration,
    EntrySourceConstructor,
    LoaderOptions,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _DataConfig:
    """Base configuration shared by training and inference data declarations.

    This class is an internal implementation detail. Users should construct
    ``TrainData`` or ``InferData`` rather than instantiate ``_DataConfig``
    directly.

    The class stores the options shared by every dataset represented by one
    data invocation. Concrete resolvers consume these options when constructing
    entry sources, datasets, and dataloaders.

    Attributes:
        default_entry_source:
            Default entry-source class used to interpret compact dataset
            declarations.

            A compact declaration contains only a dataset root:

                "datasets/casia_v1"

            The resolver constructs the default entry source with that root:

                source = config.entry(Path(root))

            A declaration may instead contain a local entry-source class:

                (
                    "datasets/custom",
                    CustomEntrySource,
                )

            In that form, the local class takes precedence over this default.

        augment:
            Optional Albumentations transform passed to every ``OwlDataset``
            constructed from this configuration.

            The transform operates on NumPy arrays inside ``OwlDataset``. It
            may perform spatial transformations and normalization, but tensor
            conversion remains the responsibility of ``OwlDataset``.

            Training and inference use separate configuration objects, so they
            can naturally define different augmentation pipelines.

        hook:
            Optional sample-level hook passed to every ``OwlDataset`` built
            from this configuration.

            The hook runs after the Albumentations transform and before tensor
            conversion. It may update labels, generate edge supervision, or
            apply other operations that require the complete transformed
            sample.

        loader:
            Keyword arguments used by the concrete resolver when constructing
            ``torch.utils.data.DataLoader`` objects.

            Typical options include ``batch_size``, ``shuffle``,
            ``num_workers``, ``pin_memory``, ``drop_last``, ``sampler``, and
            ``collate_fn``.

            Training options apply to the final combined training dataset.
            Inference options are copied and applied independently to every
            named inference dataset.
    """

    default_entry_source: EntrySourceConstructor = OwlV1EntrySource
    augment: A.Compose | None = None
    hook: SampleHook | None = None
    loader: LoaderOptions = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach the configuration from its caller-owned loader mapping.

        ``frozen=True`` prevents reassignment of ``loader`` after construction,
        but it does not copy a mutable mapping supplied by the caller.

        The mapping is therefore shallow-copied so later additions, removals,
        or replacements in the original mapping do not modify this
        configuration. Mapping values such as samplers, generators, worker
        initialization functions, and collate functions preserve their
        identities.
        """

        object.__setattr__(
            self,
            "loader",
            dict(self.loader),
        )


def _build_entry_source(
    declaration: DataDeclaration,
    *,
    default_entry: EntrySourceConstructor,
) -> EntrySource:
    """Construct one entry source from a data declaration.

    Compact declarations contain only a dataset root and use
    ``default_entry``:

        "datasets/casia_v1"

    Override declarations contain a dataset root and a local entry-source
    constructor:

        (
            "datasets/custom",
            CustomEntrySource,
        )

    Dataset roots are normalized to ``Path`` objects immediately before entry
    source construction. Entry-source constructor exceptions are allowed to
    propagate unchanged.

    Args:
        declaration:
            Compact root declaration or
            ``(root, entry_source_constructor)`` override.

        default_entry:
            Entry-source constructor used by compact declarations.

    Returns:
        Constructed entry source.

    Raises:
        TypeError:
            If the declaration shape, dataset root, or selected entry-source
            constructor is invalid.
    """
    if isinstance(declaration, (str, Path)):
        root = declaration
        constructor = default_entry

    elif isinstance(declaration, tuple):
        if len(declaration) != 2:
            raise TypeError(
                "data declaration tuple must contain exactly "
                "(root, entry_source_constructor)"
            )

        root, constructor = declaration

        if not isinstance(root, (str, Path)):
            raise TypeError(
                "data declaration root must be a string or Path"
            )

    else:
        raise TypeError(
            "data declaration must be a string, Path, or "
            "(root, entry_source_constructor) tuple"
        )

    if not callable(constructor):
        raise TypeError(
            "entry source constructor must be callable"
        )

    return constructor(Path(root))


def _build_dataset(
    declaration: DataDeclaration,
    *,
    default_entry: EntrySourceConstructor,
    augment: A.Compose | None,
    hook: SampleHook | None,
) -> OwlDataset:
    """Construct one ``OwlDataset`` from a data declaration.

    The declaration is first resolved into an entry source. The resulting
    source is then combined with the shared augmentation pipeline and sample
    hook owned by the data configuration.

    ``OwlDataset`` performs its own source-level validation, including empty
    source detection and edge-availability consistency checks. Those errors are
    allowed to propagate unchanged.

    Args:
        declaration:
            Dataset root declaration with an optional local entry-source class.
        default_entry:
            Entry-source class used when the declaration contains only a root.
        augment:
            Optional Albumentations transform passed to ``OwlDataset``.
        hook:
            Optional post-transform sample hook passed to ``OwlDataset``.

    Returns:
        Constructed Owl dataset.
    """
    source = _build_entry_source(
        declaration,
        default_entry=default_entry,
    )

    return OwlDataset(
        source=source,
        transform=augment,
        hook=hook,
    )


__all__ = [
    "_DataConfig",
    "_build_dataset",
    "_build_entry_source",
]
