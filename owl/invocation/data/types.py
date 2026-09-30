"""Type definitions for data invocation configurations.

This module defines the declarative types used by owl data invocation objects.

A data invocation does not load resources or construct datasets directly.
Instead, it records enough information for the invocation resolver to build the
data pipeline in three distinct stages:

    dataset root
    -> EntrySource
    -> OwlDataset
    -> DataLoader

Keeping these stages separate preserves the responsibilities of the existing
data layer:

- ``EntrySource`` describes stable source-level sample facts.
- ``OwlDataset`` loads resources and applies transforms and sample hooks.
- ``DataLoader`` controls batching and iteration behavior.

The aliases in this module describe configuration values only. They do not
perform path normalization, file-system validation, entry-source construction,
or dataloader argument validation.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

from ...data.entry import EntrySource


PathLike: TypeAlias = str | Path
"""Path representation accepted by data invocation declarations.

String paths are preserved while the invocation configuration is being
declared. The data resolver converts each value to ``Path`` immediately before
constructing the configured entry source.

The invocation layer intentionally does not normalize paths eagerly because
configuration construction should not depend on the current working directory
or access the file system.
"""


EntrySourceConstructor: TypeAlias = Callable[[Path], EntrySource]
"""Callable that constructs an entry source from a dataset root.

The constructor receives one normalized ``Path`` and returns an object
satisfying the ``EntrySource`` structural protocol:

    entry_source = constructor(root)

An entry-source class commonly satisfies this alias:

    class CustomEntrySource:
        def __init__(self, root: Path) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, index: int) -> SampleEntry:
            ...

The class object itself acts as the constructor:

    constructor = CustomEntrySource
    entry_source = constructor(
        Path("datasets/custom"),
    )

A compatible function or callable object may also be used:

    def create_entry_source(root: Path) -> EntrySource:
        return CustomEntrySource(root)

The alias describes the required call signature rather than a concrete class
or inheritance relationship. Entry-source implementations therefore do not
need to inherit from ``EntrySource`` explicitly.
"""


DataDeclaration: TypeAlias = (
    PathLike
    | tuple[PathLike, EntrySourceConstructor]
)
"""Declaration of one logical dataset source.

A declaration supports two forms.

The compact form contains only a dataset root:

    "datasets/casia_v1"

The owning data configuration supplies the default entry-source constructor for
this form.

The override form contains a dataset root and a local entry-source constructor:

    (
        "datasets/custom",
        CustomEntrySource,
    )

The local constructor takes precedence over the default constructor configured
at the data-object level.

During resolution, the selected constructor receives the normalized dataset
root:

    entry_source = constructor(
        Path(root),
    )

Entry-source constructors are expected to accept one normalized dataset root as
their only resolver-injected argument in v0.0.2.
"""


LoaderOptions: TypeAlias = Mapping[str, Any]
"""Keyword arguments forwarded to ``torch.utils.data.DataLoader``.

The invocation layer stores dataloader options as a generic mapping instead of
duplicating the complete PyTorch ``DataLoader`` constructor in an owl-specific
configuration class.

Configuration objects copy this mapping during initialization so later
top-level mutations to the caller-owned mapping do not change the stored
configuration. Data resolvers create their own ``dict`` before forwarding the
options to ``DataLoader``.

Compatibility between individual options, such as ``shuffle`` and ``sampler``,
remains the responsibility of PyTorch's ``DataLoader`` implementation.
"""


__all__ = [
    "DataDeclaration",
    "EntrySourceConstructor",
    "LoaderOptions",
    "PathLike",
]