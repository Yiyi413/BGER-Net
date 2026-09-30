"""Visualizer process declaration types.

This module defines the declarative forms accepted by owl invocations for
inference visualizer construction.

A visualizer converts model-provided visualization tensors into final image
tensors and saves those images to destinations managed by the inference
runtime. Its runtime protocol is defined by
``owl.schemas.processors.Visualizer``.

Visualizer implementations are declared as classes rather than instantiated
objects. A future client constructs a fresh visualizer instance before creating
an inference session.

A visualizer declaration may contain either:

- a visualizer class;
- a tuple containing the visualizer class and its constructor keyword arguments.

A visualizer without additional constructor arguments may be supplied directly:

    visualizer=BinaryMaskVisualizer

A visualizer that requires configuration may be paired with a mapping:

    visualizer=(
        BinaryMaskVisualizer,
        {
            "threshold": None,
        },
    )

Custom visualizers follow the same declaration format:

    visualizer=(
        OverlayVisualizer,
        {
            "alpha": 0.5,
            "include_target": True,
        },
    )

The invocation layer does not instantiate the visualizer, inspect its
constructor, generate output paths, or save visualization images. Those
responsibilities belong to the client and inference runtime layers.
"""

from typing import Any, Mapping, TypeAlias

from ...schemas.processors import Visualizer


VisualizerType: TypeAlias = type[Visualizer]
"""Class whose instances implement the owl visualizer protocol.

The declaration stores the visualizer class itself rather than an already
constructed visualizer instance.

For example:

    visualizer_type = BinaryMaskVisualizer
    visualizer = visualizer_type()

Using classes allows the client to construct an independent visualizer for each
resolved inference configuration.
"""


VisualizerArgs: TypeAlias = Mapping[str, Any]
"""Keyword arguments supplied to a visualizer constructor.

The client resolves a configured visualizer declaration by expanding the
mapping:

    visualizer = visualizer_type(
        **dict(visualizer_args),
    )

A generic mapping is used because constructor options belong to the concrete
visualizer implementation rather than to owl's visualizer protocol.
"""


VisualizerDeclaration: TypeAlias = (
    VisualizerType
    | tuple[VisualizerType, VisualizerArgs]
)
"""Declarative visualizer construction specification.

The direct form contains a visualizer class that can be instantiated without
additional arguments:

    BinaryMaskVisualizer

The configured form contains the visualizer class and its constructor keyword
arguments:

    (
        BinaryMaskVisualizer,
        {
            "threshold": None,
        },
    )

The client resolves the direct form as follows:

    visualizer = visualizer_type()

The configured form is resolved as follows:

    visualizer = visualizer_type(
        **dict(visualizer_args),
    )

The declaration deliberately excludes visualizer instances. Constructing
visualizers in the client keeps invocation objects declarative and ensures that
separate inference sessions do not unintentionally share processor snapshot.
"""


__all__ = [
    "VisualizerArgs",
    "VisualizerDeclaration",
    "VisualizerType",
]
