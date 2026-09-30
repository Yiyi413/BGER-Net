"""Shared type aliases for runtime output schemas.

This module defines the primitive value types used by model outputs,
criterion outputs, and parsed runtime outputs. It does not perform
schema validation or output parsing.
"""

from typing import Literal, TypeAlias, Union

from torch import Tensor


PrefixedOutputNamespace: TypeAlias = Literal["visual", "metric"]
"""Supported namespaces for prefixed model output keys.

Only ``visual:*`` and ``metric:*`` use prefixed keys. The ``loss`` and
``eval`` model outputs are fixed keys and are not treated as namespaces.
"""


MetricValue: TypeAlias = Union[int, float, str, bool]
"""Scalar value that can be logged as a metric.

Metric values are intended for logging, monitoring, and event emission.
Tensor values should be converted to Python scalars before being exposed
as metrics.
"""


TensorOutputValue: TypeAlias = Union[Tensor, list[Tensor], tuple[Tensor, ...]]
"""Tensor payload accepted by tensor-oriented runtime consumers.

This type is used by fixed ``loss`` and ``eval`` outputs, and by prefixed
``visual:*`` outputs. Multiple tensors can be represented by a list or tuple.
"""


ModelOutputValue: TypeAlias = Union[TensorOutputValue, MetricValue]
"""Value accepted in a raw model output dictionary.

The accepted value type depends on the output key. ``loss``, ``eval``, and
``visual:*`` require tensor payloads. ``metric:*`` requires scalar metric
values.
"""


CriterionOutputValue: TypeAlias = Union[Tensor, MetricValue]
"""Value accepted in a raw criterion output dictionary.

Criterion outputs use a fixed ``loss`` key for the backward target and
prefixed ``metric:*`` keys for optional scalar metrics.
"""


ParsedTensorOutputs: TypeAlias = dict[str, TensorOutputValue]
"""Named tensor outputs after prefix parsing.

This type is used for outputs that can appear multiple times with names, such
as ``visual:*`` model outputs.
"""


ParsedMetricOutputs: TypeAlias = dict[str, MetricValue]
"""Metric outputs after prefix parsing.

The dictionary key is the metric name after removing the ``metric:`` prefix.
For example, ``metric:score`` becomes ``score``.
"""