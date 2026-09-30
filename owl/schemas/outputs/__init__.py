"""Output schemas used by owl runtimes.

This package defines raw output schemas returned by user components and parsed
output schemas consumed internally by runtimes.
"""

from .criterion import CriterionOutput
from .model import ModelOutput
from .parsed import ParsedCriterionOutput, ParsedModelOutput
from .types import (
    CriterionOutputValue,
    MetricValue,
    ModelOutputValue,
    ParsedMetricOutputs,
    ParsedTensorOutputs,
    PrefixedOutputNamespace,
    TensorOutputValue,
)

__all__ = [
    "CriterionOutput",
    "CriterionOutputValue",
    "MetricValue",
    "ModelOutput",
    "ModelOutputValue",
    "ParsedCriterionOutput",
    "ParsedMetricOutputs",
    "ParsedModelOutput",
    "ParsedTensorOutputs",
    "PrefixedOutputNamespace",
    "TensorOutputValue",
]