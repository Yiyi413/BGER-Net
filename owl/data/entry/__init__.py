"""Dataset entry definitions and built-in entry sources."""

from .entry import EntrySource, SampleEntry
from .owl_v1 import OwlV1EntrySource

__all__ = [
    "EntrySource",
    "OwlV1EntrySource",
    "SampleEntry",
]

