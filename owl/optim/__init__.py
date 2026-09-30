"""Built-in optimization presets."""

from .optimizer import adamw
from .scheduler import constant, poly

__all__ = [
    "adamw",
    "constant",
    "poly",
]