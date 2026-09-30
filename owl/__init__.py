"""Public Owl package API.

Runtime-heavy public objects are loaded lazily so lightweight operations, such
as invoking the CLI version command, do not import orchestration or machine
learning dependencies.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .__about__ import __version__


if TYPE_CHECKING:
    from .orchestration.invoke import invoke


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "invoke": (".orchestration.invoke", "invoke"),
}

__all__ = [
    "__version__",
    "invoke",
]


def __getattr__(name: str) -> Any:
    """Load a public package attribute on first access.

    Loaded attributes are cached in the module namespace so subsequent access
    uses the resolved object directly without repeating the import.
    """
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = target
    module = import_module(module_name, package=__name__)
    value = getattr(module, attribute_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eagerly and lazily available package attributes."""
    return sorted(set(globals()) | set(__all__))