"""Standalone inference execution configuration.

This module defines invocation-level configuration for standalone inference
execution.

``InferExecution`` contains only execution concerns:

- the device used by the inference session;
- the parent directory used to create the invocation workspace.

Component construction, dataset declaration, checkpoint loading, process
resolution, evaluation output, and visualization output belong to separate
invocation or workspace domains.

The workspace path is a parent directory declaration. It is not the concrete
workspace directory for the run. A later workspace layer creates a timestamped
directory such as ``workspace-YYYYMMDDHHMMSSmmm`` inside that parent directory.
When no workspace parent is supplied, owl uses its default parent directory.
"""

from dataclasses import dataclass
from pathlib import Path

from torch import device as TorchDevice

from ..data.types import PathLike


@dataclass(frozen=True, slots=True, kw_only=True)
class InferExecution:
    """Configuration describing inference execution behavior.

    Attributes:
        device:
            Device on which the model and inference tensors reside.

            The value may be a PyTorch device object or any string accepted by
            ``torch.device``, such as ``"cpu"``, ``"cuda"``, or ``"cuda:0"``.

            The value is normalized into a ``torch.device`` during invocation
            construction. Device availability is not checked at this layer.

        workspace:
            Optional parent directory used to create the invocation workspace.

            This value is not the final workspace directory. When provided, owl
            creates a timestamped child directory such as
            ``workspace-YYYYMMDDHHMMSSmmm`` under this parent. When ``None``,
            owl uses its default workspace parent directory.

            The directory is normalized into a ``Path`` when provided. No file
            system operations are performed at this layer.
    """

    device: str | TorchDevice
    workspace: PathLike | None = None

    def __post_init__(self) -> None:
        """Normalize device and optional workspace parent declarations."""

        object.__setattr__(
            self,
            "device",
            TorchDevice(self.device),
        )

        if self.workspace is not None:
            object.__setattr__(
                self,
                "workspace",
                Path(self.workspace),
            )


__all__ = [
    "InferExecution",
]