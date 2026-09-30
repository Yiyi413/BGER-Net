"""Training execution configuration.

This module defines the invocation-level configuration that controls how a
training run is executed.

``TrainExecution`` contains only execution concerns:

- the total number of epochs in the training plan;
- the device used by the training session;
- the parent directory used to create the invocation workspace;
- the checkpoint-saving policy applied during execution.

Component construction, dataset declaration, checkpoint loading, and evaluation
configuration belong to separate invocation domains.

The starting epoch is intentionally not exposed as a user-configurable field.
A new training run starts from epoch zero. When a complete checkpoint is
restored, the invocation pipeline determines the effective starting epoch from
the progress stored in that checkpoint.

Checkpoint saving is always represented by a concrete ``CheckpointSave``
configuration. When the user does not provide one, the default configuration
disables automatic epoch-level saving.

The workspace path is a parent directory declaration. It is not the concrete
workspace directory for the run. A later workspace layer creates a timestamped
directory such as ``workspace-YYYYMMDDHHMMSSmmm`` inside that parent directory.
When no workspace parent is supplied, owl uses its default parent directory.
"""

from dataclasses import dataclass, field
from pathlib import Path

from torch import device as TorchDevice

from ..data.types import PathLike
from .checkpoint import CheckpointSave


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainExecution:
    """Configuration describing training execution behavior.

    This object is declarative. It does not select hardware, move components,
    create runtime sessions, create workspace directories, execute epochs, or
    write checkpoint files.

    A future client consumes this configuration after resolving component and
    data declarations.

    Attributes:
        total_epochs:
            Total number of epochs in the complete training plan.

            This value represents the final epoch count rather than the number
            of additional epochs to execute after checkpoint restoration.

            For example, when ``total_epochs`` is ``100`` and a checkpoint
            records that epoch ``39`` has completed, the restored run continues
            from epoch ``40`` and stops before epoch ``100``.

        device:
            Device on which the training model and runtime tensors should
            reside.

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

        checkpoint:
            Checkpoint-saving configuration used during training execution.

            The default value is a new ``CheckpointSave`` instance with
            ``autosave=False``. The runtime performs no automatic epoch-level
            saving unless explicitly requested.
    """

    total_epochs: int
    device: str | TorchDevice
    workspace: PathLike | None = None
    checkpoint: CheckpointSave = field(default_factory=CheckpointSave)

    def __post_init__(self) -> None:
        """Normalize device and optional workspace parent declarations.

        This method performs no hardware, backend, or file-system availability
        checks. Values such as ``"cuda:0"`` are accepted as declarations even
        when CUDA is not available in the current process.

        ``object.__setattr__`` is required because the dataclass is frozen.
        """

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
    "TrainExecution",
]