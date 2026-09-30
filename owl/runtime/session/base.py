"""Base runtime session definitions.

This module defines the shared session behavior used by training and inference
runtimes. A session receives components that have already been instantiated by
the client and exposes operations for managing those components during runtime
execution.

Component construction, checkpoint restoration, and runtime control flow are
not handled by this module.
"""

from torch import device as TorchDevice
from torch.nn import Module

from ...data.dataset import DatasetBatch


class BaseSession:
    """Base session shared by training and inference runtimes.

    The client is responsible for constructing the model, moving it to the
    target device, and then creating the session. The session stores the
    prepared model and exposes runtime operations that manage model snapshot and
    runtime tensors.

    Args:
        model:
            Instantiated model used by the runtime.
        device:
            Device on which the model and runtime tensors reside.
    """

    def __init__(self, model: Module, device: TorchDevice) -> None:
        self.model = model
        self.device = device

    def move_batch_to_device(
        self,
        batch: DatasetBatch,
    ) -> DatasetBatch:
        """Move every tensor in a dataset batch to the session device.

        Non-tensor metadata remains on the host and is copied into the returned
        batch unchanged.

        Args:
            batch:
                Batch produced by an owl dataloader.

        Returns:
            Batch whose tensors reside on the session device and whose metadata
            is preserved unchanged.
        """
        return DatasetBatch(
            tp_name=batch["tp_name"],
            tp=batch["tp"].to(self.device),
            gt=batch["gt"].to(self.device),
            label=batch["label"].to(self.device),
            edge=batch["edge"].to(self.device),
        )

    def set_model_eval_mode(self) -> None:
        """Set the model to evaluation mode.

        This operation delegates to ``torch.nn.Module.eval`` and affects
        modules whose behavior differs between training and evaluation, such
        as dropout and batch normalization layers.
        """
        self.model.eval()
