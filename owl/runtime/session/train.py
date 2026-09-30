"""Training runtime session definitions.

This module defines the session used by training runtimes. A training session
receives components that have already been instantiated by the client and
provides operations for controlling those components during runtime execution.

The runtime owns the training loop and determines when each session operation
is invoked. Component construction and training control flow are not handled
by this module. Component call conventions are defined by
``owl.schemas.calls`` and executed by the session.
"""

from collections.abc import Sized

from torch import Tensor
from torch import device as TorchDevice
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ...data.dataset import DatasetBatch
from ...schemas.checkpoint.v1 import OwlCheckpointV1
from ...schemas.calls.context import TrainCallContext
from ...schemas.outputs.criterion import CriterionOutput
from ...schemas.outputs.model import ModelOutput
from ...schemas.outputs.types import TensorOutputValue
from .base import BaseSession


class TrainSession(BaseSession):
    """Session containing instantiated components required for training.

    The client must construct the model, criterion, optimizer, scheduler, and
    dataloader before creating the session. It also determines the starting
    epoch after initializing or restoring the training snapshot.

    The runtime uses this session to invoke components, update component
    states, and maintain epoch progress.

    Args:
        model:
            Instantiated model used for training.
        device:
            Device on which the model and training tensors reside.
        criterion:
            Instantiated criterion used to compute the training loss.
        optimizer:
            Instantiated optimizer used to update model parameters.
        scheduler:
            Instantiated learning-rate scheduler associated with the optimizer.
            Fixed learning rates should use a scheduler that preserves the
            current learning rate.
        train_dataloader:
            Dataloader that provides training batches.
        total_epochs:
            Total number of epochs planned for the training run.
        start_epoch:
            Zero-based epoch index from which this training invocation begins.

    Raises:
        TypeError:
            If ``train_dataloader`` does not provide a length.
        ValueError:
            If ``total_epochs`` is not positive, ``start_epoch`` is outside the
            configured training range, or the dataloader is empty.
    """

    def __init__(
        self,
        model: Module,
        device: TorchDevice,
        criterion: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_dataloader: DataLoader,
        total_epochs: int,
        start_epoch: int = 0,
    ) -> None:
        super().__init__(model=model, device=device)

        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")

        if not 0 <= start_epoch <= total_epochs:
            raise ValueError(
                "start_epoch must be in "
                f"[0, {total_epochs}], got {start_epoch}"
            )

        if not isinstance(train_dataloader, Sized):
            raise TypeError("train_dataloader must provide a length")

        if len(train_dataloader) == 0:
            raise ValueError("train_dataloader must contain at least one batch")

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader

        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.current_epoch = -1

    @property
    def steps_per_epoch(self) -> int:
        """Return the number of training batches in one epoch."""
        return len(self.train_dataloader)

    @property
    def total_steps(self) -> int:
        """Return the total number of training batches in the training plan."""
        return self.total_epochs * self.steps_per_epoch

    def get_learning_rates(self) -> list[float]:
        """Return the current learning rate of every optimizer parameter group.

        Returns:
            Learning rates in the same order as the optimizer parameter groups.
        """
        return [
            float(param_group["lr"])
            for param_group in self.optimizer.param_groups
        ]

    def create_checkpoint(self) -> OwlCheckpointV1:
        """Create an Owl checkpoint v1 for the current epoch state.

        The runtime should invoke this method only after the current epoch has
        completed successfully. The checkpoint contains the current model,
        optimizer, and scheduler state dictionaries together with the zero-based
        current epoch.

        This method only constructs the checkpoint dictionary. It does not
        generate a destination path or write the checkpoint to disk.

        Returns:
            Complete Owl checkpoint v1 representing the current epoch state.

        Raises:
            RuntimeError:
                If the runtime has not set the current epoch.
        """
        if self.current_epoch < 0:
            raise RuntimeError(
                "current epoch must be set before creating a checkpoint"
            )

        return {
            "format_version": 1,
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def set_model_train_mode(self) -> None:
        """Set the model to training mode.

        This operation affects modules whose behavior differs between training
        and evaluation, such as dropout and batch normalization layers.
        """
        self.model.train()

    def set_current_epoch(self, epoch: int) -> None:
        """Update the epoch currently executed by the runtime.

        Args:
            epoch:
                Zero-based epoch index currently executed by the runtime.

        Raises:
            ValueError:
                If ``epoch`` is outside the training range represented by this
                session.
        """
        if not self.start_epoch <= epoch < self.total_epochs:
            raise ValueError(
                "epoch must be in "
                f"[{self.start_epoch}, {self.total_epochs}), got {epoch}"
            )

        self.current_epoch = epoch

    def _create_call_context(self, current_batch: int) -> TrainCallContext:
        """Create the call context for the current training batch.

        Args:
            current_batch:
                Zero-based index of the batch within the current epoch.

        Returns:
            Immutable context passed to the model and criterion.

        Raises:
            RuntimeError:
                If the runtime has not set the current epoch.
            ValueError:
                If ``current_batch`` is outside the dataloader range.
        """
        if self.current_epoch < 0:
            raise RuntimeError(
                "current epoch must be set before invoking training components"
            )

        if not 0 <= current_batch < self.steps_per_epoch:
            raise ValueError(
                "current_batch must be in "
                f"[0, {self.steps_per_epoch}), got {current_batch}"
            )

        return TrainCallContext(
            current_epoch=self.current_epoch,
            current_batch=current_batch,
            total_epochs=self.total_epochs,
            total_batches=self.steps_per_epoch,
        )

    def forward_model(
        self,
        batch: DatasetBatch,
        current_batch: int,
    ) -> ModelOutput:
        """Invoke the model for the current training batch.

        The batch follows the standard owl dataset batch schema. The model may
        consume the fields required by its task and ignore the remaining
        fields.

        Args:
            batch:
                Batch produced by an owl training dataloader.
            current_batch:
                Zero-based index of the batch within the current epoch.

        Returns:
            Raw model output consumed by the model output parser.
        """
        context = self._create_call_context(current_batch)

        return self.model(batch, context)

    def forward_criterion(
        self,
        loss_output: TensorOutputValue,
        batch: DatasetBatch,
        current_batch: int,
    ) -> CriterionOutput:
        """Invoke the criterion for the current training batch.

        Args:
            loss_output:
                Tensor payload parsed from the model's fixed ``loss`` output
                key.
            batch:
                Batch produced by an owl training dataloader.
            current_batch:
                Zero-based index of the batch within the current epoch.

        Returns:
            Raw criterion output consumed by the criterion output parser.
        """
        context = self._create_call_context(current_batch)

        return self.criterion(loss_output, batch, context)

    def backward_loss(self, loss: Tensor) -> None:
        """Compute gradients from the parsed training loss."""
        loss.backward()

    def clear_optimizer_gradients(self) -> None:
        """Clear gradients maintained by the optimizer."""
        self.optimizer.zero_grad()

    def update_model_parameters(self) -> None:
        """Update model parameters using the optimizer."""
        self.optimizer.step()

    def update_learning_rate(self) -> None:
        """Advance the scheduler after one optimizer update."""
        self.scheduler.step()
