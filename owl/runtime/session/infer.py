"""Inference runtime session definitions.

This module defines the session used by inference runtimes. An inference
session receives components that have already been instantiated by the client
and provides operations for executing model inference across named
dataloaders.

The runtime owns the inference loop and determines when each session operation
is invoked. Component construction and inference control flow are not handled
by this module. Model call conventions are defined by ``owl.schemas.calls``
and executed by the session.
"""

from torch import device as TorchDevice
from torch.nn import Module
from torch.utils.data import DataLoader

from ...data.dataset import DatasetBatch
from ...schemas.outputs.model import ModelOutput
from ...schemas.processors import Evaluator, Visualizer
from .base import BaseSession


class InferSession(BaseSession):
    """Session containing instantiated components required for inference.

    The client must construct the model, dataloaders, and inference processor
    before creating the session. Multiple dataloaders are stored by name so
    the runtime can execute inference across multiple IMDL datasets within one
    invocation.

    Exactly one inference processor must be supplied:

    - an evaluator for aggregating dataset-level metrics; or
    - a visualizer for generating and saving inference visualizations.

    Args:
        model:
            Instantiated model used for inference.
        device:
            Device on which the model and inference tensors reside.
        dataloaders:
            Named dataloaders used by the inference runtime. Each key
            identifies the corresponding dataset.
        evaluator:
            Optional evaluator used to collect batch outputs and compute
            dataset-level metrics.
        visualizer:
            Optional visualizer used to generate and save visualization
            images.

    Raises:
        ValueError:
            If ``dataloaders`` is empty, contains an empty dataset name, or
            does not receive exactly one of ``evaluator`` and ``visualizer``.
    """

    def __init__(
        self,
        model: Module,
        device: TorchDevice,
        dataloaders: dict[str, DataLoader],
        evaluator: Evaluator | None = None,
        visualizer: Visualizer | None = None,
    ) -> None:
        super().__init__(model=model, device=device)

        if not dataloaders:
            raise ValueError("dataloaders must contain at least one dataset")

        if any(not name.strip() for name in dataloaders):
            raise ValueError("dataloader names must not be empty")

        if (evaluator is None) == (visualizer is None):
            raise ValueError(
                "exactly one of evaluator or visualizer must be provided"
            )

        self.dataloaders = dataloaders
        self.evaluator = evaluator
        self.visualizer = visualizer

    def forward_model(self, batch: DatasetBatch) -> ModelOutput:
        """Invoke the model for the current inference batch.

        The batch follows the standard owl dataset batch schema. The model may
        consume the fields required by its task and ignore the remaining
        fields.

        Args:
            batch:
                Batch produced by an owl inference dataloader.

        Returns:
            Raw model output consumed by the model output parser.
        """
        return self.model(batch)
