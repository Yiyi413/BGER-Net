"""Training runtime implementations.

This module defines the runtime responsible for executing a training session.
The runtime controls epoch and batch iteration, moves each batch to the session
device, delegates component operations to ``TrainSession``, parses raw
component outputs, persists training artifacts through the active workspace,
applies checkpoint-saving policy, and optionally invokes an ``InferRuntime``
after each training epoch.

The training and inference sessions must reference the same model instance.
This allows validation to observe the parameters updated by the training
optimizer without copying or synchronizing model state.

Example:
    Create runtimes and execute training with validation:

        infer_runtime = InferRuntime()
        train_runtime = TrainRuntime(
            infer_runtime=infer_runtime,
            workspace=workspace,
            checkpoint_save=checkpoint_save,
        )

        train_runtime.run(
            train_session=train_session,
            infer_session=infer_session,
        )

    Execute training without validation:

        train_runtime.run(
            train_session=train_session,
        )
"""

from typing import cast

import torch

from ..data.dataset import DatasetBatch
from ..invocation.execution.checkpoint import CheckpointSave
from ..workspace.workspace import Workspace
from .infer import InferRuntime
from .parser import parse_criterion_output, parse_model_output
from .session.infer import InferSession
from .session.train import TrainSession


class TrainRuntime:
    """Runtime responsible for executing model training.

    ``TrainRuntime`` owns the training control flow but does not construct or
    directly manage training components. It moves each batch to the session
    device before component invocation. Model, criterion, optimizer, scheduler,
    and dataloader operations are delegated to ``TrainSession``.

    An ``InferRuntime`` is injected so the same inference loop can be used for
    both standalone inference and validation during training. The active
    workspace receives lifecycle, snapshot, history, checkpoint, and validation
    artifacts produced during execution.

    Args:
        infer_runtime:
            Runtime used to execute the optional inference session after each
            training epoch.

        workspace:
            Active workspace owned by the invocation orchestration entry point.

        checkpoint_save:
            Checkpoint-saving policy applied after completed training epochs.
    """

    def __init__(
        self,
        *,
        infer_runtime: InferRuntime,
        workspace: Workspace,
        checkpoint_save: CheckpointSave,
    ) -> None:
        self.infer_runtime = infer_runtime
        self.workspace = workspace
        self.checkpoint_save = checkpoint_save

    def run(
        self,
        train_session: TrainSession,
        infer_session: InferSession | None = None,
    ) -> None:
        """Execute the epochs represented by the training session.

        Training begins at ``train_session.start_epoch`` and continues until
        ``train_session.total_epochs`` is reached.

        Each epoch performs the following operations:

        1. Publish the training lifecycle stage.
        2. Update the current epoch maintained by the session.
        3. Place the model in training mode.
        4. Execute and persist every training batch.
        5. Execute and persist optional validation.
        6. Save an optional complete checkpoint.

        Args:
            train_session:
                Session containing the components and progress required for
                training.

            infer_session:
                Optional session used for validation after each epoch. When
                provided, it must reference the same model instance and device
                as ``train_session``.

        Raises:
            ValueError:
                If the training and inference sessions do not reference the
                same model instance or use different devices.

            RuntimeError:
                If a training model output does not contain the fixed ``loss``
                field, or training validation does not return evaluation
                results.

            OSError:
                If workspace artifact or checkpoint persistence fails.
        """
        if infer_session is not None:
            self._validate_sessions(
                train_session=train_session,
                infer_session=infer_session,
            )

        for epoch in range(
            train_session.start_epoch,
            train_session.total_epochs,
        ):
            self.workspace.set_stage("train")

            train_session.set_current_epoch(epoch)
            train_session.set_model_train_mode()

            self._run_epoch(train_session)

            if infer_session is not None:
                self._run_validation(
                    infer_session=infer_session,
                    completed_epoch=epoch,
                )

            self._save_checkpoint_if_enabled(train_session)

    def _validate_sessions(
        self,
        *,
        train_session: TrainSession,
        infer_session: InferSession,
    ) -> None:
        """Validate resources shared by training and inference.

        Validation must use the exact model instance updated by the training
        optimizer. Matching model structures or state dictionaries do not
        guarantee that optimizer updates are visible to the inference session.

        Args:
            train_session:
                Session used by the training loop.

            infer_session:
                Session used for validation.

        Raises:
            ValueError:
                If the sessions do not reference the same model instance or
                use different devices.
        """
        if train_session.model is not infer_session.model:
            raise ValueError(
                "train and inference sessions must reference the same "
                "model instance"
            )

        if train_session.device != infer_session.device:
            raise ValueError(
                "train and inference sessions must use the same device"
            )

    def _run_epoch(
        self,
        session: TrainSession,
    ) -> None:
        """Execute every batch in the current training epoch.

        The current epoch must be set on the session before this method is
        invoked. Batch indices are passed to session operations so they can
        construct the corresponding ``TrainCallContext``.

        Args:
            session:
                Training session whose dataloader and components are executed.
        """
        for current_batch, raw_batch in enumerate(session.train_dataloader):
            batch = cast(DatasetBatch, raw_batch)

            self._run_batch(
                session=session,
                batch=batch,
                current_batch=current_batch,
            )

    def _run_batch(
        self,
        *,
        session: TrainSession,
        batch: DatasetBatch,
        current_batch: int,
    ) -> None:
        """Execute and persist one model optimization step.

        The batch execution order is:

        1. Move the batch to the session device.
        2. Clear gradients from the previous optimization step.
        3. Invoke and parse the model output.
        4. Pass the model loss payload to the criterion.
        5. Parse the criterion output.
        6. Compute gradients from the criterion loss.
        7. Update model parameters.
        8. Advance the learning-rate scheduler.
        9. Publish the resulting training artifacts.

        Args:
            session:
                Training session used to invoke and update components.

            batch:
                Batch produced by an owl training dataloader.

            current_batch:
                Zero-based index of the batch within the current epoch.

        Raises:
            RuntimeError:
                If the parsed model output does not contain the fixed ``loss``
                payload required by the criterion.

            TypeError:
                If a model or criterion output violates its output schema.

            ValueError:
                If a model or criterion output contains unsupported or
                malformed keys.
        """
        batch = session.move_batch_to_device(batch)

        session.clear_optimizer_gradients()

        raw_model_output = session.forward_model(
            batch=batch,
            current_batch=current_batch,
        )
        model_output = parse_model_output(raw_model_output)

        if model_output.loss_output is None:
            raise RuntimeError(
                "training model output must contain the required 'loss' key"
            )

        raw_criterion_output = session.forward_criterion(
            loss_output=model_output.loss_output,
            batch=batch,
            current_batch=current_batch,
        )
        criterion_output = parse_criterion_output(raw_criterion_output)

        session.backward_loss(criterion_output.loss)
        session.update_model_parameters()
        session.update_learning_rate()

        loss = float(criterion_output.loss.detach().item())
        learning_rates = session.get_learning_rates()

        epoch = session.current_epoch + 1
        batch_index = current_batch + 1

        self.workspace.update_train_snapshot(
            current_epoch=epoch,
            total_epoch=session.total_epochs,
            current_batch=batch_index,
            total_batch=session.steps_per_epoch,
            loss=loss,
            learning_rates=learning_rates,
        )

        self.workspace.append_train_history(
            epoch=epoch,
            batch=batch_index,
            loss=loss,
            learning_rates=learning_rates,
        )

        if model_output.metric_outputs:
            self.workspace.append_model_metric_history(
                epoch=epoch,
                batch=batch_index,
                metrics=model_output.metric_outputs,
            )

        if criterion_output.metric_outputs:
            self.workspace.append_criterion_metric_history(
                epoch=epoch,
                batch=batch_index,
                metrics=criterion_output.metric_outputs,
            )

    def _run_validation(
            self,
            *,
            infer_session: InferSession,
            completed_epoch: int,
    ) -> None:
        """Execute and persist validation for one completed training epoch.

        Args:
            infer_session:
                Inference session used to evaluate the model.

            completed_epoch:
                Zero-based index of the training epoch that completed immediately
                before this validation run.
        """
        self.workspace.set_stage("infer")

        results = self.infer_runtime.run(infer_session)

        if results is None:
            raise RuntimeError(
                "training validation must return evaluation results"
            )

        self.workspace.append_evaluation_history(
            results=results,
            epoch=completed_epoch + 1,
        )

    def _save_checkpoint_if_enabled(
        self,
        session: TrainSession,
    ) -> None:
        """Save a complete checkpoint after a successfully completed epoch."""
        if not self.checkpoint_save.autosave:
            return

        display_epoch = session.current_epoch + 1
        path = (
            self.workspace.checkpoint_dir()
            / f"epoch-{display_epoch:04d}.pt"
        )

        torch.save(
            session.create_checkpoint(),
            path,
        )