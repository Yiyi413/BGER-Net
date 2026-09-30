"""Inference runtime implementations.

This module defines the runtime responsible for executing model inference
across the named dataloaders contained in an ``InferSession``.

The runtime selects either evaluation or visualization once at the beginning
of an inference run. Both execution paths move batches to the session device,
invoke the model through the session, and parse raw model outputs before
passing the relevant payload to the configured inference processor.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

from torch import Tensor, inference_mode
from torch.utils.data import DataLoader

from ..data.dataset import DatasetBatch
from ..schemas.outputs.parsed import ParsedModelOutput
from ..schemas.outputs.types import ParsedMetricOutputs, TensorOutputValue
from ..schemas.processors import Evaluator, VisualizationOutputs, Visualizer
from .parser import parse_model_output
from .session.infer import InferSession


class InferRuntime:
    """Runtime responsible for executing model inference.

    The runtime selects one execution path according to the processor stored
    by the inference session:

    - evaluation aggregates metrics independently for every named dataloader;
    - visualization generates and saves named images for every input sample.

    Visualization directories and file names are managed by the runtime.
    Visualizer implementations generate named visualization values and encode
    each individual image at the supplied destination path.

    Args:
        visualization_dir:
            Optional root directory used for visualization outputs. It is
            required when the session contains a visualizer and ignored during
            evaluation.
    """

    def __init__(
        self,
        visualization_dir: Path | None = None,
    ) -> None:
        self.visualization_dir = visualization_dir

    def run(
        self,
        session: InferSession,
    ) -> dict[str, ParsedMetricOutputs] | None:
        """Execute inference across all dataloaders in a session.

        The model is placed in evaluation mode before inference begins and
        autograd is disabled for the complete inference run.

        Evaluation mode returns one metric mapping for every named dataloader.
        Visualization mode saves generated images and returns ``None``.

        Args:
            session:
                Inference session containing the model, target device,
                dataloaders, and selected inference processor.

        Returns:
            Metrics grouped by dataloader name when evaluation is selected.
            ``None`` when visualization is selected.

        Raises:
            RuntimeError:
                If a required parsed evaluation or visualization output is
                missing.
            TypeError:
                If a visualizer returns an invalid output mapping or a
                visualization value that is not a tensor or tensor sequence.
            ValueError:
                If visualization is selected without an output directory, or
                if generated visualization outputs do not satisfy the required
                naming and batch-shape constraints.
        """
        session.set_model_eval_mode()

        with inference_mode():
            if session.evaluator is not None:
                return self._run_evaluation(
                    session=session,
                    evaluator=session.evaluator,
                )

            if session.visualizer is None:
                raise RuntimeError(
                    "inference session does not contain a processor"
                )

            self._run_visualization(
                session=session,
                visualizer=session.visualizer,
            )

        return None

    def _run_evaluation(
        self,
        *,
        session: InferSession,
        evaluator: Evaluator,
    ) -> dict[str, ParsedMetricOutputs]:
        """Evaluate every named dataloader in the session."""
        results: dict[str, ParsedMetricOutputs] = {}

        for dataloader_name, dataloader in session.dataloaders.items():
            evaluator.reset()

            for raw_batch in dataloader:
                batch, parsed_output = self._infer_batch(
                    session=session,
                    raw_batch=raw_batch,
                )

                eval_output = parsed_output.eval_output

                if eval_output is None:
                    raise RuntimeError(
                        "evaluation model output must contain the required "
                        "'eval' key"
                    )

                evaluator.update(
                    eval_output=eval_output,
                    batch=batch,
                )

            results[dataloader_name] = evaluator.compute()

        return results

    def _run_visualization(
        self,
        *,
        session: InferSession,
        visualizer: Visualizer,
    ) -> None:
        """Generate and save visualizations for every named dataloader."""
        if self.visualization_dir is None:
            raise ValueError(
                "visualization_dir is required when using a visualizer"
            )

        for dataloader_name, dataloader in session.dataloaders.items():
            output_dir = self.visualization_dir / dataloader_name
            output_dir.mkdir(parents=True, exist_ok=True)

            self._visualize_dataloader(
                session=session,
                dataloader=dataloader,
                output_dir=output_dir,
                visualizer=visualizer,
            )

    def _visualize_dataloader(
        self,
        *,
        session: InferSession,
        dataloader: DataLoader,
        output_dir: Path,
        visualizer: Visualizer,
    ) -> None:
        """Generate and save named visualizations for one dataloader."""
        for raw_batch in dataloader:
            batch, parsed_output = self._infer_batch(
                session=session,
                raw_batch=raw_batch,
            )

            visual_outputs = parsed_output.visual_outputs

            if not visual_outputs:
                raise RuntimeError(
                    "visualization model output must contain at least one "
                    "'visual:*' key"
                )

            images = visualizer.visualize(
                visual_outputs=visual_outputs,
            )

            if not isinstance(images, dict):
                raise TypeError(
                    "visualizer must return a dictionary of named "
                    "visualization outputs"
                )

            if not images:
                raise ValueError(
                    "visualizer must return at least one visualization output"
                )

            tp_names = batch["tp_name"]

            for output_name, image_batch in self._iter_visualization_outputs(
                images
            ):
                self._validate_visualization_output(
                    output_name=output_name,
                    image_batch=image_batch,
                    batch_size=len(tp_names),
                )

                for image, tp_name in zip(
                    image_batch,
                    tp_names,
                    strict=True,
                ):
                    path = output_dir / f"{tp_name}_{output_name}.png"

                    visualizer.save(
                        image=image.detach().cpu(),
                        path=path,
                    )

    @staticmethod
    def _iter_visualization_outputs(
        images: VisualizationOutputs,
    ) -> Iterator[tuple[str, Tensor]]:
        """Expand visualization outputs into named tensor batches.

        A single tensor keeps its original name. A list or tuple of tensors is
        expanded using one-based numeric suffixes.

        Example:
            {
                "prediction": prediction,
                "edge": [edge1, edge2, edge3],
            }

            becomes:

            ("prediction", prediction)
            ("edge_1", edge1)
            ("edge_2", edge2)
            ("edge_3", edge3)

        Args:
            images:
                Mapping returned by ``Visualizer.visualize``.

        Yields:
            Pairs of expanded output names and corresponding batched tensors.

        Raises:
            TypeError:
                If an output name is not a string, or if an output value is not
                a tensor, list of tensors, or tuple of tensors.
            ValueError:
                If an output name is empty or contains path components, or if
                a list or tuple visualization value is empty.
        """
        for output_name, output_value in images.items():
            if not isinstance(output_name, str):
                raise TypeError(
                    "visualization output names must be strings"
                )

            if (
                not output_name.strip()
                or Path(output_name).name != output_name
            ):
                raise ValueError(
                    "visualization output names must be non-empty file-name "
                    f"components, got {output_name!r}"
                )

            if isinstance(output_value, Tensor):
                yield output_name, output_value
                continue

            if isinstance(output_value, (list, tuple)):
                if not output_value:
                    raise ValueError(
                        "visualization output sequences must not be empty, "
                        f"got {output_name!r}"
                    )

                for index, image_batch in enumerate(output_value, start=1):
                    if not isinstance(image_batch, Tensor):
                        raise TypeError(
                            "each visualization output sequence element must "
                            "be a torch.Tensor"
                        )

                    yield f"{output_name}_{index}", image_batch
                continue

            raise TypeError(
                "visualization outputs must be tensors or non-empty tensor "
                f"sequences, got {type(output_value).__name__} for "
                f"{output_name!r}"
            )

    @staticmethod
    def _validate_visualization_output(
        *,
        output_name: str,
        image_batch: Tensor,
        batch_size: int,
    ) -> None:
        """Validate one expanded batch of visualization images.

        Args:
            output_name:
                Marker appended to the TP name when generating output files.
            image_batch:
                Batched visualization tensor returned by the visualizer or
                expanded from a visualization tensor sequence.
            batch_size:
                Number of TP names in the corresponding dataset batch.

        Raises:
            TypeError:
                If the image batch is not a tensor.
            ValueError:
                If the image tensor does not have the required shape and batch
                size.
        """
        if not isinstance(image_batch, Tensor):
            raise TypeError(
                "each visualization output must be a torch.Tensor"
            )

        if image_batch.ndim != 4:
            raise ValueError(
                "visualization output must have shape [B, C, H, W], "
                f"got name={output_name!r} and "
                f"shape={tuple(image_batch.shape)}"
            )

        if image_batch.shape[0] != batch_size:
            raise ValueError(
                "visualization output batch size must match tp_name count, "
                f"got name={output_name!r}, "
                f"images={image_batch.shape[0]}, and names={batch_size}"
            )

    @staticmethod
    def _infer_batch(
        *,
        session: InferSession,
        raw_batch: object,
    ) -> tuple[DatasetBatch, ParsedModelOutput]:
        """Move, infer, and parse one dataloader batch."""
        batch = cast(DatasetBatch, raw_batch)
        batch = session.move_batch_to_device(batch)

        raw_output = session.forward_model(batch)
        parsed_output = parse_model_output(raw_output)

        return batch, parsed_output