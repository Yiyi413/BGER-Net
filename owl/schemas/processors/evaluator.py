"""Evaluator protocol definitions.

This module defines the protocol used by inference runtimes to evaluate model
outputs over a complete dataloader.
"""

from typing import Protocol

from ...data.dataset import DatasetBatch
from ..outputs.types import ParsedMetricOutputs, TensorOutputValue


class Evaluator(Protocol):
    """Protocol implemented by inference evaluators.

    Evaluation is performed independently for every named dataloader.

    Before processing a dataloader, the inference runtime invokes ``reset`` to
    clear previously accumulated snapshot. It then invokes ``update`` once for
    every batch and finally invokes ``compute`` to obtain the aggregated
    dataset-level metrics.

    The model's fixed ``eval`` output may contain either:

    - one tensor representing a single prediction output; or
    - an ordered list or tuple of tensors representing multiple prediction
      outputs, such as progressive decoder stages or auxiliary supervision
      branches.

    The runtime passes this payload to ``update`` unchanged. It does not split
    multiple tensors or invoke ``update`` separately for each tensor.
    Evaluator implementations are therefore responsible for interpreting the
    payload structure and accumulating the required metrics.

    Example:
        Define an evaluator that supports both single-output and multi-output
        models while accumulating image-level F1 scores without retaining
        predictions from previous batches:

        >>> import torch
        >>> from torch import Tensor
        >>>
        >>> class ProgressiveEvaluator:
        ...     def __init__(self) -> None:
        ...         self.f1_sums: list[float] = []
        ...         self.sample_count = 0
        ...
        ...     def reset(self) -> None:
        ...         self.f1_sums.clear()
        ...         self.sample_count = 0
        ...
        ...     def update(
        ...         self,
        ...         *,
        ...         eval_output: TensorOutputValue,
        ...         batch: DatasetBatch,
        ...     ) -> None:
        ...         if isinstance(eval_output, Tensor):
        ...             predictions = [eval_output]
        ...         else:
        ...             predictions = list(eval_output)
        ...
        ...         if not self.f1_sums:
        ...             self.f1_sums = [0.0] * len(predictions)
        ...
        ...         if len(predictions) != len(self.f1_sums):
        ...             raise ValueError(
        ...                 "evaluation output count changed between batches"
        ...             )
        ...
        ...         target = batch["gt"] >= 0.5
        ...         reduce_dims = tuple(range(1, target.ndim))
        ...
        ...         for index, prediction in enumerate(predictions):
        ...             binary_prediction = torch.sigmoid(prediction) >= 0.5
        ...
        ...             true_positive = (
        ...                 binary_prediction & target
        ...             ).sum(dim=reduce_dims)
        ...             false_positive = (
        ...                 binary_prediction & ~target
        ...             ).sum(dim=reduce_dims)
        ...             false_negative = (
        ...                 ~binary_prediction & target
        ...             ).sum(dim=reduce_dims)
        ...
        ...             denominator = (
        ...                 2 * true_positive
        ...                 + false_positive
        ...                 + false_negative
        ...             )
        ...             batch_f1 = torch.where(
        ...                 denominator > 0,
        ...                 2 * true_positive / denominator,
        ...                 torch.zeros_like(
        ...                     denominator,
        ...                     dtype=torch.float32,
        ...                 ),
        ...             )
        ...
        ...             self.f1_sums[index] += batch_f1.sum().item()
        ...
        ...         self.sample_count += target.shape[0]
        ...
        ...     def compute(self) -> ParsedMetricOutputs:
        ...         metrics: ParsedMetricOutputs = {}
        ...         multiple_outputs = len(self.f1_sums) > 1
        ...
        ...         for index, f1_sum in enumerate(
        ...             self.f1_sums,
        ...             start=1,
        ...         ):
        ...             metric_name = (
        ...                 f"stage_{index}/pixel_f1"
        ...                 if multiple_outputs
        ...                 else "pixel_f1"
        ...             )
        ...             metrics[metric_name] = (
        ...                 f1_sum / self.sample_count
        ...                 if self.sample_count > 0
        ...                 else 0.0
        ...             )
        ...
        ...         return metrics

        The inference runtime uses the evaluator as follows:

        >>> evaluator.reset()
        >>> evaluator.update(
        ...     eval_output=eval_output,
        ...     batch=batch,
        ... )
        >>> metrics = evaluator.compute()

    Notes:
        The runtime guarantees that ``eval_output`` is present before invoking
        ``update``. Evaluators therefore do not need to handle ``None``.

        A single tensor is conventionally treated as one prediction output.

        A list or tuple is conventionally treated as an ordered collection of
        prediction outputs. Its order is preserved from the model output. For
        example, ``[pred_1, pred_2, pred_3]`` may represent three progressive
        decoder stages.

        Owl does not assign semantic names to tensors in a list or tuple.
        Evaluators must decide what each position means and how each output is
        evaluated.

        When multiple outputs are evaluated, the metric keys returned by
        ``compute`` should include an unambiguous stage or output marker.
        Recommended names include::

            {
                "stage_1/pixel_f1": 0.71,
                "stage_1/pixel_auc": 0.82,
                "stage_2/pixel_f1": 0.75,
                "stage_2/pixel_auc": 0.86,
                "stage_3/pixel_f1": 0.79,
                "stage_3/pixel_auc": 0.90,
            }

        Numeric suffixes such as ``pixel_f1_1`` are also valid, but hierarchical
        names such as ``stage_1/pixel_f1`` are generally easier to group in
        logs and reports.

        Evaluators may extract any required ground-truth masks, image-level
        labels, or other metadata from the supplied dataset batch.

        Evaluators may retain complete predictions when required by a metric.
        Metrics that support streaming aggregation should generally accumulate
        only the running statistics required by ``compute``.
    """

    def reset(self) -> None:
        """Clear all snapshot accumulated from a previous dataloader.

        The runtime invokes this method once before processing each named
        dataloader. Implementations must clear all running statistics and any
        other dataset-specific snapshot accumulated during the previous
        evaluation.
        """
        ...

    def update(
        self,
        *,
        eval_output: TensorOutputValue,
        batch: DatasetBatch,
    ) -> None:
        """Accumulate evaluation data from one inference batch.

        Args:
            eval_output:
                Parsed tensor payload produced from the model's fixed ``eval``
                output key.

                A single tensor represents one prediction output. A list or
                tuple represents an ordered collection of prediction outputs.
                The runtime passes the payload unchanged, so the evaluator is
                responsible for interpreting and accumulating every tensor it
                intends to evaluate.
            batch:
                Dataset batch containing the ground-truth masks, labels, or
                metadata required by the evaluator.
        """
        ...

    def compute(self) -> ParsedMetricOutputs:
        """Compute metrics accumulated over the complete dataloader.

        Returns:
            Named metric values for the evaluated dataloader.

            Evaluators processing multiple prediction outputs should include
            the corresponding stage or output marker in each metric key, such
            as ``stage_1/pixel_f1`` and ``stage_2/pixel_f1``.
        """
        ...
