"""Learning-rate scheduler constructor protocol definitions.

This module defines the callable protocol used by owl to construct
learning-rate schedulers after the optimizer and training plan have been
resolved.

A scheduler constructor is typically created during configuration, where it
captures user-defined options such as the warmup duration and minimum
learning-rate ratio. Owl later injects the resolved optimizer and training-plan
values and invokes the constructor to create the scheduler used by the training
session.
"""

from typing import Protocol

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SchedulerConstructor(Protocol):
    """Callable that constructs a scheduler from injected training objects.

    Scheduler construction is commonly divided into two stages.

    The outer configuration function receives user-defined options such as the
    warmup duration and minimum learning-rate ratio. It captures those options
    and returns a scheduler constructor:

        >>> import math
        >>>
        >>> from torch.optim import Optimizer
        >>> from torch.optim.lr_scheduler import LambdaLR, LRScheduler
        >>>
        >>> def create_cosine_scheduler(
        ...     *,
        ...     min_lr_ratio: float = 0.0,
        ... ) -> SchedulerConstructor:
        ...     def constructor(
        ...         *,
        ...         optimizer: Optimizer,
        ...         total_epochs: int,
        ...         total_steps: int,
        ...     ) -> LRScheduler:
        ...         def lr_lambda(step: int) -> float:
        ...             progress = min(step / total_steps, 1.0)
        ...             cosine = 0.5 * (
        ...                 1.0 + math.cos(math.pi * progress)
        ...             )
        ...             return (
        ...                 min_lr_ratio
        ...                 + (1.0 - min_lr_ratio) * cosine
        ...             )
        ...
        ...         return LambdaLR(
        ...             optimizer,
        ...             lr_lambda,
        ...         )
        ...
        ...     return constructor

    The constructor can be configured before the optimizer and training plan
    are available:

        >>> scheduler_constructor = create_cosine_scheduler(
        ...     min_lr_ratio=0.01,
        ... )

    Owl later injects the resolved optimizer and training-plan values:

        >>> scheduler = scheduler_constructor(
        ...     optimizer=optimizer,
        ...     total_epochs=100,
        ...     total_steps=50000,
        ... )

    Implementations do not need to inherit from this protocol. Any compatible
    callable is accepted, including functions, closures, callable instances,
    and class objects with a compatible call signature.

    The outer configuration function is not constrained by this protocol.
    Different scheduler implementations may expose different configuration
    parameters.

    A scheduler constructor may ignore training-plan values that are not needed
    by its scheduling strategy.
    """

    def __call__(
        self,
        *,
        optimizer: Optimizer,
        total_epochs: int,
        total_steps: int,
    ) -> LRScheduler:
        """Construct a scheduler from injected training objects.

        Args:
            optimizer:
                Resolved optimizer whose learning rate will be scheduled.
            total_epochs:
                Total number of epochs configured for the training run.
            total_steps:
                Total number of planned optimizer updates across the complete
                training run.

        Returns:
            Learning-rate scheduler associated with the supplied optimizer.
        """
        ...