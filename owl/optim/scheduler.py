"""Built-in learning-rate scheduler factory presets.

This module provides configurable scheduler factories that can be passed to
owl clients. Each preset captures user-defined options and returns a
``SchedulerFactory`` that constructs the scheduler after the optimizer and
training plan become available.
"""

from typing import cast

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, PolynomialLR

from ..schemas.optim import SchedulerConstructor

__all__ = ["constant", "poly"]


def constant() -> SchedulerConstructor:
    """Create a constant learning-rate scheduler factory.

    The returned scheduler preserves the optimizer's initial learning rates
    throughout the complete training run. It is used when no learning-rate
    decay strategy is configured.

    Returns:
        Scheduler factory that creates a constant learning-rate scheduler for
        an injected optimizer.

    Example:
        Configure a constant learning rate:

        >>> scheduler_factory = constant()

        The owl client later injects the optimizer and training plan:

        >>> scheduler = scheduler_factory(
        ...     optimizer=optimizer,
        ...     total_epochs=100,
        ...     total_steps=50000,
        ... )
    """

    def factory(
        *,
        optimizer: Optimizer,
        total_epochs: int,
        total_steps: int,
    ) -> LRScheduler:
        return LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda _: 1.0,
        )

    return cast(SchedulerConstructor, factory)


def poly(*, power: float = 1.0) -> SchedulerConstructor:
    """Create a configurable polynomial learning-rate scheduler factory.

    The polynomial power is configured immediately. The returned factory is
    invoked later by the owl client with the instantiated optimizer and
    training plan.

    The scheduler advances once after every optimizer update and decays the
    learning rate over ``total_steps`` updates.

    Args:
        power:
            Exponent used by the polynomial decay schedule.

    Returns:
        Scheduler factory that creates a ``PolynomialLR`` scheduler for an
        injected optimizer and training plan.

    Example:
        Configure the scheduler before the optimizer is available:

        >>> scheduler_factory = poly(power=0.9)

        The owl client later injects the optimizer and training plan:

        >>> scheduler = scheduler_factory(
        ...     optimizer=optimizer,
        ...     total_epochs=100,
        ...     total_steps=50000,
        ... )
    """

    def factory(
        *,
        optimizer: Optimizer,
        total_epochs: int,
        total_steps: int,
    ) -> LRScheduler:
        return PolynomialLR(
            optimizer=optimizer,
            total_iters=total_steps,
            power=power,
        )

    return cast(SchedulerConstructor, factory)