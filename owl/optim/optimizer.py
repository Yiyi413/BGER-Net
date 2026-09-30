"""Built-in optimizer factory presets.

This module provides configurable optimizer factories that can be passed to
owl clients. Each preset captures user-defined options and returns an
``OptimizerFactory`` that constructs the optimizer after the model becomes
available.
"""

from typing import cast

from torch.nn import Module
from torch.optim import AdamW, Optimizer

from ..schemas.optim import OptimizerConstructor

__all__ = ["adamw"]


def adamw(
    *,
    lr: float,
    weight_decay: float,
) -> OptimizerConstructor:
    """Create a configurable AdamW optimizer factory.

    The learning rate and weight decay are configured immediately. The
    returned factory is invoked later by the owl client with the instantiated
    model.

    Args:
        lr:
            Learning rate used by the AdamW optimizer.
        weight_decay:
            Weight-decay coefficient applied during optimization.

    Returns:
        Optimizer factory that creates an AdamW optimizer for an injected
        model.

    Example:
        Configure the optimizer before the model is available:

        >>> optimizer_factory = adamw(
        ...     lr=1e-4,
        ...     weight_decay=1e-2,
        ... )

        The owl client later injects the instantiated model:

        >>> optimizer = optimizer_factory(model=model)
    """

    def factory(*, model: Module) -> Optimizer:
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return cast(OptimizerConstructor, factory)