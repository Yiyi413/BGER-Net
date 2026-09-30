"""Optimizer constructor protocol definitions.

This module defines the callable protocol used by owl to construct optimizers
after the model has been resolved.

An optimizer constructor is typically created during configuration, where it
captures user-defined options such as the learning rate and weight decay. Owl
later injects the resolved model and invokes the constructor to create the
optimizer used by the training session.
"""

from typing import Protocol

from torch.nn import Module
from torch.optim import Optimizer


class OptimizerConstructor(Protocol):
    """Callable that constructs an optimizer from an injected model.

    Optimizer construction is commonly divided into two stages.

    The outer configuration function receives user-defined options such as the
    learning rate and weight decay. It captures those options and returns an
    optimizer constructor:

        >>> from torch.nn import Module
        >>> from torch.optim import AdamW, Optimizer
        >>>
        >>> def create_adamw(
        ...     *,
        ...     lr: float = 1e-4,
        ...     weight_decay: float = 1e-2,
        ... ) -> OptimizerConstructor:
        ...     def constructor(*, model: Module) -> Optimizer:
        ...         return AdamW(
        ...             model.parameters(),
        ...             lr=lr,
        ...             weight_decay=weight_decay,
        ...         )
        ...
        ...     return constructor

    The constructor can be configured before the model is available:

        >>> optimizer_constructor = create_adamw(
        ...     lr=2e-4,
        ...     weight_decay=1e-2,
        ... )

    Owl later injects the resolved model:

        >>> optimizer = optimizer_constructor(model=model)

    Implementations do not need to inherit from this protocol. Any compatible
    callable is accepted, including functions, closures, callable instances,
    and class objects with a compatible call signature.

    The outer configuration function is not constrained by this protocol.
    Different optimizer implementations may expose different configuration
    parameters.
    """

    def __call__(self, *, model: Module) -> Optimizer:
        """Construct an optimizer for the injected model.

        Args:
            model:
                Resolved model whose parameters will be optimized.

        Returns:
            Optimizer associated with the supplied model.
        """
        ...