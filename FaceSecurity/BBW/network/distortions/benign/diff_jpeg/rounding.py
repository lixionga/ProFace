from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function


class _DifferentiableRoundingSTE(Function):
    """Implements differentiable rounding by using rounding in the forward pass and a surrogate in the backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
    ) -> Tensor:
        """Forward pass just applies non-differentiable rounding operation.

        Args:
            ctx (Any): Context variable.
            input (Tensor): Input tensor to be clipped of any shape.

        Returns:
            output (Tensor): Clipped input tensor of the same shape.
        """
        # Save for backward pass
        ctx.save_for_backward(input)
        # Get dtype
        dtype = input.dtype
        # Perform clipping
        output: Tensor = torch.round(input).to(dtype)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        """Implements the backward pass by utilizing STE with surrogate of rounding operation (soft rounding).

        Args:
            ctx (Any): Context variable.
            grad_output (Tensor): Gradient output of previous layer.

        Returns:
            gradient (Tensor): Gradient w.r.t. the input to the rounding operations.
        """
        # Get input tensor
        (input,) = ctx.saved_tensors  # type: Tensor,
        # Make gradient tensor
        grad: Tensor = (3 * (input - torch.round(input)) ** 2) * grad_output
        return grad


differentiable_rounding_ste = _DifferentiableRoundingSTE.apply


class _DifferentiableFloorSTE(Function):
    """Implements differentiable floor by using floor in the forward pass and a surrogate in the backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
    ) -> Tensor:
        """Forward pass just applies non-differentiable floor operation.

        Args:
            ctx (Any): Context variable.
            input (Tensor): Input tensor to be clipped of any shape.

        Returns:
            output (Tensor): Clipped input tensor of the same shape.
        """
        # Save for backward pass
        ctx.save_for_backward(input)
        # Get dtype
        dtype = input.dtype
        # Perform clipping
        output: Tensor = torch.floor(input).to(dtype)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        """Implements the backward pass by utilizing STE with surrogate of rounding operation (floor rounding).

        Args:
            ctx (Any): Context variable.
            grad_output (Tensor): Gradient output of previous layer.

        Returns:
            gradient (Tensor): Gradient w.r.t. the input to the rounding operations.
        """
        # Get input tensor
        (input,) = ctx.saved_tensors  # type: Tensor,
        # Make gradient tensor
        input = input - 0.5
        grad: Tensor = (3 * (input - torch.round(input)) ** 2) * grad_output
        return grad


differentiable_floor_ste = _DifferentiableFloorSTE.apply


def differentiable_polynomial_rounding(input: Tensor) -> Tensor:
    """This function implements differentiable rounding.

    Args:
        input (Tensor): Input tensor of any shape to be rounded.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    output: Tensor = torch.round(input) + (input - torch.round(input)) ** 3
    return output


def differentiable_polynomial_floor(input: Tensor) -> Tensor:
    """This function implements differentiable floor.

    Args:
        input (Tensor): Input tensor of any shape to be floored.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    input = input - 0.5
    output: Tensor = torch.round(input) + (input - torch.round(input)) ** 3
    return output
