from typing import Any, Optional

import torch
from torch import Tensor
from torch.autograd import Function


class _DifferentiableClippingSTE(Function):
    """Implements differentiable clipping by using clipping in the forward pass and a surrogate in the backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        min: float = 1.0,
        max: float = 255.0,
        scale: float = 1e-03,
    ) -> Tensor:
        """Forward pass just applies non-differentiable clipping.

        Args:
            ctx (Any): Context variable.
            input (Tensor): Input tensor to be clipped of any shape.
            min (float): Minimum value. Smaller values will be clipped to min.
            max (float): Maximum value. Larger values will be clipped to max.
            scale (float): Scale parameter for approximate backward pass. Default 1e-03.

        Returns:
            output (Tensor): Clipped input tensor of the same shape.
        """
        # Save for backward pass
        ctx.save_for_backward(input)
        ctx.min: float = min
        ctx.max: float = max
        ctx.scale: float = scale
        # Get dtype
        dtype = input.dtype
        # Perform clipping
        output: Tensor = input.clip(min=min, max=max).to(dtype)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        """Implements the backward pass by utilizing STE with surrogate of clipping operation (soft clipping).

        Args:
            ctx (Any): Context variable.
            grad_output (Tensor): Gradient output of previous layer.

        Returns:
            gradient (Tensor): Gradient w.r.t. the input to the clipping operations.
        """
        # Get input tensor
        (input,) = ctx.saved_tensors  # type: Tensor,
        # Make gradient tensor
        grad: Tensor = torch.where(torch.logical_and(input >= ctx.min, input <= ctx.max), 1.0, ctx.scale) * grad_output
        return grad, None, None


differentiable_clipping_ste = _DifferentiableClippingSTE.apply


def differentiable_clipping(
    input: Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
    scale: float = 1e-03,
) -> Tensor:
    """This function implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min (Optional[float]): Minimum value.
        max (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.001.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    # Make a copy of the input tensor
    output: Tensor = input.clone()
    # Make differentiable soft clipping
    if max is not None:
        output = torch.where(output > max, max + (output - max) * scale, output)
    if min is not None:
        output = torch.where(output < min, min + (output - min) * scale, output)
    return output
