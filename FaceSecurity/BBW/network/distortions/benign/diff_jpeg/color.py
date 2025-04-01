from typing import Tuple

import torch
import torch.nn.functional as F
import kornia
from torch import Tensor

M: Tensor = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=torch.float,
).T

B = torch.tensor([0, 128, 128], dtype=torch.float)


def rgb_to_ycbcr(input_rgb: Tensor) -> Tensor:
    """Converts an RGB input to YCbCr.

    Args:
        input_rgb (Tensor): RGB input tensor of the shape [*, 3].

    Returns:
        output_ycbcr (Tensor): YCbCr output tensor of the shape [*, 3].
    """
    # Check if input is a tensor with the correct shape
    assert isinstance(input_rgb, Tensor), f"Given input must be a torch.Tensor, got {type(input_rgb)}."
    assert input_rgb.shape[-1] == 3, f"Last axis of the input must have 3 dimensions, got {input_rgb.shape[-1]}."
    # Get original shape and dtype
    dtype: torch.dtype = input_rgb.dtype
    device: torch.device = input_rgb.device
    # Convert from RGB to YCbCr
    output_ycbcr: Tensor = torch.einsum(
        "...i, ij -> ...j",
        input_rgb,
        M.to(dtype=dtype, device=device),
    )
    output_ycbcr = output_ycbcr + B.to(dtype=dtype, device=device)
    return output_ycbcr


def ycbcr_to_rgb(input_ycbcr: Tensor) -> Tensor:
    """Converts a YCbCr to RGB.

    Args:
        input_ycbcr (Tensor): YCbCr input of the shape [*, 3]

    Returns:
        output_rgb (Tensor): RGB output of the shape [*, 3].
    """
    # Check if input is a tensor with the correct shape
    assert isinstance(input_ycbcr, Tensor), f"Given input must be a torch.Tensor, got {type(input_ycbcr)}."
    assert input_ycbcr.shape[-1] == 3, f"Last axis of the input must have 3 dimensions, got {input_ycbcr.shape[-1]}."
    # Get original shape and dtype
    dtype: torch.dtype = input_ycbcr.dtype
    device: torch.device = input_ycbcr.device
    # Convert from RGB to YCbCr
    output_rgb: Tensor = torch.einsum(
        "...i, ij -> ...j",
        input_ycbcr - B.to(dtype=dtype, device=device),
        torch.inverse(M.to(dtype=dtype, device=device)),
    )
    return output_rgb


def chroma_subsampling(input_ycbcr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """This function implements chroma subsampling as an avg. pool operation.

    Args:
        input_ycbcr (Tensor): YCbCr input tensor of the shape [B, 3, H, W].

    Returns:
        output_y (Tensor): Y component (not-subsampled), shape is [B, H, W].
        output_cb (Tensor): Cb component (subsampled), shape is [B, H // 2, W // 2].
        output_cr (Tensor): Cr component (subsampled), shape is [B, H // 2, W // 2].
    """
    # Get components
    output_y: Tensor = input_ycbcr[:, 0]
    output_cb: Tensor = input_ycbcr[:, 1]
    output_cr: Tensor = input_ycbcr[:, 2]
    # Perform average pooling o Cb and Cr
    output_cb = kornia.geometry.rescale(
        output_cb[None], factor=0.5, interpolation="bilinear", align_corners=False, antialias=True
    )
    output_cr = kornia.geometry.rescale(
        output_cr[None], factor=0.5, interpolation="bilinear", align_corners=False, antialias=True
    )
    return output_y, output_cb[0], output_cr[0]


def chroma_upsampling(input_c: Tensor) -> Tensor:
    """Function performs chroma upsampling.

    Args:
        input_c (Tensor): Cb or Cr component to be upsampled of the shape [B, H, W].

    Returns:
        output_c (Tensor): Upsampled C(b or r) component of the shape [B, H * 2, W * 2].
    """
    # Upsample component
    output_c: Tensor = F.interpolate(input_c[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
    return output_c
