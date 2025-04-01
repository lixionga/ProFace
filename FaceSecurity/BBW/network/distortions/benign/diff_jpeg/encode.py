import math
from typing import Callable, Tuple

import torch
from torch import Tensor

from .color import rgb_to_ycbcr, chroma_subsampling
from .utils import jpeg_quality_to_scale


def patchify_8x8(input: Tensor) -> Tensor:
    """Function extracts non-overlapping 8 x 8 patches from the given input image.

    Args:
        input (Tensor): Input image of the shape [B, H, W].

    Returns:
        output (Tensor): Image patchify of the shape [B, N, 8, 8]
    """
    # Get input shape
    B, H, W = input.shape  # type: int, int, int
    # Patchify to shape [B, N, H // 8, W // 8]
    output: Tensor = input.view(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4).reshape(B, -1, 8, 8)
    return output


def dct_8x8(input: Tensor) -> Tensor:
    """Performs a 8 x 8 discrete cosine transform.

    Args:
        input (Tensor): Patched input tensor of the shape [B, N, 8, 8].

    Returns:
        output (Tensor): DCT output tensor of the shape [B, N, 8, 8].
    """
    # Get dtype and device
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)  # type: Tensor, Tensor, Tensor, Tensor
    dct_tensor: Tensor = ((2.0 * x + 1.0) * u * torch.tensor(math.pi) / 16.0).cos() * ((2.0 * y + 1.0) * v * torch.tensor(math.pi) / 16.0).cos()
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.einsum("i, j -> ij", alpha, alpha) * 0.25
    # Apply DCT
    output: Tensor = dct_scale[None, None] * torch.tensordot(input - 128.0, dct_tensor)
    return output


def quantize(
    input: Tensor,
    jpeg_quality: Tensor,
    quantization_table: Tensor,
    rounding_function: Callable[[Tensor], Tensor],
    floor_function: Callable[[Tensor], Tensor],
    clipping_function: Callable[[Tensor, int, int], Tensor],
) -> Tensor:
    """Function performs quantization.

    Args:
        input (Tensor): Input tensor of the shape [B, N, 8, 8].
        jpeg_quality (Tensor): Compression strength to be applied, shape is [B].
        quantization_table (Tensor): Quantization table of the shape [8, 8].
        rounding_function (Callable[[Tensor], Tensor): Rounding function to be used.
        floor_function (Callable[[Tensor], Tensor): Floor function to be used.
        clipping_function (Callable[[Tensor, int, int], Tensor]): Clipping function to be used.

    Returns:
        output (Tensor): Quantized output tensor of the shape [B, N, 8, 8].
    """
    # Scale quantization table
    quantization_table_scaled: Tensor = (
            quantization_table[None, None]
            * jpeg_quality_to_scale(jpeg_quality, floor_function=floor_function)[:, None, None, None]
    )
    # Perform scaling
    quantization_table: Tensor = clipping_function(floor_function((quantization_table_scaled + 50.0) / 100.0), 1, 255)
    output: Tensor = input / quantization_table
    # Perform rounding
    output = rounding_function(output)
    return output


def jpeg_encode(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    rounding_function: Callable[[Tensor], Tensor],
    floor_function: Callable[[Tensor], Tensor],
    clipping_function: Callable[[Tensor, int, int], Tensor],
    quantization_table_y: Tensor,
    quantization_table_c: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
        jpeg_quality (Tensor): Compression strength of the shape [B].
        rounding_function (Callable[[Tensor], Tensor): Rounding function to be used.
        floor_function (Callable[[Tensor], Tensor): Floor function to be used.
        clipping_function (Callable[[Tensor, int, int], Tensor]): Clipping function to be used.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        y_encoded (Tensor): Encoded Y component of the shape [B, N, 8, 8].
        cb_encoded (Tensor): Encoded Cb component of the shape [B, N, 8, 8].
        cr_encoded (Tensor): Encoded Cr component of the shape [B, N, 8, 8].
    """
    # Check inputs
    assert isinstance(image_rgb, Tensor), f"Input image (image_rgb) must be a torch.Tensor, got {type(image_rgb)}."
    assert isinstance(
        jpeg_quality, Tensor
    ), f"Compression strength (jpeg_quality) must be a torch.Tensor, got {type(jpeg_quality)}."
    assert image_rgb.ndim == 4, f"Input image (image_rgb) must be a 4D tensor, got shape {image_rgb.shape}."
    assert (
        image_rgb.shape[1] == 3
    ), f"Input (image_rgb) must be a batch of RGB image, got shape {image_rgb.shape}."
    assert image_rgb.shape[0] == jpeg_quality.shape[0], (
        f"Batch size of image_rgb and jpeg_quality must match, "
        f"got image shape {image_rgb.shape[0]} and jpeg_quality shape {jpeg_quality.shape[0]}"
    )
    assert isinstance(
        quantization_table_y, Tensor
    ), f"QT (quantization_table_y) must be a torch.Tensor, got {type(quantization_table_y)}."
    assert isinstance(
        quantization_table_c, Tensor
    ), f"QT (quantization_table_c) must be a torch.Tensor, got {type(quantization_table_c)}."
    assert quantization_table_y.shape == (8, 8,), (
        f"QT (quantization_table_y) must have the shape [8, 8], " f"got {quantization_table_y.shape}"
    )
    assert quantization_table_c.shape == (8, 8,), (
        f"QT (quantization_table_c) must have the shape [8, 8], " f"got {quantization_table_c.shape}"
    )
    # QT to device
    quantization_table_y = quantization_table_y.to(image_rgb.dtype).to(image_rgb.device)
    quantization_table_c = quantization_table_c.to(image_rgb.dtype).to(image_rgb.device)
    # Convert RGB image to YCbCr
    image_ycbcr: Tensor = rgb_to_ycbcr(image_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    # Perform chroma subsampling
    input_y, input_cb, input_cr = chroma_subsampling(image_ycbcr)  # type: Tensor, Tensor, Tensor
    # Patchify, DCT, and rounding
    input_y, input_cb, input_cr = patchify_8x8(input_y), patchify_8x8(input_cb), patchify_8x8(input_cr)
    dct_y, dct_cb, dct_cr = dct_8x8(input_y), dct_8x8(input_cb), dct_8x8(input_cr)  # type: Tensor, Tensor, Tensor
    y_encoded: Tensor = quantize(
        dct_y,
        jpeg_quality,
        quantization_table_y,
        rounding_function=rounding_function,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )
    cb_encoded: Tensor = quantize(
        dct_cb,
        jpeg_quality,
        quantization_table_c,
        rounding_function=rounding_function,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )
    cr_encoded: Tensor = quantize(
        dct_cr,
        jpeg_quality,
        quantization_table_c,
        rounding_function=rounding_function,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )
    return y_encoded, cb_encoded, cr_encoded
