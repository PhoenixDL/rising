import torch
from typing import Sequence, Union, Optional

from rising.utils import check_scalar

__all__ = ["mirror", "rot90", "resize"]


def mirror(data: torch.Tensor, dims: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Mirror data at dims

    Args:
        data: input data
        dims: dimensions to mirror

    Returns:
        tensor with mirrored dimensions
    """
    if check_scalar(dims):
        dims = (dims,)
    # batch and channel dims
    dims = [d + 2 for d in dims]
    return data.flip(dims)


def rot90(data: torch.Tensor, k: int, dims: Union[int, Sequence[int]]):
    """
    Rotate 90 degrees around dims

    Args:
        data: input data
        k: number of times to rotate
        dims: dimensions to mirror

    Returns:
        tensor with mirrored dimensions
    """
    dims = [d + 2 for d in dims]
    return torch.rot90(data, k, dims)


def resize(data: torch.Tensor,
           size: Optional[Union[int, Sequence[int]]] = None,
           scale_factor: Optional[Union[float, Sequence[float]]] = None,
           mode: str = 'nearest', align_corners: Optional[bool] = None,
           preserve_range: bool = False):
    """
    Down/up-sample sample to either the given :attr:`size` or the given
    :attr:`scale_factor`
    The modes available for resizing are: nearest, linear (3D-only), bilinear,
    bicubic (4D-only), trilinear (5D-only), area

    Args:
        data: input tensor of shape batch x channels x height x width x [depth]
        size: spatial output size (excluding batch size and number of channels)
        scale_factor: multiplier for spatial size
        mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
            ``trilinear``, ``area``
            (for more inforamtion see :func:`torch.nn.functional.interpolate`)
        align_corners: input and output tensors are aligned by the center
            points of their corners pixels, preserving the values at the
            corner pixels.
        preserve_range:  output tensor has same range as input tensor

    Returns:
        interpolated tensor

    See Also:
        :func:`torch.nn.functional.interpolate`
    """
    out = torch.nn.functional.interpolate(data, size=size, scale_factor=scale_factor,
                                          mode=mode, align_corners=align_corners)

    if preserve_range:
        out.clamp_(data.min(), data.max())
    return out
