import torch
from typing import Sequence, Union

from rising.utils import check_scalar

__all__ = ["mirror", "rot90", "resize"]


def mirror(data: torch.Tensor, dims: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Mirror data at dims

    Parameters
    ----------
    data: torch.Tensor
        input data
    dims: int or Sequence[int]
        dimensions to mirror

    Returns
    -------
    torch.Tensor
        tensor with mirrored dimensions
    """
    if check_scalar(dims):
        dims = (dims,)
    dims = [d + 2 for d in dims]
    return data.flip(dims)


def rot90(data: torch.Tensor, k: int, dims: Union[int, Sequence[int]]):
    """
    Rotate 90 degrees around dims

    Parameters
    ----------
    data: torch.Tensor
        input data
    k: int
        number of times to rotate
    dims: int or Sequence[int]
        dimensions to mirror

    Returns
    -------
    torch.Tensor
        tensor with mirrored dimensions
    """
    dims = [d + 2 for d in dims]
    return torch.rot90(data, k, dims)


def resize(data: torch.Tensor, size: Union[int, Sequence[int]] = None,
           scale_factor: Union[float, Sequence[float]] = None,
           mode: str = 'nearest', align_corners: bool = None,
           preserve_range: bool = False):
    """
    Down/up-sample sample to either the given :param:`size` or the given :param:`scale_factor`
    The modes available for resizing are: nearest, linear (3D-only), bilinear,
    bicubic (4D-only), trilinear (5D-only), area

    Parameters
    ----------
    data: torch.Tensor
        input tensor of shape batch x channels x height x width x [depth]
    size: Union[int, Sequence[int]]
        output size (with channel and batch dim)
    scale_factor: Union[int, Sequence[int]]
        multiplier for spatial size
    mode: str
        one of :param:`nearest`, :param:`linear`, :param:`bilinear`, :param:`bicubic`,
        :param:`trilinear`, :param:`area` (for more inforamtion see :func:`torch.nn.functional.interpolate`
    align_corners: bool
        input and output tensors are aligned by the center points of their corners pixels,
        preserving the values at the corner pixels.
    preserve_range: bool
        output tensor has same range as input tensor

    Returns
    -------
    torch.Tensor
        interpolated tensor

    See Also
    --------
    :func:`torch.nn.functional.interpolate`
    """
    out = torch.nn.functional.interpolate(data, size=size, scale_factor=scale_factor,
                                          mode=mode, align_corners=align_corners)

    if preserve_range:
        out.clamp_(data.min(), data.max())
    return out
