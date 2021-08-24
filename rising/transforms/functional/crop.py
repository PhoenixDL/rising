import random
from typing import List, Sequence, Tuple, Union

import torch

from rising.utils import check_scalar

__all__ = ["crop", "center_crop", "random_crop"]


def crop(data: torch.Tensor, corner: Sequence[int], size: Sequence[int]) -> torch.Tensor:
    """
    Extract crop from last dimensions of data

    Args:
    data: input tensor
    corner: top left corner point
    size: size of patch

    Returns:
        torch.Tensor: cropped data
    """
    _slices = []
    if len(corner) < data.ndim:
        for i in range(data.ndim - len(corner)):
            _slices.append(slice(0, data.shape[i]))

    _slices = _slices + [slice(c, c + s) for c, s in zip(corner, size)]
    return data[_slices]


def center_crop(data: torch.Tensor, size: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Crop patch from center

    Args:
    data: input tensor
    size: size of patch

    Returns:
        torch.Tensor: output tensor cropped from input tensor
    """
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]

    corner = [int(round((img_dim - crop_dim) / 2.0)) for img_dim, crop_dim in zip(data.shape[2:], size)]
    return crop(data, corner, size)


def random_crop(
    data: torch.Tensor, size: Union[int, Sequence[int]], dist: Union[int, Sequence[int]] = 0
) -> torch.Tensor:
    """
    Crop random patch/volume from input tensor

    Args:
        data: input tensor
        size: size of patch/volume
        dist: minimum distance to border. By default zero

    Returns:
        torch.Tensor: cropped output
        List[int]: top left corner used for crop
    """
    if check_scalar(dist):
        dist = [dist] * (data.ndim - 2)
    if isinstance(dist[0], torch.Tensor):
        dist = [int(i) for i in dist]
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]

    if any([crop_dim + dist_dim >= img_dim for img_dim, crop_dim, dist_dim in zip(data.shape[2:], size, dist)]):
        raise TypeError(f"Crop can not be realized with given size {size} and dist {dist}.")

    corner = [
        torch.randint(0, img_dim - crop_dim - dist_dim, (1,)).item()
        for img_dim, crop_dim, dist_dim in zip(data.shape[2:], size, dist)
    ]
    return crop(data, corner, size)
