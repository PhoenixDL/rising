import torch
import random
from typing import Union, Sequence

from rising.utils import check_scalar

__all__ = ["crop", "center_crop", "random_crop"]


def crop(data: torch.Tensor, corner: Sequence[int], size: Sequence[int],
         grid_crop: bool = False):
    """
    Extract crop from last dimensions of data

    Parameters
    ----------
    data: torch.Tensor
        input tensor [... , spatial dims] spatial dims can be arbitrary
        spatial dimensions. Leading dimensions will be preserved.
    corner: Sequence[int]
        top left corner point
    size: Sequence[int]
        size of patch
    grid_crop: bool
        crop from grid of shape [N, spatial dims, NDIM], where N is the batch
        size, spatial dims can be arbitrary spatial dimensions and NDIM
        is the number of spatial dimensions

    Returns
    -------
    torch.Tensor
        cropped data
    """
    _slices = []
    ndim = data.ndimension() - int(bool(grid_crop))
    if len(corner) < ndim:
        for i in range(ndim - len(corner)):
            _slices.append(slice(0, data.shape[i]))

    _slices = _slices + [slice(c, c + s) for c, s in zip(corner, size)]
    if grid_crop:
        _slices.append(slice(0, data.shape[-1]))
    return data[_slices]


def center_crop(data: torch.Tensor, size: Union[int, Sequence[int]],
                grid_crop: bool = False) -> torch.Tensor:
    """
    Crop patch from center

    Parameters
    ----------
    data: torch.Tensor
        input tensor [... , spatial dims] spatial dims can be arbitrary
        spatial dimensions. Leading dimensions will be preserved.
    size: Union[int, Sequence[int]]
        size of patch
    grid_crop: bool
        crop from grid of shape [N, spatial dims, NDIM], where N is the batch
        size, spatial dims can be arbitrary spatial dimensions and NDIM
        is the number of spatial dimensions

    Returns
    -------
    torch.Tensor
        output tensor cropped from input tensor
    """
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]

    if grid_crop:
        data_shape = data.shape[1:-1]
    else:
        data_shape = data.shape[2:]

    corner = [int(round((img_dim - crop_dim) / 2.)) for img_dim, crop_dim in zip(data_shape, size)]
    return crop(data, corner, size, grid_crop=grid_crop)


def random_crop(data: torch.Tensor, size: Union[int, Sequence[int]],
                dist: Union[int, Sequence[int]] = 0,
                grid_crop: bool = False) -> torch.Tensor:
    """
    Crop random patch/volume from input tensor

    Parameters
    ----------
    data: torch.Tensor
        input tensor [... , spatial dims] spatial dims can be arbitrary
        spatial dimensions. Leading dimensions will be preserved.
    size: Union[int, Sequence[int]]
        size of patch/volume
    dist: Union[int, Sequence[int]]
        minimum distance to border. By default zero
    grid_crop: bool
        crop from grid of shape [N, spatial dims, NDIM], where N is the batch
        size, spatial dims can be arbitrary spatial dimensions and NDIM
        is the number of spatial dimensions

    Returns
    -------
    torch.Tensor
        cropped output
    """
    if check_scalar(dist):
        dist = [dist] * (data.ndim - 2)
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]

    if grid_crop:
        data_shape = data.shape[1:-1]
    else:
        data_shape = data.shape[2:]

    if any([crop_dim + dist_dim >= img_dim for img_dim, crop_dim, dist_dim in zip(data_shape, size, dist)]):
        raise TypeError(f"Crop can not be realized with given size {size} and dist {dist}.")

    corner = [random.randrange(0, img_dim - crop_dim - dist_dim) for
              img_dim, crop_dim, dist_dim in zip(data_shape, size, dist)]
    return crop(data, corner, size, grid_crop=grid_crop)
