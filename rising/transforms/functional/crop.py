import torch
import random
from typing import Union, Sequence

from rising.utils import check_scalar

__all__ = ["crop", "center_crop", "random_crop"]


def crop(data: torch.Tensor, corner: Sequence[int], size: Sequence[int],
         exclude_last: bool = False):
    """
    Extract crop from last dimensions of data

    Parameters
    ----------
    data: torch.Tensor
        input tensor
    corner: Sequence[int]
        top left corner point
    size: Sequence[int]
        size of patch
    exclude_last: bool
        exclude last dimension from cropping (this is used to crop grids
        from :class:`GridTransform`

    Returns
    -------
    torch.Tensor
        cropped data
    """
    _slices = []
    ndim = data.ndimension() - int(bool(exclude_last))
    if len(corner) < ndim:
        for i in range(ndim - len(corner)):
            _slices.append(slice(0, data.shape[i]))

    _slices = _slices + [slice(c, c + s) for c, s in zip(corner, size)]
    if exclude_last:
        _slices.append(_slices.append(slice(0, data.shape[-1])))
    print(_slices)
    return data[_slices]


def center_crop(data: torch.Tensor, size: Union[int, Sequence[int]],
                exclude_last: bool = False) -> torch.Tensor:
    """
    Crop patch from center

    Parameters
    ----------
    data: torch.Tensor
        input tensor
    size: Union[int, Sequence[int]]
        size of patch
    exclude_last: bool
        exclude last dimension from cropping (this is used to crop grids
        from :class:`GridTransform`

    Returns
    -------
    torch.Tensor
        output tensor cropped from input tensor
    """
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]

    if exclude_last:
        size = size[:-1]
        data_shape = data.shape[2:-1]
    else:
        data_shape = data.shape[2:]

    corner = [int(round((img_dim - crop_dim) / 2.)) for img_dim, crop_dim in zip(data_shape, size)]
    return crop(data, corner, size)


def random_crop(data: torch.Tensor, size: Union[int, Sequence[int]],
                dist: Union[int, Sequence[int]] = 0,
                exclude_last: bool = False) -> torch.Tensor:
    """
    Crop random patch/volume from input tensor

    Parameters
    ----------
    data: torch.Tensor
        input tensor
    size: Union[int, Sequence[int]]
        size of patch/volume
    dist: Union[int, Sequence[int]]
        minimum distance to border. By default zero
    exclude_last: bool
        exclude last dimension from cropping (this is used to crop grids
        from :class:`GridTransform`

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

    if exclude_last:
        size = size[:-1]
        data_shape = data.shape[2:-1]
    else:
        data_shape = data.shape[2:]

    if any([crop_dim + dist_dim >= img_dim for img_dim, crop_dim, dist_dim in zip(data_shape, size, dist)]):
        raise TypeError(f"Crop can not be realized with given size {size} and dist {dist}.")

    corner = [random.randrange(0, img_dim - crop_dim - dist_dim) for
              img_dim, crop_dim, dist_dim in zip(data_shape, size, dist)]
    return crop(data, corner, size)
