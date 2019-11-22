import torch
from typing import Sequence, Union

from phdata.utils import check_scalar


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

    if 0 in dims:
        data = data[:, :, ::-1]
    if 1 in dims:
        data = data[:, :, :, ::-1]
    if 2 in dims:
        data = data[:, :, :, :, ::-1]
    return data


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


def rot():
    # TODO: Implement rot transformation
    raise NotImplementedError
