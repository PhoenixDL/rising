import itertools
from math import pi
from typing import Optional, Union

import torch


def points_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms points from cartesian to homogeneous coordinates

    Args:
        batch: the batch of points to transform. Should be of shape
            BATCHSIZE x NUMPOINTS x DIM.

    Returns:
        torch.Tensor: the batch of points in homogeneous coordinates

    """
    return torch.cat([batch, batch.new_ones((*batch.size()[:-1], 1))], dim=-1)


def matrix_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms a given transformation matrix to a homogeneous
    transformation matrix.

    Args:
        batch: the batch of matrices to convert [N, dim, dim]

    Returns:
        torch.Tensor: the converted batch of matrices

    """
    if batch.size(-1) == batch.size(-2):
        missing = batch.new_zeros(size=(*batch.shape[:-1], 1))
        batch = torch.cat([batch, missing], dim=-1)

    missing = torch.zeros(
        (batch.size(0), *[1 for tmp in batch.shape[1:-1]], batch.size(-1)), device=batch.device, dtype=batch.dtype
    )

    missing[..., -1] = 1

    return torch.cat([batch, missing], dim=-2)


def matrix_to_cartesian(batch: torch.Tensor, keep_square: bool = False) -> torch.Tensor:
    """
    Transforms a matrix for a homogeneous transformation back to cartesian
    coordinates.

    Args:
        batch: the batch oif matrices to convert back
        keep_square: if False: returns a NDIM x NDIM+1 matrix to keep the
            translation part
            if True: returns a NDIM x NDIM matrix but looses the translation
            part. defaults to False.

    Returns:
        torch.Tensor: the given matrix in cartesian coordinates

    """
    batch = batch[:, :-1, ...]
    if keep_square:
        batch = batch[..., :-1]

    return batch


def points_to_cartesian(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms a batch of points in homogeneous coordinates back to cartesian
    coordinates.

    Args:
        batch: batch of points in homogeneous coordinates. Should be of shape
            BATCHSIZE x NUMPOINTS x NDIM+1

    Returns:
        torch.Tensor: the batch of points in cartesian coordinates

    """

    return batch[..., :-1] / batch[..., -1, None]


def matrix_revert_coordinate_order(batch: torch.Tensor) -> torch.Tensor:
    """
    Reverts the coordinate order of a matrix (e.g. from xyz to zyx).

    Args:
        batch: the batched transformation matrices; Should be of shape
            BATCHSIZE x NDIM x NDIM

    Returns:
        torch.Tensor: the matrix performing the same transformation on vectors with a
            reversed coordinate order
    """
    batch[:, :-1, :] = batch[:, :-1, :].flip(1).clone()
    batch[:, :-1, :-1] = batch[:, :-1, :-1].flip(2).clone()
    return batch


def get_batched_eye(
    batchsize: int,
    ndim: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[Union[torch.dtype, str]] = None,
) -> torch.Tensor:
    """
    Produces a batched matrix containing 1s on the diagonal

    Args:
        batchsize : int
            the batchsize (first dimension)
        ndim : int
            the dimensionality of the eyes (second and third dimension)
        device : torch.device, str, optional
            the device to put the resulting tensor to. Defaults to the default
            device
        dtype : torch.dtype, str, optional
            the dtype of the resulting trensor. Defaults to the default dtype

    Returns:
        torch.Tensor: batched eye matrix

    """
    return torch.eye(ndim, device=device, dtype=dtype).view(1, ndim, ndim).expand(batchsize, -1, -1).clone()


def deg_to_rad(angles: Union[torch.Tensor, float, int]) -> Union[torch.Tensor, float, int]:
    """
    Converts from degree to radians.

    Args:
        angles: the (vectorized) angles to convert

    Returns:
        torch.Tensor: the transformed (vectorized) angles

    """
    return angles * pi / 180


def unit_box(n: int, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Create a (scaled) version of a unit box

    Args:
        n: number of dimensions
        scale: scaling of each dimension

    Returns:
        torch.Tensor: scaled unit box
    """
    box = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=n)])
    if scale is not None:
        box = box.to(scale) * scale[None]
    return box
