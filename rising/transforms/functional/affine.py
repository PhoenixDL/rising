import torch
from rising.utils.affine import points_to_cartesian, matrix_to_homogeneous, \
    points_to_homogeneous, matrix_revert_coordinate_order
from rising.utils.checktype import check_scalar
import warnings

__all__ = [
    'affine_image_transform',
    'affine_point_transform'
]


def affine_point_transform(point_batch: torch.Tensor,
                           matrix_batch: torch.Tensor) -> torch.Tensor:
    """
    Function to perform an affine transformation onto point batches

    Parameters
    ----------
    point_batch : torch.Tensor
        a point batch of shape BATCHSIZE x NUM_POINTS x NDIM
    matrix_batch : torch.Tensor
        a batch of affine matrices with shape N x NDIM-1 x NDIM

    Returns
    -------
    torch.Tensor
        the batch of transformed points in cartesian coordinates)

    """
    point_batch = points_to_homogeneous(point_batch)
    matrix_batch = matrix_to_homogeneous(matrix_batch)

    matrix_batch = matrix_revert_coordinate_order(matrix_batch)

    transformed_points = torch.bmm(point_batch,
                                   matrix_batch.permute(0, 2, 1))

    return points_to_cartesian(transformed_points)


def affine_image_transform(image_batch: torch.Tensor,
                           matrix_batch: torch.Tensor,
                           output_size: tuple = None,
                           adjust_size: bool = False,
                           interpolation_mode: str = 'bilinear',
                           padding_mode: str = 'zeros',
                           align_corners: bool = False) -> torch.Tensor:
    """
    Performs an affine transformation on a batch of images

    Parameters
    ----------
    image_batch : torch.Tensor
        the batch to transform. Should have shape of N x C x (D x) H x W
    matrix_batch : torch.Tensor
        a batch of affine matrices with shape N x NDIM-1 x NDIM
    output_size : Iterable
        if given, this will be the resulting image size. Defaults to ``None``
    adjust_size : bool
        if True, the resulting image size will be calculated dynamically to
        ensure that the whole image fits.
    interpolation_mode : str
        interpolation mode to calculate output values 'bilinear' | 'nearest'.
        Default: 'bilinear'
    padding_mode :
        padding mode for outside grid values
        'zeros' | 'border' | 'reflection'. Default: 'zeros'
    align_corners : Geometrically, we consider the pixels of the input as
        squares rather than points. If set to True, the extrema (-1 and 1) are
        considered as referring to the center points of the input’s corner
        pixels. If set to False, they are instead considered as referring to
        the corner points of the input’s corner pixels, making the sampling
        more resolution agnostic.

    Returns
    -------
    torch.Tensor
        transformed image

    Warnings
    --------
    When align_corners = True, the grid positions depend on the pixel size
    relative to the input image size, and so the locations sampled by
    grid_sample() will differ for the same input given at different
    resolutions (that is, after being upsampled or downsampled).

    Notes
    -----
    :param:`output_size` and :param:`adjust_size` are mutually exclusive.
    If None of them is set, the resulting image will have the same size as the
    input image
    """

    # add batch dimension if necessary
    if len(matrix_batch.shape) < 3:
        matrix_batch = matrix_batch[None, ...].expand(image_batch.size(0),
                                                      -1, -1).clone()

    image_size = image_batch.shape[2:]

    if output_size is not None:
        if check_scalar(output_size):
            output_size = tuple([output_size] * matrix_batch.size(-2))

        if adjust_size:
            warnings.warn("Adjust size is mutually exclusive with a "
                          "given output size.", UserWarning)

        new_size = output_size

    elif adjust_size:
        new_size = tuple([int(tmp.item())
                          for tmp in _check_new_img_size(image_size,
                                                         matrix_batch)])
    else:
        new_size = image_size

    if len(image_size) < len(image_batch.shape):
        missing_dims = len(image_batch.shape) - len(image_size)
        new_size = (*image_batch.shape[:missing_dims], *new_size)

    matrix_batch = matrix_batch.to(device=image_batch.device,
                                   dtype=image_batch.dtype)

    grid = torch.nn.functional.affine_grid(matrix_batch, size=new_size,
                                           align_corners=align_corners)

    return torch.nn.functional.grid_sample(image_batch, grid,
                                           mode=interpolation_mode,
                                           padding_mode=padding_mode,
                                           align_corners=align_corners)


def _check_new_img_size(curr_img_size, matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the image size so that the whole image content fits the image.
    The resulting size will be the maximum size of the batch, so that the
    images can remain batched.

    Parameters
    ----------
    curr_img_size : int or Iterable
        the size of the current image. If int, it will be used as size for
        all image dimensions
    matrix : torch.Tensor
        a batch of affine matrices with shape N x NDIM x NDIM + 1

    Returns
    -------
    torch.Tensor
        the new image size

    """

    n_dim = matrix.size(-1) - 1

    if check_scalar(curr_img_size):
        curr_img_size = [curr_img_size] * n_dim

    curr_img_size = [tmp - 1 for tmp in curr_img_size]

    if n_dim == 2:
        possible_points = torch.tensor([[0., 0.], [0., curr_img_size[1]],
                                        [curr_img_size[0], 0], curr_img_size],
                                       dtype=matrix.dtype,
                                       device=matrix.device)
    elif n_dim == 3:
        possible_points = torch.tensor(
            [
                [0., 0., 0.],
                [0., 0., curr_img_size[2]],
                [0., curr_img_size[1], 0],
                [0., curr_img_size[1], curr_img_size[2]],
                [curr_img_size[0], 0., 0.],
                [curr_img_size[0], 0., curr_img_size[2]],
                [curr_img_size[0], curr_img_size[1], 0.],
                curr_img_size
            ], device=matrix.device, dtype=matrix.dtype
        )

    else:
        raise ValueError('Invalid number of dimensions! Expected One of '
                         '{2, 3}, but got %s' % str(n_dim))

    transformed_edges = affine_point_transform(
        possible_points[None].expand(
            matrix.size(0),
            *[-1 for _ in possible_points.shape]).clone(),
        matrix)

    return (transformed_edges.max(1)[0]
            - transformed_edges.min(1)[0]).max(0)[0] + 1
