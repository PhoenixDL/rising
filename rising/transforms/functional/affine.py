import torch
import warnings
from torch import Tensor
from typing import Union, Sequence

from rising.utils.affine import points_to_cartesian, matrix_to_homogeneous, \
    points_to_homogeneous, unit_box, get_batched_eye, deg_to_rad, matrix_to_cartesian
from rising.utils.checktype import check_scalar


__all__ = [
    'affine_image_transform',
    'affine_point_transform',
    "create_rotation",
    "create_scale",
    "create_translation",
]


AffineParamType = Union[int, float, Sequence, torch.Tensor]


def expand_scalar_param(param: AffineParamType, batchsize: int, ndim: int) -> Tensor:
    """
    Bring affine params to shape (batchsize, ndim)

    Parameters
    ----------
    param: AffineParamType
        affine parameter
    batchsize: int
        size of batch
    ndim: int
        number of spatial dimensions

    Returns
    -------
    Tensor:
        affine params in correct shape
    """
    if check_scalar(param):
        return torch.tensor([[param] * ndim] * batchsize).float()

    if not torch.is_tensor(param):
        param = torch.tensor(param)
    else:
        param = param.clone()

    if not param.ndimension() == 2:
        if param.shape[0] == ndim:  # scalar per dim
            param = param.reshape(1, -1).expand(batchsize, ndim)
        elif param.shape[0] == batchsize:  # scalar per batch
            param = param.reshape(-1, 1).expand(batchsize, ndim)
        else:
            raise ValueError("Unknown param for expanding. "
                             f"Found {param} for batchsize {batchsize} and ndim {ndim}")
    assert all([i == j for i, j in zip(param.shape, (batchsize, ndim))]), \
        (f"Affine param need to have shape (batchsize, ndim)"
         f"({(batchsize, ndim)}) but found {param.shape}")
    return param.float()


def create_scale(scale: AffineParamType,
                 batchsize: int, ndim: int,
                 device: Union[torch.device, str] = None,
                 dtype: Union[torch.dtype, str] = None) -> torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Parameters
    ----------
    scale : torch.Tensor, int, float
        the scale factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a scaling factor of 1
    batchsize : int
        the number of samples per batch
    ndim : int
        the dimensionality of the transform
    device : torch.device, str, optional
        the device to put the resulting tensor to. Defaults to the default
        device
    dtype : torch.dtype, str, optional
        the dtype of the resulting trensor. Defaults to the default dtype

    Returns
    -------
    torch.Tensor
        the homogeneous transformation matrix [N, NDIM + 1, NDIM + 1], N is
        the batch size and NDIM is the number of spatial dimensions
    """
    if scale is None:
        scale = 1

    scale = expand_scalar_param(scale, batchsize, ndim).to(
        device=device, dtype=dtype)
    scale_matrix = torch.stack(
        [eye * s for eye, s in zip(get_batched_eye(
            batchsize=batchsize, ndim=ndim, device=device, dtype=dtype), scale)])
    return matrix_to_homogeneous(scale_matrix)


def create_translation(offset: AffineParamType,
                       batchsize: int, ndim: int,
                       device: Union[torch.device, str] = None,
                       dtype: Union[torch.dtype, str] = None
                       ) -> torch.Tensor:
    """
    Formats the given translation parameters to a homogeneous transformation
    matrix

    Parameters
    ----------
    offset : torch.Tensor, int, float
        the translation offset(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a translation offset of 0
    batchsize : int
        the number of samples per batch
    ndim : int
        the dimensionality of the transform
    device : torch.device, str, optional
        the device to put the resulting tensor to. Defaults to the default
        device
    dtype : torch.dtype, str, optional
        the dtype of the resulting trensor. Defaults to the default dtype

    Returns
    -------
    torch.Tensor
        the homogeneous transformation matrix [N, NDIM + 1, NDIM + 1], N is
        the batch size and NDIM is the number of spatial dimensions

    """
    if offset is None:
        offset = 0
    offset = expand_scalar_param(offset, batchsize, ndim).to(
        device=device, dtype=dtype)
    eye_batch = get_batched_eye(batchsize=batchsize, ndim=ndim, device=device, dtype=dtype)
    translation_matrix = torch.stack([torch.cat([eye, o.view(-1, 1)], dim=1)
                                      for eye, o in zip(eye_batch, offset)])
    return matrix_to_homogeneous(translation_matrix)


def create_rotation(rotation: AffineParamType,
                    batchsize: int, ndim: int,
                    degree: bool = False,
                    device: Union[torch.device, str] = None,
                    dtype: Union[torch.dtype, str] = None) -> torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Parameters
    ----------
    rotation : torch.Tensor, int, float
        the rotation factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a rotation factor of 0
    batchsize : int
        the number of samples per batch
    ndim : int
        the dimensionality of the transform
    degree : bool
        whether the given rotation(s) are in degrees.
        Only valid for rotation parameters, which aren't passed as full
        transformation matrix.
    device : torch.device, str, optional
        the device to put the resulting tensor to. Defaults to the default
        device
    dtype : torch.dtype, str, optional
        the dtype of the resulting trensor. Defaults to the default dtype

    Returns
    -------
    torch.Tensor
        the homogeneous transformation matrix [N, NDIM + 1, NDIM + 1], N is
        the batch size and NDIM is the number of spatial dimensions

    """
    if rotation is None:
        rotation = 0
    num_rot_params = 1 if ndim == 2 else ndim

    rotation = expand_scalar_param(rotation, batchsize, num_rot_params)
    if degree:
        rotation = deg_to_rad(rotation)

    matrix_fn = create_rotation_2d if ndim == 2 else create_rotation_3d
    sin, cos = torch.sin(rotation), torch.cos(rotation)
    rotation_matrix = torch.stack([matrix_fn(s, c) for s, c in zip(sin, cos)]).to(
        device=device, dtype=dtype)
    return matrix_to_homogeneous(rotation_matrix)


def create_rotation_2d(sin: Tensor, cos: Tensor) -> Tensor:
    """
    Create a 2d rotation matrix

    Parameters
    ----------
    sin: Tensor
        sin value to use for rotation matrix, [1]
    cos: Tensor
        cos value to use for rotation matrix, [1]

    Returns
    -------
    Tensor
        rotation matrix, [2, 2]
    """
    return torch.tensor([[cos.clone(), -sin.clone()], [sin.clone(), cos.clone()]])


def create_rotation_3d(sin: Tensor, cos: Tensor) -> Tensor:
    """
    Create a 3d rotation matrix which sequentially applies the rotation
    around axis (rot axis 0 -> rot axis 1 -> rot axis 2)

    Parameters
    ----------
    sin: Tensor
        sin values to use for the rotation, (axis 0, axis 1, axis 2)[3]
    cos: Tensor
        cos values to use for the rotation, (axis 0, axis 1, axis 2)[3]

    Returns
    -------
    Tensor
        rotation matrix, [3, 3]
    """
    rot_0 = create_rotation_3d_0(sin[0], cos[0])
    rot_1 = create_rotation_3d_1(sin[1], cos[1])
    rot_2 = create_rotation_3d_2(sin[2], cos[2])
    return rot_2 @ (rot_1 @ rot_0)


def create_rotation_3d_0(sin: Tensor, cos: Tensor) -> Tensor:
    """
    Create a rotation matrix around the zero-th axis

    Parameters
    ----------
    sin: Tensor
        sin value to use for rotation matrix, [1]
    cos: Tensor
        cos value to use for rotation matrix, [1]

    Returns
    -------
    Tensor:
        rotation matrix, [3, 3]
    """
    return torch.tensor([[1., 0., 0.],
                         [0., cos.clone(), -sin.clone()],
                         [0., sin.clone(), cos.clone()]])


def create_rotation_3d_1(sin: Tensor, cos: Tensor) -> Tensor:
    """
    Create a rotation matrix around the first axis

    Parameters
    ----------
    sin: Tensor
        sin value to use for rotation matrix, [1]
    cos: Tensor
        cos value to use for rotation matrix, [1]

    Returns
    -------
    Tensor:
        rotation matrix, [3, 3]
    """
    return torch.tensor([[cos.clone(), 0., sin.clone()],
                         [0., 1., 0.],
                         [-sin.clone(), 0., cos.clone()]])


def create_rotation_3d_2(sin: Tensor, cos: Tensor) -> Tensor:
    """
    Create a rotation matrix around the second axis

    Parameters
    ----------
    sin: Tensor
        sin value to use for rotation matrix, [1]
    cos: Tensor
        cos value to use for rotation matrix, [1]

    Returns
    -------
    Tensor:
        rotation matrix, [3, 3]
    """
    return torch.tensor([[cos.clone(), -sin.clone(), 0.],
                         [sin.clone(), cos.clone(), 0.],
                         [0., 0., 1.]])


def parametrize_matrix(scale: AffineParamType,
                       rotation: AffineParamType,
                       translation: AffineParamType,
                       batchsize: int, ndim: int,
                       degree: bool = False,
                       device: Union[torch.device, str] = None,
                       dtype: Union[torch.dtype, str] = None) -> torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Parameters
    ----------
    scale : torch.Tensor, int, float
        the scale factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a scaling factor of 1
    rotation : torch.Tensor, int, float
        the rotation factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a rotation factor of 1
    translation : torch.Tensor, int, float
        the translation offset(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a translation offset of 0
    batchsize : int
        the number of samples per batch
    ndim : int
        the dimensionality of the transform
    degree : bool
        whether the given rotation(s) are in degrees.
        Only valid for rotation parameters, which aren't passed as full
        transformation matrix.
    device : torch.device, str, optional
        the device to put the resulting tensor to. Defaults to the default
        device
    dtype : torch.dtype, str, optional
        the dtype of the resulting trensor. Defaults to the default dtype

    Returns
    -------
    torch.Tensor
        the transformation matrix [N, NDIM, NDIM+1], N is
        the batch size and NDIM is the number of spatial dimensions
    """
    scale = create_scale(scale, batchsize=batchsize, ndim=ndim,
                         device=device, dtype=dtype)
    rotation = create_rotation(rotation, batchsize=batchsize, ndim=ndim,
                               degree=degree, device=device, dtype=dtype)

    translation = create_translation(translation, batchsize=batchsize,
                                     ndim=ndim, device=device, dtype=dtype)
    return torch.bmm(torch.bmm(scale, rotation), translation)[:, :-1]


def assemble_matrix_if_necessary(batchsize: int, ndim: int,
                                 scale: AffineParamType,
                                 rotation: AffineParamType,
                                 translation: AffineParamType,
                                 matrix: torch.Tensor,
                                 degree: bool,
                                 device: Union[torch.device, str],
                                 dtype: Union[torch.dtype, str]
                                 ) -> torch.Tensor:
    """
    Assembles a matrix, if the matrix is not already given

    Parameters
    ----------
    batchsize : int
        number of samples per batch
    ndim : int
        the image dimensionality
    scale : torch.Tensor, int, float
        the scale factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a scaling factor of 1
    rotation : torch.Tensor, int, float
        the rotation factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a rotation factor of 1
    translation : torch.Tensor, int, float
        the translation offset(s). Supported are:
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a parameter per sample, which will be
                replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
                batch samples
            * a parameter per sampler per dimension
        None will be treated as a translation offset of 0
    matrix : torch.Tensor
        the transformation matrix. If other than None: overwrites separate
        parameters for :param:`scale`, :param:`rotation` and
        :param:`translation`
    degree : bool
        whether the given rotation is in degrees. Only valid for explicit
        rotation parameters
    device : str, torch.device
        the device, the matrix should be put on
    dtype : str, torch.dtype
        the datatype, the matrix should have

    Returns
    -------
    torch.Tensor
        the assembled transformation matrix [N, NDIM, NDIM+1], N is
        the batch size and NDIM is the number of spatial dimensions

    """
    if matrix is None:
        matrix = parametrize_matrix(
            scale=scale, rotation=rotation, translation=translation,
            batchsize=batchsize, ndim=ndim, degree=degree,
            device=device, dtype=dtype)
    else:
        if not torch.is_tensor(matrix):
            matrix = torch.tensor(matrix)
        matrix = matrix.to(dtype=dtype, device=device)

    # batch dimension missing -> Replicate for each sample in batch
    if len(matrix.shape) == 2:
        matrix = matrix[None].expand(batchsize, -1, -1).clone()
    if matrix.shape == (batchsize, ndim, ndim + 1):
        return matrix
    elif matrix.shape == (batchsize, ndim + 1, ndim + 1):
        return matrix_to_cartesian(matrix)

    raise ValueError(
        "Invalid Shape for affine transformation matrix. "
        "Got %s but expected %s" % (
            str(tuple(matrix.shape)),
            str((batchsize, ndim, ndim + 1))))


def affine_point_transform(point_batch: torch.Tensor,
                           matrix_batch: torch.Tensor) -> torch.Tensor:
    """
    Function to perform an affine transformation onto point batches

    Parameters
    ----------
    point_batch : torch.Tensor
        a point batch of shape [N, NP, NDIM] NP is the number of points,
            N is the batch size, NDIM is the number of spatial dimensions
    matrix_batch : torch.Tensor
        a batch of affine matrices with shape [N, NDIM, NDIM + 1],
            N is the batch size and NDIM is the number of spatial dimensions

    Returns
    -------
    torch.Tensor
        the batch of transformed points in cartesian coordinates)
        [N, NP, NDIM] NP is the number of points, N is the batch size,
        NDIM is the number of spatial dimensions
    """
    point_batch = points_to_homogeneous(point_batch)
    matrix_batch = matrix_to_homogeneous(matrix_batch)
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
        the batch to transform. Should have shape of [N, C, NDIM]
    matrix_batch : torch.Tensor
        a batch of affine matrices with shape [N, NDIM, NDIM+1]
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
        matrix_batch = matrix_batch[None, ...].expand(
            image_batch.size(0), -1, -1).clone()

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

    matrix_batch = matrix_batch.to(image_batch)

    grid = torch.nn.functional.affine_grid(matrix_batch, size=new_size,
                                           align_corners=align_corners)

    return torch.nn.functional.grid_sample(image_batch, grid,
                                           mode=interpolation_mode,
                                           padding_mode=padding_mode,
                                           align_corners=align_corners)


def _check_new_img_size(curr_img_size, matrix: torch.Tensor,
                        zero_border: bool = False) -> torch.Tensor:
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
        a batch of affine matrices with shape [N, NDIM, NDIM+1]
    zero_border : bool
        whether or not to have a fixed image border at zero

    Returns
    -------
    torch.Tensor
        the new image size
    """
    n_dim = matrix.size(-1) - 1
    if check_scalar(curr_img_size):
        curr_img_size = [curr_img_size] * n_dim
    possible_points = unit_box(n_dim, torch.tensor(curr_img_size)).to(matrix)

    transformed_edges = affine_point_transform(
        possible_points[None].expand(
            matrix.size(0), *[-1 for _ in possible_points.shape]).clone(),
        matrix)

    if zero_border:
        substr = 0
    else:
        substr = transformed_edges.min(1)[0]

    return (transformed_edges.max(1)[0] - substr).max(0)[0]
