import torch
import warnings
from typing import Union, Sequence

from rising.utils.affine import points_to_cartesian, matrix_to_homogeneous, \
    points_to_homogeneous, unit_box, get_batched_eye, deg_to_rad, matrix_to_cartesian
from rising.utils.checktype import check_scalar


__all__ = [
    'affine_image_transform',
    'affine_point_transform'
]


AffineParamType = Union[int, float, Sequence, torch.Tensor]


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
            * a full transformation matrix of shape (BATCHSIZE x NDIM x NDIM)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
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
        the homogeneous transformation matrix

    """

    if scale is None:
        scale = 1

    if check_scalar(scale):

        scale = get_batched_eye(batchsize=batchsize, ndim=ndim, device=device,
                                dtype=dtype) * scale

    elif not torch.is_tensor(scale):
        scale = torch.tensor(scale, dtype=dtype, device=device)

    # scale must be tensor by now
    scale = scale.to(device=device, dtype=dtype)

    # scale is already batched matrix
    if scale.size() == (batchsize, ndim, ndim) or scale.size() == (batchsize, ndim, ndim + 1):
        return matrix_to_homogeneous(scale)

    # scale is batched matrix with same element for each dimension or just
    # not diagonalized
    if scale.size() == (batchsize, ndim) or scale.size() == (batchsize,):
        new_scale = get_batched_eye(batchsize=batchsize, ndim=ndim,
                                    device=device, dtype=dtype)

        return matrix_to_homogeneous(new_scale * scale.view(batchsize, -1, 1))

    # scale contains a non-diagonalized form (will be repeated for each batch
    # item)
    elif scale.size() == (ndim,):
        return matrix_to_homogeneous(
            torch.diag(scale).view(1, ndim, ndim).expand(batchsize,
                                                         -1, -1).clone())

    # scale contains a diagonalized but not batched matrix
    # (will be repeated for each batch item)
    elif scale.size() == (ndim, ndim):
        return matrix_to_homogeneous(
            scale.view(1, ndim, ndim).expand(batchsize, -1, -1).clone())

    raise ValueError("Unknown shape for scale matrix: %s"
                     % str(tuple(scale.size())))


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
            * a full homogeneous transformation matrix of shape
                (BATCHSIZE x NDIM+1 x NDIM+1)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
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
        the homogeneous transformation matrix

    """
    if offset is None:
        offset = 0

    if check_scalar(offset):
        offset = torch.tensor([offset] * ndim, device=device, dtype=dtype)

    elif not torch.is_tensor(offset):
        offset = torch.tensor(offset, device=device, dtype=dtype)

    # assumes offset to be tensor from now on
    offset = offset.to(device=device, dtype=dtype)

    # translation matrix already built
    if offset.size() == (batchsize, ndim + 1, ndim + 1):
        return offset
    elif offset.size() == (batchsize, ndim, ndim + 1):
        return matrix_to_homogeneous(offset)

    # not completely built so far -> bring in shape (batchsize, ndim)
    if offset.size() == (batchsize,):
        offset = offset.view(-1, 1).expand(-1, ndim).clone()
    elif offset.size() == (ndim,):
        offset = offset.view(1, -1).expand(batchsize, -1).clone()
    elif not offset.size() == (batchsize, ndim):
        raise ValueError("Unknown shape for offsets: %s"
                         % str(tuple(offset.shape)))

    # directly build homogeneous form -> use dim+1
    whole_translation_matrix = get_batched_eye(batchsize=batchsize,
                                               ndim=ndim + 1, device=device,
                                               dtype=dtype)

    whole_translation_matrix[:, :-1, -1] = offset.clone()
    return whole_translation_matrix


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
            * a full transformation matrix of shape (BATCHSIZE x NDIM(+1) x NDIM(+1))
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
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
        the homogeneous transformation matrix

    """
    if rotation is None:
        rotation = 0

    num_rot_params = 1 if ndim == 2 else ndim

    if check_scalar(rotation):
        rotation = torch.ones(batchsize, num_rot_params,
                              device=device, dtype=dtype) * rotation
    elif not torch.is_tensor(rotation):
        rotation = torch.tensor(rotation, device=device, dtype=dtype)

    # assumes rotation to be tensor by now
    rotation = rotation.to(device=device, dtype=dtype)

    # already complete
    if rotation.size() == (batchsize, ndim, ndim) or rotation.size() == (batchsize, ndim, ndim + 1):
        return matrix_to_homogeneous(rotation)
    elif rotation.size() == (batchsize, ndim + 1, ndim + 1):
        return rotation

    if degree:
        rotation = deg_to_rad(rotation)

    # repeat along batch dimension
    if rotation.size() == (ndim, ndim) or rotation.size() == (ndim + 1, ndim + 1):
        rotation = rotation[None].expand(batchsize, -1, -1).clone()
        if rotation.size(-1) == ndim:
            rotation = matrix_to_homogeneous(rotation)

        return rotation
    # bring it to default size of (batchsize, num_rot_params)
    elif rotation.size() == (batchsize,):
        rotation = rotation.view(batchsize, 1).expand(-1, num_rot_params).clone()
    elif rotation.size() == (num_rot_params,):
        rotation = rotation.view(1, num_rot_params).expand(batchsize,
                                                           -1).clone()
    elif rotation.size() != (batchsize, num_rot_params):
        raise ValueError("Invalid shape for rotation parameters: %s"
                         % (str(tuple(rotation.size()))))

    sin, cos = rotation.sin(), rotation.cos()

    whole_rot_matrix = get_batched_eye(batchsize=batchsize, ndim=ndim,
                                       device=device, dtype=dtype)

    # assemble the actual matrix
    if num_rot_params == 1:
        whole_rot_matrix[:, 0, 0] = cos[0].clone()
        whole_rot_matrix[:, 1, 1] = cos[0].clone()
        whole_rot_matrix[:, 0, 1] = (-sin[0]).clone()
        whole_rot_matrix[:, 1, 0] = sin[0].clone()
    else:
        whole_rot_matrix[:, 0, 0] = (cos[:, 0] * cos[:, 1] * cos[:, 2]
                                     - sin[:, 0] * sin[:, 2]).clone()
        whole_rot_matrix[:, 0, 1] = (-cos[:, 0] * cos[:, 1] * sin[:, 2]
                                     - sin[:, 0] * cos[:, 2]).clone()
        whole_rot_matrix[:, 0, 2] = (cos[:, 0] * sin[:, 1]).clone()
        whole_rot_matrix[:, 1, 0] = (sin[:, 0] * cos[:, 1] * cos[:, 2]
                                     + cos[:, 0] * sin[:, 2]).clone()
        whole_rot_matrix[:, 1, 1] = (-sin[:, 0] * cos[:, 1] * sin[:, 2]
                                     + cos[:, 0] * cos[:, 2]).clone()
        whole_rot_matrix[:, 2, 0] = (-sin[:, 1] * cos[:, 2]).clone()
        whole_rot_matrix[:, 2, 1] = (-sin[:, 1] * sin[:, 2]).clone()
        whole_rot_matrix[:, 2, 2] = (cos[:, 1]).clone()

    return matrix_to_homogeneous(whole_rot_matrix)


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
            * a full transformation matrix of shape (BATCHSIZE x NDIM x NDIM)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
        None will be treated as a scaling factor of 1
    rotation : torch.Tensor, int, float
        the rotation factor(s). Supported are:
            * a full transformation matrix of shape (BATCHSIZE x NDIM x NDIM)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
        None will be treated as a rotation factor of 1
    translation : torch.Tensor, int, float
        the translation offset(s). Supported are:
            * a full homogeneous transformation matrix of shape
                (BATCHSIZE x NDIM+1 x NDIM+1)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
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
        the transformation matrix (of shape (BATCHSIZE x NDIM x NDIM+1)

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
            * a full transformation matrix of shape (BATCHSIZE x NDIM x NDIM)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
        None will be treated as a scaling factor of 1
    rotation : torch.Tensor, int, float
        the rotation factor(s). Supported are:
            * a full transformation matrix of shape (BATCHSIZE x NDIM x NDIM)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
        None will be treated as a rotation factor of 1
    translation : torch.Tensor, int, float
        the translation offset(s). Supported are:
            * a full homogeneous transformation matrix of shape
                (BATCHSIZE x NDIM+1 x NDIM+1)
            * a single parameter (as float or int), which will be replicated
                for all dimensions and batch samples
            * a single parameter per sample (as a 1d tensor), which will be
                replicated for all dimensions
            * a single parameter per dimension (either as 1d tensor or as
                2d transformation matrix), which will be replicated for all
                batch samples
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
        the assembled transformation matrix

    """
    if matrix is None:
        matrix = parametrize_matrix(scale=scale, rotation=rotation,
                                    translation=translation,
                                    batchsize=batchsize,
                                    ndim=ndim,
                                    degree=degree,
                                    device=device,
                                    dtype=dtype)

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
        a batch of affine matrices with shape N x NDIM x NDIM + 1
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
