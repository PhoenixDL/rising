import torch
from rising.utils.checktype import check_scalar
from math import pi
from typing import Union, Sequence

AffineParamType = Union[int, float, Sequence, torch.Tensor]


def points_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms points from cartesian to homogeneous coordinates

    Parameters
    ----------
    batch : torch.Tensor
        the batch of points to transform. Should be of shape
        BATCHSIZE x NUMPOINTS x DIM.

    Returns
    -------
    torch.Tensor
        the batch of points in homogeneous coordinates

    """
    return torch.cat([batch,
                      batch.new_ones((*batch.size()[:-1], 1))],
                     dim=-1)


def matrix_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms a given transformation matrix to a homogeneous
    transformation matrix.

    Parameters
    ----------
    batch : torch.Tensor
        the batch of matrices to convert

    Returns
    -------
    torch.Tensor
        the converted batch of matrices

    """
    if batch.size(-1) == batch.size(-2):
        missing = batch.new_zeros(size=(*batch.shape[:-1], 1))
        batch = torch.cat([batch, missing], dim=-1)

    missing = torch.zeros((batch.size(0),
                           *[1 for tmp in batch.shape[1:-1]],
                           batch.size(-1)),
                          device=batch.device, dtype=batch.dtype)

    missing[..., -1] = 1

    return torch.cat([batch, missing], dim=-2)


def matrix_to_cartesian(batch: torch.Tensor, keep_square: bool = False
                        ) -> torch.Tensor:
    """
    Transforms a matrix for a homogeneous transformation back to cartesian
    coordinates.

    Parameters
    ----------
    batch : torch.Tensor
        the batch oif matrices to convert back
    keep_square : bool
        if False: returns a NDIM x NDIM+1 matrix to keep the translation part
        if True: returns a NDIM x NDIM matrix but looses the translation part
        defaults to False.

    Returns
    -------
    torch.Tensor
        the given matrix in cartesian coordinates

    """
    batch = batch[:, :-1, ...]
    if keep_square:
        batch = batch[..., :-1]

    return batch


def points_to_cartesian(batch: torch.Tensor) -> torch.Tensor:
    """
    Transforms a batch of points in homogeneous coordinates back to cartesian
    coordinates.

    Parameters
    ----------
    batch : torch.Tensor
        batch of points in homogeneous coordinates. Should be of shape
        BATCHSIZE x NUMPOINTS x NDIM+1

    Returns
    -------
    torch.Tensor
        the batch of points in cartesian coordinates

    """

    return batch[..., :-1] / batch[..., -1, None]


def matrix_revert_coordinate_order(batch: torch.Tensor) -> torch.Tensor:
    """
    Reverts the coordinate order of a matrix (e.g. from xyz to zyx).

    Parameters
    ----------
    batch : torch.Tensor
        the batched transformation matrices; Should be of shape
        BATCHSIZE x NDIM x NDIM

    Returns
    -------
    torch.Tensor
        the matrix performing the same transformation on vectors with a
        reversed coordinate order

    """
    batch[:, :-1, :] = batch[:, :-1, :].flip(1).clone()
    batch[:, :-1, :-1] = batch[:, :-1, :-1].flip(2).clone()
    return batch


def get_batched_eye(batchsize: int, ndim: int,
                    device: Union[torch.device, str] = None,
                    dtype: Union[torch.dtype, str] = None) -> torch.Tensor:
    """
    Produces a batched matrix containing 1s on the diagonal

    Parameters
    ----------
    batchsize : int
        the batchsize (first dimension)
    ndim : int
        the dimensionality of the eyes (second and third dimension)
    device : torch.device, str, optional
        the device to put the resulting tensor to. Defaults to the default
        device
    dtype : torch.dtype, str, optional
        the dtype of the resulting trensor. Defaults to the default dtype

    Returns
    -------
    torch.Tensor
        batched eye matrix

    """
    return torch.eye(ndim, device=device, dtype=dtype).view(
        1, ndim, ndim).expand(batchsize, -1, -1).clone()


def _format_scale(scale: AffineParamType,
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


def _format_translation(offset: AffineParamType,
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


def deg_to_rad(angles: Union[torch.Tensor, float, int]
               ) -> Union[torch.Tensor, float, int]:
    """
    Converts from degree to radians.

    Parameters
    ----------
    angles : torch.Tensor, float, int
        the (vectorized) angles to convert

    Returns
    -------
    torch.Tensor, int, float
        the transformed (vectorized) angles

    """
    return angles * pi / 180


def _format_rotation(rotation: AffineParamType,
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
        rotation = rotation.view(batchsize, 1).expand(-1,
                                                      num_rot_params).clone()
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
    scale = _format_scale(scale, batchsize=batchsize, ndim=ndim,
                          device=device, dtype=dtype)
    rotation = _format_rotation(rotation, batchsize=batchsize, ndim=ndim,
                                degree=degree, device=device, dtype=dtype)

    translation = _format_translation(translation, batchsize=batchsize,
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
