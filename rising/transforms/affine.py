import torch
from torch import Tensor
from typing import Sequence, Union, Iterable, Dict, Tuple

from rising.transforms.grid import GridTransform
from rising.transforms.functional.affine import create_affine_grid, \
    AffineParamType, parametrize_matrix
from rising.utils.affine import matrix_to_homogeneous, matrix_to_cartesian
from rising.utils.checktype import check_scalar


__all__ = [
    'Affine',
    'StackedAffine',
    'Rotate',
    'Scale',
    'Translate',
    'Resize',
]


class Affine(GridTransform):
    def __init__(self,
                 matrix: Union[Tensor, Sequence[Sequence[float]]] = None,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs):
        """
        Class Performing an Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        matrix : Tensor, optional
            if given, overwrites the parameters for :param:`scale`,
            :param:rotation` and :param:`translation`.
            Should be a matrix of shape [(BATCHSIZE,) NDIM, NDIM(+1)]
            This matrix represents the whole transformation matrix
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
            additional keyword arguments passed to grid sample
        """
        super().__init__(keys=keys, grad=grad, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         **kwargs)
        self.matrix = matrix
        self.output_size = output_size
        self.adjust_size = adjust_size

    def assemble_matrix(self,
                        batch_shape: Sequence[int],
                        device: Union[torch.device, str] = None,
                        dtype: Union[torch.dtype, str] = None,
                        ) -> Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Parameters
        ----------
        batch_shape : Sequence[int]
            shape of batch
        device: Union[torch.device, str]
            device where grid will be cached
        dtype: Union[torch.dtype, str]
            data type of grid

        Returns
        -------
        Tensor
            the (batched) transformation matrix
        """
        if self.matrix is None:
            raise ValueError("Matrix needs to be initialized or overwritten.")
        if not torch.is_tensor(self.matrix):
            self.matrix = Tensor(self.matrix)
        self.matrix = self.matrix.to(device=device, dtype=dtype)

        batchsize = batch_shape[0]
        ndim = len(batch_shape) - 2  # channel and batch dim

        # batch dimension missing -> Replicate for each sample in batch
        if len(self.matrix.shape) == 2:
            self.matrix = self.matrix[None].expand(batchsize, -1, -1).clone()
        if self.matrix.shape == (batchsize, ndim, ndim + 1):
            return self.matrix
        elif self.matrix.shape == (batchsize, ndim, ndim):
            return matrix_to_homogeneous(self.matrix)[:, :-1]
        elif self.matrix.shape == (batchsize, ndim + 1, ndim + 1):
            return matrix_to_cartesian(self.matrix)

        raise ValueError(
            "Invalid Shape for affine transformation matrix. "
            "Got %s but expected %s" % (
                str(tuple(self.matrix.shape)),
                str((batchsize, ndim, ndim + 1))))

    def create_grid(self, input_size: Sequence[Sequence[int]],
                    matrix: Tensor = None) -> Dict[Tuple, Tensor]:
        grid = {}
        for size in input_size:
            if tuple(size) not in grid:
                grid[tuple(size)] = create_affine_grid(
                    size, self.assemble_matrix(size), output_size=self.output_size,
                    adjust_size=self.adjust_size, align_corners=self.align_corners,
                    )
        return grid

    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:
        return grid

    def __add__(self, other):
        """
        Makes ``trafo + other_trafo work``
        (stacks them for dynamic assembling)

        Parameters
        ----------
        other : Tensor, Affine
            the other transformation

        Returns
        -------
        StackedAffine
            a stacked affine transformation
        """
        if not isinstance(other, GridTransform):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad,
                           output_size=self.output_size,
                           adjust_size=self.adjust_size,
                           interpolation_mode=self.interpolation_mode,
                           padding_mode=self.padding_mode,
                           align_corners=self.align_corners,
                           **self.kwargs)

        if isinstance(other, Affine):
            return StackedAffine(self, other, keys=self.keys, grad=self.grad,
                                 output_size=self.output_size,
                                 adjust_size=self.adjust_size,
                                 interpolation_mode=self.interpolation_mode,
                                 padding_mode=self.padding_mode,
                                 align_corners=self.align_corners, **self.kwargs)
        else:
            return super().__add__(other)

    def __radd__(self, other):
        """
        Makes ``other_trafo + trafo`` work
        (stacks them for dynamic assembling)

        Parameters
        ----------
        other : Tensor, Affine
            the other transformation

        Returns
        -------
        StackedAffine
            a stacked affine transformation
        """
        if not isinstance(other, GridTransform):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad,
                           output_size=self.output_size,
                           adjust_size=self.adjust_size,
                           interpolation_mode=self.interpolation_mode,
                           padding_mode=self.padding_mode,
                           align_corners=self.align_corners, **self.kwargs)

        if isinstance(other, Affine):
            return StackedAffine(other, self, grad=other.grad,
                                 output_size=other.output_size,
                                 adjust_size=other.adjust_size,
                                 interpolation_mode=other.interpolation_mode,
                                 padding_mode=other.padding_mode,
                                 align_corners=other.align_corners,
                                 **other.kwargs)
        else:
            return super().__add__(other)


class StackedAffine(Affine):
    def __init__(
            self,
            *transforms: Union[Affine, Sequence[
                Union[Sequence[Affine], Affine]]],
            keys: Sequence = ('data',),
            grad: bool = False,
            output_size: tuple = None,
            adjust_size: bool = False,
            interpolation_mode: str = 'bilinear',
            padding_mode: str = 'zeros',
            align_corners: bool = False,
            **kwargs):
        """
        Class Performing an Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        transforms : sequence of Affines
            the transforms to stack. Each transform must have a function
            called ``assemble_matrix``, which is called to dynamically
            assemble stacked matrices. Afterwards these transformations are
            stacked by matrix-multiplication to only perform a single
            interpolation
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
            additional keyword arguments passed to the affine transform
        """
        if isinstance(transforms, (tuple, list)):
            if isinstance(transforms[0], (tuple, list)):
                transforms = transforms[0]

        # ensure trafos are Affines and not raw matrices
        transforms = tuple(
            [trafo if isinstance(trafo, Affine) else Affine(matrix=trafo)
             for trafo in transforms])

        super().__init__(keys=keys, grad=grad,
                         output_size=output_size, adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)

        self.transforms = transforms

    def assemble_matrix(self,
                        batch_shape: Sequence[int],
                        device: Union[torch.device, str] = None,
                        dtype: Union[torch.dtype, str] = None,
                        ) -> Tensor:
        """
        Handles the matrix assembly and stacking

        Parameters
        ----------
        batch_shape : Sequence[int]
            shape of batch
        device: Union[torch.device, str]
            device where grid will be cached
        dtype: Union[torch.dtype, str]
            data type of grid

        Returns
        -------
        Tensor
            the (batched) transformation matrix

        """
        whole_trafo = None
        for trafo in self.transforms:
            matrix = matrix_to_homogeneous(trafo.assemble_matrix(
                batch_shape=batch_shape, device=device, dtype=dtype
            ))
            if whole_trafo is None:
                whole_trafo = matrix
            else:
                whole_trafo = torch.bmm(whole_trafo, matrix)
        return matrix_to_cartesian(whole_trafo)


class BaseAffine(Affine):
    def __init__(self,
                 scale: AffineParamType = None,
                 rotation: AffineParamType = None,
                 translation: AffineParamType = None,
                 degree: bool = False,
                 image_transform: bool = True,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs,
                 ):
        """
        Class performing a basic Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        scale : Tensor, int, float, optional
            the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a scaling factor of 1
        rotation : Tensor, int, float, optional
            the rotation factor(s). The rotation is performed in
             consecutive order axis0 -> axis1 (-> axis 2). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a rotation factor of 1
        translation : Tensor, int, float
            the translation offset(s) relative to image (should be in the
            range [0, 1]). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a translation offset of 0
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        degree : bool
            whether the given rotation(s) are in degrees.
            Only valid for rotation parameters, which aren't passed as full
            transformation matrix.
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
            additional keyword arguments passed to the affine transform
        """
        super().__init__(keys=keys, grad=grad, output_size=output_size,
                         adjust_size=adjust_size, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         **kwargs)
        self.scale = scale
        self.rotation = rotation
        self.translation = translation
        self.degree = degree
        self.image_transform = image_transform

    def assemble_matrix(self,
                        batch_shape: Sequence[int],
                        device: Union[torch.device, str] = None,
                        dtype: Union[torch.dtype, str] = None,
                        ) -> Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Parameters
        ----------
        batch_shape : Sequence[int]
            shape of batch
        device: Union[torch.device, str]
            device where grid will be cached
        dtype: Union[torch.dtype, str]
            data type of grid

        Returns
        -------
        Tensor
            the (batched) transformation matrix
        """
        batchsize = batch_shape[0]
        ndim = len(batch_shape) - 2  # channel and batch dim

        self.matrix = parametrize_matrix(
            scale=self.scale, rotation=self.rotation, translation=self.translation,
            batchsize=batchsize, ndim=ndim, degree=self.degree,
            device=device, dtype=dtype, image_transform=self.image_transform)
        return self.matrix


class Rotate(BaseAffine):
    def __init__(self,
                 rotation: AffineParamType,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 degree: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs):
        """
        Class Performing a Rotation-OnlyAffine Transformation on a given
        sample dict. The rotation is applied in consecutive order:
        rot axis 0 -> rot axis 1 -> rot axis 2
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        rotation : Tensor, int, float, optional
            the rotation factor(s). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a rotation factor of 1
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        degree : bool
            whether the given rotation(s) are in degrees.
            Only valid for rotation parameters, which aren't passed as full
            transformation matrix.
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
                    additional keyword arguments passed to the affine transform
        """
        super().__init__(scale=None,
                         rotation=rotation,
                         translation=None,
                         matrix=None,
                         keys=keys,
                         grad=grad,
                         degree=degree,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)


class Translate(BaseAffine):
    def __init__(self,
                 translation: AffineParamType,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 unit: str = 'relative',
                 **kwargs):
        """
        Class Performing an Translation-Only
        Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        translation : Tensor, int, float
            the translation offset(s). The translation unit can be specified.
            Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a translation offset of 0
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode : str
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        unit: str
            defines the unit of the translation parameter.
            'pixel': define number of pixels to translate | 'relative':
            translation should be in the range [0, 1] and is scaled
            with the image size
        **kwargs :
            additional keyword arguments passed to the affine transform
        """
        super().__init__(scale=None,
                         rotation=None,
                         translation=translation,
                         matrix=None,
                         keys=keys,
                         grad=grad,
                         degree=False,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)
        self.unit = unit

    def assemble_matrix(self,
                        batch_shape: Sequence[int],
                        device: Union[torch.device, str] = None,
                        dtype: Union[torch.dtype, str] = None,
                        ) -> Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Parameters
        ----------
        batch_shape : Sequence[int]
            shape of batch
        device: Union[torch.device, str]
            device where grid will be cached
        dtype: Union[torch.dtype, str]
            data type of grid

        Returns
        -------
        Tensor
            the (batched) transformation matrix [N, NDIM, NDIM]
        """
        matrix = super().assemble_matrix(batch_shape=batch_shape,
                                         device=device, dtype=dtype)
        if self.unit.lower() == 'pixel':
            img_size = torch.tensor(batch_shape[2:]).to(matrix)
            matrix[..., -1] = matrix[..., -1] / img_size
        return matrix


class Scale(BaseAffine):
    def __init__(self,
                 scale: AffineParamType,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs):
        """
        Class Performing a Scale-Only Affine Transformation on a given
        sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        scale : Tensor, int, float, optional
            the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a scaling factor of 1
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        degree : bool
            whether the given rotation(s) are in degrees.
            Only valid for rotation parameters, which aren't passed as full
            transformation matrix.
        output_size : Iterable
            if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size : bool
            if True, the resulting image size will be calculated dynamically
            to ensure that the whole image fits.
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
            additional keyword arguments passed to the affine transform

        Warnings
        --------
        This transform is not applied around the image center
        """
        super().__init__(scale=scale,
                         rotation=None,
                         translation=None,
                         matrix=None,
                         keys=keys,
                         grad=grad,
                         degree=False,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)


class Resize(Scale):
    def __init__(self,
                 size: Union[int, Iterable],
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs):
        """
        Class Performing a Resizing Affine Transformation on a given
        sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        size : int, Iterable
            the target size. If int, this will be repeated for all the
            dimensions
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        interpolation_mode : str
            interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode :
            padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners : Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        **kwargs :
            additional keyword arguments passed to the affine transform

        Note
        ----
        The offsets for shifting back and to origin are calculated on the
        entry matching the first item iin :attr:`keys` for each batch

        Note
        ----
        The target size must be specified in x, y (,z) order and will be
        converted to (D,) H, W order internally

        """
        super().__init__(output_size=size,
                         scale=None,
                         keys=keys,
                         grad=grad,
                         adjust_size=False,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)

    def assemble_matrix(self,
                        batch_shape: Sequence[int],
                        device: Union[torch.device, str] = None,
                        dtype: Union[torch.dtype, str] = None,
                        ) -> Tensor:

        """
        Handles the matrix assembly and calculates the scale factors for
        resizing

        Parameters
        ----------
        batch_shape : Sequence[int]
            shape of batch
        device: Union[torch.device, str]
            device where grid will be cached
        dtype: Union[torch.dtype, str]
            data type of grid

        Returns
        -------
        Tensor
            the (batched) transformation matrix

        """
        curr_img_size = batch_shape[2:]

        was_scalar = check_scalar(self.output_size)

        if was_scalar:
            self.output_size = [self.output_size] * len(curr_img_size)

        self.scale = [self.output_size[i] / curr_img_size[-i]
                      for i in range(len(curr_img_size))]

        matrix = super().assemble_matrix(batch_shape=batch_shape,
                                         device=device, dtype=dtype)

        if was_scalar:
            self.output_size = self.output_size[0]

        return matrix
