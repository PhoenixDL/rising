from rising.transforms.abstract import BaseTransform
from rising.transforms.functional.affine import affine_image_transform
from rising.utils.affine import AffineParamType, \
    assemble_matrix_if_necessary, matrix_to_homogeneous, matrix_to_cartesian
import torch
from typing import Sequence, Union

__all__ = [
    'Affine',
    'StackedAffine',
    'Rotate',
    'Scale',
    'Translate'
]


class Affine(BaseTransform):
    def __init__(self, scale: AffineParamType = None,
                 rotation: AffineParamType = None,
                 translation: AffineParamType = None,
                 matrix: torch.Tensor = None,
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
        Class Performing an Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        scale : torch.Tensor, int, float, optional
            the scale factor(s). Supported are:
                * a full transformation matrix of shape
                    (BATCHSIZE x NDIM x NDIM)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
            None will be treated as a scaling factor of 1
        rotation : torch.Tensor, int, float, optional
            the rotation factor(s). Supported are:
                * a full transformation matrix of shape
                    (BATCHSIZE x NDIM x NDIM)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
            None will be treated as a rotation factor of 1
        translation : torch.Tensor, int, float
            the translation offset(s). Supported are:
                * a full homogeneous transformation matrix of shape
                    (BATCHSIZE x NDIM+1 x NDIM+1)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
            None will be treated as a translation offset of 0
        matrix : torch.Tensor, optional
            if given, overwrites the parameters for :param:`scale`,
            :param:rotation` and :param:`translation`.
            Should be a matrix o shape (BATCHSIZE,) NDIM, NDIM+1.
            This matrix represents the whole homogeneous transformation matrix
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

        Notes
        -----
        If a :param:`matrix` is specified, it overwrites  all arguments given
        for :param:`scale`, :param:rotation` and :param:`translation`
        """
        super().__init__(augment_fn=affine_image_transform,
                         keys=keys,
                         grad=grad,
                         **kwargs)
        self.scale = scale
        self.rotation = rotation
        self.translation = translation
        self.matrix = matrix
        self.degree = degree
        self.output_size = output_size
        self.adjust_size = adjust_size
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Parameters
        ----------
        **data :
            the data to be transformed. Will be used to determine batchsize,
            dimensionality, dtype and device

        Returns
        -------
        torch.Tensor
            the (batched) transformation matrix

        """

        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2  # channel and batch dim
        device = data[self.keys[0]].device
        dtype = data[self.keys[0]].dtype

        matrix = assemble_matrix_if_necessary(
            batchsize, ndim, scale=self.scale, rotation=self.rotation,
            translation=self.translation, matrix=self.matrix,
            degree=self.degree, device=device, dtype=dtype)

        return matrix

    def forward(self, **data) -> dict:
        """
        Assembles the matrix and applies it to the specified sample-entities.

        Parameters
        ----------
        **data :
            the data to transform

        Returns
        -------
        dict
            dictionary containing the transformed data

        """
        matrix = self.assemble_matrix(**data)

        for key in self.keys:
            data[key] = self.augment_fn(
                data[key], matrix_batch=matrix,
                output_size=self.output_size,
                adjust_size=self.adjust_size,
                interpolation_mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                **self.kwargs
            )

        return data

    def __add__(self, other):
        """
        Makes ``trafo + other_trafo work``
        (stacks them for dynamic assembling)

        Parameters
        ----------
        other : torch.Tensor, Affine
            the other transformation

        Returns
        -------
        StackedAffine
            a stacked affine transformation

        """
        if not isinstance(other, Affine):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad,
                           output_size=self.output_size,
                           adjust_size=self.adjust_size,
                           interpolation_mode=self.interpolation_mode,
                           padding_mode=self.padding_mode,
                           align_corners=self.align_corners,
                           **self.kwargs)

        return StackedAffine(self, other, keys=self.keys, grad=self.grad,
                             output_size=self.output_size,
                             adjust_size=self.adjust_size,
                             interpolation_mode=self.interpolation_mode,
                             padding_mode=self.padding_mode,
                             align_corners=self.align_corners, **self.kwargs)

    def __radd__(self, other):
        """
        Makes ``other_trafo + trafo`` work
        (stacks them for dynamic assembling)

        Parameters
        ----------
        other : torch.Tensor, Affine
            the other transformation

        Returns
        -------
        StackedAffine
            a stacked affine transformation

        """
        if not isinstance(other, Affine):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad,
                           output_size=self.output_size,
                           adjust_size=self.adjust_size,
                           interpolation_mode=self.interpolation_mode,
                           padding_mode=self.padding_mode,
                           align_corners=self.align_corners, **self.kwargs)

        return StackedAffine(other, self, grad=other.grad,
                             output_size=other.output_size,
                             adjust_size=other.adjust_size,
                             interpolation_mode=other.interpolation_mode,
                             padding_mode=other.padding_mode,
                             align_corners=other.align_corners,
                             **other.kwargs)


class Rotate(Affine):
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
        sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        rotation : torch.Tensor, int, float, optional
            the rotation factor(s). Supported are:
                * a full transformation matrix of shape
                    (BATCHSIZE x NDIM x NDIM)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
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

        Warnings
        --------
        This transform is not applied around the image center

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


class Translate(Affine):
    def __init__(self,
                 translation: AffineParamType,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 **kwargs):
        """
        Class Performing an Translation-Only
        Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        translation : torch.Tensor, int, float
            the translation offset(s). Supported are:
                * a full homogeneous transformation matrix of shape
                    (BATCHSIZE x NDIM+1 x NDIM+1)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
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


class Scale(Affine):
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
        scale : torch.Tensor, int, float, optional
            the scale factor(s). Supported are:
                * a full transformation matrix of shape
                    (BATCHSIZE x NDIM x NDIM)
                * a single parameter (as float or int), which will be
                    replicated for all dimensions and batch samples
                * a single parameter per sample (as a 1d tensor), which will
                    be replicated for all dimensions
                * a single parameter per dimension (either as 1d tensor or as
                    2d transformation matrix), which will be replicated for
                    all batch samples
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


class StackedAffine(Affine):
    def __init__(
            self,
            *transforms: Union[Affine, Sequence[Union[Sequence[Affine],
                                                      Affine]]],
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

        super().__init__(matrix=None,
                         scale=None, rotation=None, translation=None,
                         keys=keys, grad=grad, degree=False,
                         output_size=output_size, adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)

        self.transforms = transforms

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Handles the matrix assembly and stacking

        Parameters
        ----------
        **data :
            the data to be transformed. Will be used to determine batchsize,
            dimensionality, dtype and device

        Returns
        -------
        torch.Tensor
            the (batched) transformation matrix

        """
        whole_trafo = None

        for trafo in self.transforms:
            matrix = matrix_to_homogeneous(trafo.assemble_matrix(**data))

            if whole_trafo is None:
                whole_trafo = matrix
            else:
                whole_trafo = torch.bmm(whole_trafo, matrix)

        return matrix_to_cartesian(whole_trafo)

# TODO: Add transforms around image center
# TODO: Add Resize Transform
