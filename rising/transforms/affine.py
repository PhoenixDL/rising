import torch
from typing import Sequence, Union, Iterable

from rising.transforms.abstract import BaseTransform
from rising.transforms.functional.affine import affine_image_transform, \
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


class Affine(BaseTransform):
    def __init__(self,
                 matrix: Union[torch.Tensor, Sequence[Sequence[float]]] = None,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 reverse_order: bool = False,
                 **kwargs):
        """
        Class Performing an Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        matrix : torch.Tensor, optional
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
        **kwargs :
            additional keyword arguments passed to the affine transform
        """
        super().__init__(augment_fn=affine_image_transform,
                         keys=keys,
                         grad=grad,
                         **kwargs)
        self.matrix = matrix
        self.output_size = output_size
        self.adjust_size = adjust_size
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.reverse_order = reverse_order

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
        if self.matrix is None:
            raise ValueError("Matrix needs to be initialized or overwritten.")
        if not torch.is_tensor(self.matrix):
            self.matrix = torch.tensor(self.matrix)
        self.matrix = self.matrix.to(data[self.keys[0]])

        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2  # channel and batch dim

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
                reverse_order=self.reverse_order,
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
            reverse_order: bool = False,
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
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
                         reverse_order=reverse_order,
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
                 reverse_order: bool = False,
                 **kwargs,
                 ):
        """
        Class performing a basic Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        scale : torch.Tensor, int, float, optional
            the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be replicated
                    for all dimensions and batch samples
                * a parameter per sample, which will be
                    replicated for all dimensions
                * a parameter per dimension, which will be replicated for all
                    batch samples
                * a parameter per sampler per dimension
            None will be treated as a scaling factor of 1
        rotation : torch.Tensor, int, float, optional
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
        translation : torch.Tensor, int, float
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
        **kwargs :
            additional keyword arguments passed to the affine transform
        """
        super().__init__(keys=keys, grad=grad, output_size=output_size,
                         adjust_size=adjust_size, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         reverse_order=reverse_order,
                         **kwargs)
        self.register_sampler('scale', scale)
        self.register_sampler('rotation', rotation)
        self.register_sampler('translation', translation)

        self.degree = degree
        self.image_transform = image_transform

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
                 reverse_order: bool = False,
                 **kwargs):
        """
        Class Performing a Rotation-OnlyAffine Transformation on a given
        sample dict. The rotation is applied in consecutive order:
        rot axis 0 -> rot axis 1 -> rot axis 2
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        rotation : torch.Tensor, int, float, optional
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
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
                         reverse_order=reverse_order,
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
                 reverse_order: bool = False,
                 **kwargs):
        """
        Class Performing an Translation-Only
        Affine Transformation on a given sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Parameters
        ----------
        translation : torch.Tensor, int, float
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
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
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
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
                         reverse_order=reverse_order,
                         **kwargs)
        self.unit = unit

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
            the (batched) transformation matrix [N, NDIM, NDIM]
        """
        matrix = super().assemble_matrix(**data)
        if self.unit.lower() == 'pixel':
            img_size = torch.tensor(data[self.keys[0]].shape[2:]).to(matrix)
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
                 reverse_order: bool = False,
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
        **kwargs :
            additional keyword arguments passed to the affine transform
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
                         reverse_order=reverse_order,
                         **kwargs)


class Resize(Scale):
    def __init__(self,
                 size: Union[int, Iterable],
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 reverse_order: bool = False,
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
        align_corners : bool
            Geometrically, we consider the pixels of the input as
            squares rather than points. If set to True, the extrema (-1 and 1)
            are considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
        reverse_order: bool
            reverses the coordinate order of the transformation to conform
            to the pytorch convention: transformation params order [W,H(,D)] and
            batch order [(D,)H,W]
        **kwargs :
            additional keyword arguments passed to the affine transform

        Note
        ----
        The offsets for shifting back and to origin are calculated on the
        entry matching the first item iin :attr:`keys` for each batch
        """
        super().__init__(output_size=size,
                         scale=None,
                         keys=keys,
                         grad=grad,
                         adjust_size=False,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         reverse_order=reverse_order,
                         **kwargs)

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Handles the matrix assembly and calculates the scale factors for
        resizing

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
        curr_img_size = data[self.keys[0]].shape[2:]

        was_scalar = check_scalar(self.output_size)

        if was_scalar:
            self.output_size = [self.output_size] * len(curr_img_size)

        # TODO: Figure out a way to bypass the scale parameter
        self.scale = [self.output_size[i] / curr_img_size[-i]
                      for i in range(len(curr_img_size))]

        matrix = super().assemble_matrix(**data)

        if was_scalar:
            self.output_size = self.output_size[0]

        return matrix
