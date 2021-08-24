from typing import Any, Optional, Sequence, Tuple, Union

import torch

from rising.transforms.abstract import BaseTransform
from rising.transforms.functional.affine import AffineParamType, affine_image_transform, parametrize_matrix
from rising.utils.affine import matrix_to_cartesian, matrix_to_homogeneous
from rising.utils.checktype import check_scalar

__all__ = [
    "Affine",
    "BaseAffine",
    "StackedAffine",
    "Rotate",
    "Scale",
    "Translate",
    "Resize",
]


class Affine(BaseTransform):
    """
    Class Performing an Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(
        self,
        matrix: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]] = None,
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        per_sample: bool = True,
        **kwargs,
    ):
        """
        Args:
            matrix: if given, overwrites the parameters for :attr:`scale`,
                :attr:rotation` and :attr:`translation`.
                Should be a matrix of shape [(BATCHSIZE,) NDIM, NDIM(+1)]
                This matrix represents the whole transformation matrix
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be calculated
                dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode:padding mode for outside grid values
                ``'zeros``' | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            per_sample: sample different values for each element in the batch.
                The transform is still applied in a batched wise fashion.
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(augment_fn=affine_image_transform, keys=keys, grad=grad, **kwargs)
        self.matrix = matrix
        self.register_sampler("output_size", output_size)
        self.adjust_size = adjust_size
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.reverse_order = reverse_order
        self.per_sample = per_sample

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix
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
            "Got %s but expected %s" % (str(tuple(self.matrix.shape)), str((batchsize, ndim, ndim + 1)))
        )

    def forward(self, **data) -> dict:
        """
        Assembles the matrix and applies it to the specified sample-entities.

        Args:
            **data: the data to transform

        Returns:
            dict: dictionary containing the transformed data
        """
        matrix = self.assemble_matrix(**data)

        for key in self.keys:
            data[key] = self.augment_fn(
                data[key],
                matrix_batch=matrix,
                output_size=self.output_size,
                adjust_size=self.adjust_size,
                interpolation_mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                reverse_order=self.reverse_order,
                **self.kwargs,
            )

        return data

    def __add__(self, other: Any) -> BaseTransform:
        """
        Makes ``trafo + other_trafo work``
        (stacks them for dynamic assembling)

        Args:
            other: the other transformation

        Returns:
            StackedAffine: a stacked affine transformation
        """
        if not isinstance(other, Affine):
            other = Affine(
                matrix=other,
                keys=self.keys,
                grad=self.grad,
                output_size=self.output_size,
                adjust_size=self.adjust_size,
                interpolation_mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                **self.kwargs,
            )

        return StackedAffine(
            self,
            other,
            keys=self.keys,
            grad=self.grad,
            output_size=self.output_size,
            adjust_size=self.adjust_size,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            **self.kwargs,
        )

    def __radd__(self, other) -> BaseTransform:
        """
        Makes ``other_trafo + trafo`` work
        (stacks them for dynamic assembling)

        Args:
            other: the other transformation

        Returns:
            StackedAffine: a stacked affine transformation
        """
        if not isinstance(other, Affine):
            other = Affine(
                matrix=other,
                keys=self.keys,
                grad=self.grad,
                output_size=self.output_size,
                adjust_size=self.adjust_size,
                interpolation_mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                **self.kwargs,
            )

        return StackedAffine(
            other,
            self,
            grad=other.grad,
            output_size=other.output_size,
            adjust_size=other.adjust_size,
            interpolation_mode=other.interpolation_mode,
            padding_mode=other.padding_mode,
            align_corners=other.align_corners,
            **other.kwargs,
        )


class StackedAffine(Affine):
    """
    Class to stack multiple affines with dynamic ensembling by matrix
    multiplication to avoid multiple interpolations.
    """

    def __init__(
        self,
        *transforms: Union[Affine, Sequence[Union[Sequence[Affine], Affine]]],
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs,
    ):
        """
        Args:
            transforms: the transforms to stack.
                Each transform must have a function
                called ``assemble_matrix``, which is called to dynamically
                assemble stacked matrices. Afterwards these transformations
                are stacked by matrix-multiplication to only perform a single
                interpolation
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be calculated
                dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        if isinstance(transforms, (tuple, list)):
            if isinstance(transforms[0], (tuple, list)):
                transforms = transforms[0]

        # ensure trafos are Affines and not raw matrices
        transforms = tuple([trafo if isinstance(trafo, Affine) else Affine(matrix=trafo) for trafo in transforms])

        super().__init__(
            keys=keys,
            grad=grad,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            reverse_order=reverse_order,
            **kwargs,
        )

        self.transforms = transforms

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Handles the matrix assembly and stacking

        Args:
            **data: the data to be transformed.
                Will be used to determine batchsize,
                dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix

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
    """
    Class performing a basic Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`."""

    def __init__(
        self,
        scale: Optional[AffineParamType] = None,
        rotation: Optional[AffineParamType] = None,
        translation: Optional[AffineParamType] = None,
        degree: bool = False,
        image_transform: bool = True,
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        per_sample: bool = True,
        **kwargs,
    ):
        """
        Args:
            scale: the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a scaling factor of 1
            rotation: the rotation factor(s). The rotation is performed in
                consecutive order axis0 -> axis1 (-> axis 2). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a rotation angle of 0
            translation : torch.Tensor, int, float
                the translation offset(s) relative to image (should be in the
                range [0, 1]). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a translation offset of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            per_sample: sample different values for each element in the batch.
                The transform is still applied in a batched wise fashion.
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(
            keys=keys,
            grad=grad,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            reverse_order=reverse_order,
            per_sample=per_sample,
            **kwargs,
        )
        self.register_sampler("scale", scale)
        self.register_sampler("rotation", rotation)
        self.register_sampler("translation", translation)

        self.degree = degree
        self.image_transform = image_transform

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix
        """
        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2  # channel and batch dim
        device = data[self.keys[0]].device
        dtype = data[self.keys[0]].dtype

        self.matrix = parametrize_matrix(
            scale=self.sample_for_batch("scale", batchsize),
            rotation=self.sample_for_batch("rotation", batchsize),
            translation=self.sample_for_batch("translation", batchsize),
            batchsize=batchsize,
            ndim=ndim,
            degree=self.degree,
            device=device,
            dtype=dtype,
            image_transform=self.image_transform,
        )
        return self.matrix

    def sample_for_batch(self, name: str, batchsize: int) -> Optional[Union[Any, Sequence[Any]]]:
        """
        Sample elements for batch

        Args:
            name: name of parameter
            batchsize: batch size

        Returns:
            Optional[Union[Any, Sequence[Any]]]: sampled elements
        """
        elem = getattr(self, name)
        if elem is not None and self.per_sample:
            return [elem] + [getattr(self, name) for _ in range(batchsize - 1)]
        else:
            return elem  # either a single scalar value or None


class Rotate(BaseAffine):
    """
    Class Performing a Rotation-OnlyAffine Transformation on a given
    sample dict. The rotation is applied in consecutive order:
    rot axis 0 -> rot axis 1 -> rot axis 2
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(
        self,
        rotation: AffineParamType,
        keys: Sequence = ("data",),
        grad: bool = False,
        degree: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs,
    ):
        """
        Args:
            rotation: the rotation factor(s). The rotation is performed in
                consecutive order axis0 -> axis1 (-> axis 2). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * ``None`` will be treated as a rotation angle of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(
            scale=None,
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
            **kwargs,
        )


class Translate(BaseAffine):
    """
    Class Performing an Translation-Only
    Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(
        self,
        translation: AffineParamType,
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        unit: str = "pixel",
        reverse_order: bool = False,
        **kwargs,
    ):
        """
        Args:
            translation : torch.Tensor, int, float
                the translation offset(s) relative to image (should be in the
                range [0, 1]). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a translation offset of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            unit: defines the unit of the translation. Either ```relative'``
                to the image size or in ```pixel'``
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(
            scale=None,
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
            **kwargs,
        )
        self.unit = unit

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix [N, NDIM, NDIM]
        """
        matrix = super().assemble_matrix(**data)
        if self.unit.lower() == "pixel":
            img_size = torch.tensor(data[self.keys[0]].shape[2:]).to(matrix)
            matrix[..., -1] = matrix[..., -1] / img_size
        return matrix


class Scale(BaseAffine):
    """Class Performing a Scale-Only Affine Transformation on a given
    sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(
        self,
        scale: AffineParamType,
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs,
    ):
        """
        Args:
            scale : torch.Tensor, int, float, optional
                the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for
                all batch samples
                * None will be treated as a scaling factor of 1
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
                if True, the resulting image size will be calculated
                dynamically to ensure that the whole image fits.
            interpolation_mode : str
                interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners : bool
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema
                (-1 and 1) are considered as referring to the center points of
                the input’s corner pixels. If set to False, they are instead
                considered as referring to the corner points of the input’s
                corner pixels, making the sampling more resolution agnostic.
            reverse_order: bool
                reverses the coordinate order of the transformation to conform
                to the pytorch convention: transformation params order
                [W,H(,D)] and batch order [(D,)H,W]
            **kwargs :
                additional keyword arguments passed to the affine transform
        """
        super().__init__(
            scale=scale,
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
            **kwargs,
        )


class Resize(Scale):
    def __init__(
        self,
        size: Union[int, Tuple[int]],
        keys: Sequence = ("data",),
        grad: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs,
    ):
        """
        Class Performing a Resizing Affine Transformation on a given
        sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Args:
            size: the target size. If int, this will be repeated for all the
                dimensions
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            interpolation_mode: nterpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'
            padding_mode: padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners: Geometrically, we consider the pixels of the input
                as squares rather than points. If set to True, the extrema
                (-1 and 1) are considered as referring to the center points of
                the input’s corner pixels. If set to False, they are instead
                considered as referring to the corner points of the input’s
                corner pixels, making the sampling more resolution agnostic.
            reverse_order: reverses the coordinate order of the transformation
                to conform to the pytorch convention: transformation params
                order [W,H(,D)] and batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the affine
                transform

        Notes:
            The offsets for shifting back and to origin are calculated on the
            entry matching the first item iin :attr:`keys` for each batch
        """
        super().__init__(
            output_size=None,
            scale=None,
            keys=keys,
            grad=grad,
            adjust_size=False,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            reverse_order=reverse_order,
            **kwargs,
        )
        self.register_sampler("size", size)

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Handles the matrix assembly and calculates the scale factors for
        resizing

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix

        """
        curr_img_size = data[self.keys[0]].shape[2:]
        output_size = self.size

        if torch.is_tensor(output_size):
            self.output_size = int(output_size.item())
        else:
            self.output_size = tuple(int(t.item()) for t in output_size)

        if check_scalar(output_size):
            output_size = [output_size] * len(curr_img_size)

        self.scale = [float(output_size[i]) / float(curr_img_size[i]) for i in range(len(curr_img_size))]
        matrix = super().assemble_matrix(**data)
        return matrix
