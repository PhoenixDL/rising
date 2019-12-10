from rising.transforms.abstract import AbstractTransform, BaseTransform
from rising.transforms.functional.affine import affine_image_transform
from rising.utils.affine import AffineParamType, assemble_matrix_if_necessary
import torch
from typing import Sequence, Union


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
                 align_corners: bool = None,
                 **kwargs):

        super().__init__(augment_fn=affine_image_transform,
                         keys=keys,
                         grad=grad,
                         scale=scale,
                         rotation=rotation,
                         translation=translation,
                         matrix=matrix,
                         degree=degree,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)

    def assemble_matrix(self, **data):
        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2  # channel and batch dim
        device = data[self.keys[0]].device
        dtype = data[self.keys[0]].dtype

        matrix = assemble_matrix_if_necessary(
            batchsize, ndim, scale=self.scale, rotation=self.rotation,
            translation=self.translation, matrix=self.matrix,
            degree=self.degree, device=device, dtype=dtype)

        return matrix

    def forward(self, **data):
        matrix = self.assemble_matrix(**data)

        for key in self.keys:
            data[key] = self.augment_fn(data[key], matrix_batch=matrix,
                                        **self.kwargs)

        return data

    def __add__(self, other):
        if not isinstance(other, Affine):
            other = Affine(matrix=other)

        return StackedAffine(self, other, keys=self.keys, grad=self.grad,
                             degree=self.degree, output_size=self.output_size,
                             adjust_size=self.adjust_size,
                             interpolation_mode=self.interpolation_mode,
                             padding_mode=self.padding_mode,
                             align_corners=self.align_corners, **self.kwargs)

    def __radd__(self, other):
        if not isinstance(other, Affine):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad,
                           degree=self.degree, output_size=self.output_size,
                           adjust_size=self.adjust_size,
                           interpolation_mode=self.interpolation_mode,
                           padding_mode=self.padding_mode,
                           align_corners=self.align_corners, **self.kwargs)

        return StackedAffine(other, self, grad=other.grad,
                             degree=other.degree,
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
                 align_corners: bool = None,
                 **kwargs):
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
                 align_corners: bool = None,
                 **kwargs):
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
                 align_corners: bool = None,
                 **kwargs):
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
            degree: bool = False,
            output_size: tuple = None,
            adjust_size: bool = False,
            interpolation_mode: str = 'bilinear',
            padding_mode: str = 'zeros',
            align_corners: bool = None,
            **kwargs):

        if isinstance(transforms, (tuple, list)):
            if isinstance(transforms[0], (tuple, list)):
                transforms = transforms[0]

        else:
            transforms = (transforms,)

        super().__init__(transforms=transforms,
                         scale=None, rotation=None, translation=None,
                         keys=keys, grad=grad, degree=degree,
                         output_size=output_size, adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         **kwargs)

    def assemble_matrix(self, **data):
        whole_trafo = None

        for trafo in self.transforms:
            matrix = trafo.assemble_matrix(**data)

            if whole_trafo is None:
                whole_trafo = matrix
            else:
                whole_trafo = torch.bmm(whole_trafo, matrix)

        return whole_trafo
