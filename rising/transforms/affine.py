from rising.transforms.abstract import BaseTransform
from rising.transforms.functional.affine import affine_image_transform, \
    AffineParamType
import torch
from typing import Sequence


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
        # TODO: Add possibility for stacking transforms (just multiply their
        #  matrices instead of separate interpolations)


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