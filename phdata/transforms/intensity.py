import torch
from typing import Sequence

from .abstract import BaseTransform, PerSampleTransform
from .functional.intensity import norm_range, norm_min_max, norm_zero_mean_unit_std, norm_mean_std


class ClampTransform(BaseTransform):
    def __init__(self, min, max, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        dims: tuple
            axes which should be mirrored
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=torch.clamp, keys=keys, grad=grad, min=min, max=max, **kwargs)


class NormRangeTransform(PerSampleTransform):
    def __init__(self, min, max, keys: Sequence = ('data',), per_channel=True, grad: bool = False,
                 **kwargs):
        super().__init__(augment_fn=norm_range, keys=keys, grad=grad,
                         min=min, max=max, per_channel=per_channel, **kwargs)


class NormMinMaxTransform(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel=True, grad: bool = False, **kwargs):
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormZeroMeanUnitStdTransform(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel=True, grad: bool = False, **kwargs):
        super().__init__(augment_fn=norm_zero_mean_unit_std, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormMeanStdTransform(PerSampleTransform):
    def __init__(self, mean, std, keys: Sequence = ('data',), per_channel=True, grad: bool = False,
                 **kwargs):
        super().__init__(augment_fn=norm_mean_std, keys=keys, grad=grad,
                         mean=mean, std=std, per_channel=per_channel, **kwargs)
