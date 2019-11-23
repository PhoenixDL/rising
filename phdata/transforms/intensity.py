import torch
from typing import Union, Sequence

from .abstract import BaseTransform, PerSampleTransform
from .functional.intensity import norm_range, norm_min_max, norm_zero_mean_unit_std, \
    norm_mean_std, add_noise


class ClampTransform(BaseTransform):
    def __init__(self, min: float, max: float, keys: Sequence = ('data',), grad: bool = False, **kwargs):
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
    def __init__(self, min: float, max: float, keys: Sequence = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """
        Scale data to provided min and max values

        Parameters
        ----------
        min: float
            minimal value
        max: float
            maximal value
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_range, keys=keys, grad=grad,
                         min=min, max=max, per_channel=per_channel, **kwargs)


class NormMinMaxTransform(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True, grad: bool = False, **kwargs):
        """
        Scale data to [0, 1]

        Parameters
        ----------
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormZeroMeanUnitStdTransform(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True, grad: bool = False, **kwargs):
        """
        Normalize mean to zero and std to one

        Parameters
        ----------
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_zero_mean_unit_std, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormMeanStdTransform(PerSampleTransform):
    def __init__(self, mean: Union[float, Sequence], std: Union[float, Sequence],
                 keys: Sequence = ('data',), per_channel: bool = True, grad: bool = False, **kwargs):
        """
        Normalize mean and std with provided values

        Parameters
        ----------
        mean: float or Sequence
            used for mean normalization
        std: float or Sequence
            used for std normalization
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_mean_std, keys=keys, grad=grad,
                         mean=mean, std=std, per_channel=per_channel, **kwargs)


class NoiseTransform(BaseTransform):
    def __init__(self, noise_type: str, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Add noise to data

        Parameters
        ----------
        noise_type: str
            supports all inplace functions of a pytorch tensor
        keys: Sequence
            keys to normalize
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to noise function

        See Also
        --------
        :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(augment_fn=add_noise, keys=keys, grad=grad, noise_type=noise_type, **kwargs)


class ExponentialNoiseTransform(NoiseTransform):
    def __init__(self, lambd: float, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Add exponential noise to data

        Parameters
        ----------
        lambd: float
            lambda of exponential distribution
        keys: Sequence
            keys to normalize
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to noise function
        """
        super().__init__(noise_type='exponential_', lambd=lambd, keys=keys, grad=grad, **kwargs)


class GaussianNoiseTransform(NoiseTransform):
    def __init__(self, mean: float, std: float, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Add noise to data

        Parameters
        ----------
        mean: float
            mean of normal distribution
        std: float
            std of normal distribution
        keys: Sequence
            keys to normalize
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to noise function
        """
        super().__init__(noise_type='normal_', mean=mean, std=std, keys=keys, grad=grad, **kwargs)
