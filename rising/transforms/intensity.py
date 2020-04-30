import torch
from typing import Union, Sequence

from rising.transforms.abstract import BaseTransform, PerSampleTransform, \
    PerChannelTransform
from rising.transforms.functional.intensity import norm_range, norm_min_max, norm_mean_std, \
    norm_zero_mean_unit_std, add_noise, gamma_correction, add_value, scale_by_value, clamp
from rising.random import AbstractParameter

__all__ = ["Clamp", "NormRange", "NormMinMax",
           "NormZeroMeanUnitStd", "NormMeanStd", "Noise",
           "GaussianNoise", "ExponentialNoise", "GammaCorrection",
           "RandomValuePerChannel", "RandomAddValue", "RandomScaleValue"]


class Clamp(BaseTransform):
    def __init__(self, min: Union[float, AbstractParameter],
                 max: Union[float, AbstractParameter],
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        min:
            lower bound of clipping
        max:
            upper bound of clipping
        dims: tuple
            axes which should be mirrored
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=clamp, keys=keys, grad=grad,
                         min=min, max=max, property_names=('min', 'max'),
                         **kwargs)


class NormRange(PerSampleTransform):
    def __init__(self, min: Union[float, AbstractParameter],
                 max: Union[float, AbstractParameter], keys: Sequence = ('data',),
                 per_channel: bool = True,
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
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_range, keys=keys, grad=grad,
                         min=min, max=max, per_channel=per_channel,
                         property_names=('min', 'max'), **kwargs)


class NormMinMax(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """
        Scale data to [0, 1]

        Parameters
        ----------
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormZeroMeanUnitStd(PerSampleTransform):
    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """
        Normalize mean to zero and std to one

        Parameters
        ----------
        keys: Sequence
            keys to normalize
        per_channel: bool
            normalize per channel
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_zero_mean_unit_std, keys=keys,
                         grad=grad,
                         per_channel=per_channel, **kwargs)


class NormMeanStd(PerSampleTransform):
    def __init__(self, mean: Union[float, Sequence[float]],
                 std: Union[float, Sequence[float]],
                 keys: Sequence[str] = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
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
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_mean_std, keys=keys, grad=grad,
                         mean=mean, std=std, per_channel=per_channel, **kwargs)


class Noise(PerChannelTransform):
    def __init__(self, noise_type: str, per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Add noise to data

        Parameters
        ----------
        noise_type: str
            supports all inplace functions of a pytorch tensor
        per_channel: bool
            enable transformation per channel
        keys: Sequence
            keys to normalize
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to noise function

        See Also
        --------
        :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(augment_fn=add_noise, per_channel=per_channel, keys=keys,
                         grad=grad, noise_type=noise_type, **kwargs)


class ExponentialNoise(Noise):
    def __init__(self, lambd: float, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Add exponential noise to data

        Parameters
        ----------
        lambd: float
            lambda of exponential distribution
        keys: Sequence
            keys to normalize
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to noise function
        """
        super().__init__(noise_type='exponential_', lambd=lambd, keys=keys,
                         grad=grad, **kwargs)


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
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
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to noise function
        """
        super().__init__(noise_type='normal_', mean=mean, std=std, keys=keys,
                         grad=grad, **kwargs)


class GammaCorrection(BaseTransform):
    def __init__(self, gamma: Union[float, AbstractParameter],
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply gamma correction

        Parameters
        ----------
        gamma:
            specify value for gamma
        keys: Sequence
            keys to normalize
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass
        """
        super().__init__(augment_fn=gamma_correction, gamma=gamma,
                         property_names=("gamma",), keys=keys, grad=grad,
                         **kwargs)


class RandomValuePerChannel(PerChannelTransform):
    def __init__(self, augment_fn: callable,
                 random_sampler: AbstractParameter,
                 per_channel: bool = False, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Apply augmentations which take random values as input by keyword
        :param:`value`

        Parameters
        ----------
        augment_fn: callable
            augmentation function
        random_sampler:
            the sampler producing random numbers
        per_channel: bool
            enable transformation per channel
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=augment_fn, per_channel=per_channel,
                         keys=keys, grad=grad, random_sampler=random_sampler,
                         property_names=('random_sampler',),
                         **kwargs)

    def forward(self, **data) -> dict:
        """
        Perform Augmentation.

        Parameters
        ----------
        data: dict
            dict with data

        Returns
        -------
        dict
            augmented data
        """
        if self.per_channel:
            seed = torch.random.seed()
            for _key in self.keys:
                torch.manual_seed(seed)
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    rand_value = self.random_sampler.__get__(self)
                    out[:, _i] = self.augment_fn(
                        data[_key][:, _i], value=rand_value, out=out[:, _i],
                        **self.kwargs)
                data[_key] = out
        else:
            rand_value = self.random_sampler.__get__(self)
            for _key in self.keys:
                data[_key] = self.augment_fn(data[_key], value=rand_value, **self.kwargs)
        return data


class RandomAddValue(RandomValuePerChannel):
    def __init__(self, random_sampler: AbstractParameter,
                 per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Increase values additively

        Parameters
        ----------
        random_sampler:
            the sampler producing random numbers
        per_channel: bool
            enable transformation per channel
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=add_value, random_sampler=random_sampler,
                         per_channel=per_channel, keys=keys, grad=grad, **kwargs)


class RandomScaleValue(RandomValuePerChannel):
    def __init__(self, random_sampler: AbstractParameter,
                 per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Scale values

        Parameters
        ----------
        random_sampler:
            the sampler producing random numbers
        per_channel: bool
            enable transformation per channel
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=scale_by_value, random_sampler=random_sampler,
                         per_channel=per_channel, keys=keys, grad=grad, **kwargs)
