from typing import Optional, Sequence, Union

import torch

from rising.random import AbstractParameter
from rising.transforms.abstract import BaseTransform, PerChannelTransform, PerSampleTransform
from rising.transforms.functional.intensity import (
    add_noise,
    add_value,
    bezier_3rd_order,
    clamp,
    gamma_correction,
    norm_mean_std,
    norm_min_max,
    norm_range,
    norm_zero_mean_unit_std,
    random_inversion,
    scale_by_value,
)

__all__ = [
    "Clamp",
    "NormRange",
    "NormMinMax",
    "NormZeroMeanUnitStd",
    "NormMeanStd",
    "Noise",
    "GaussianNoise",
    "ExponentialNoise",
    "GammaCorrection",
    "RandomValuePerChannel",
    "RandomAddValue",
    "RandomScaleValue",
    "RandomBezierTransform",
    "InvertAmplitude",
]


class Clamp(BaseTransform):
    """Apply augment_fn to keys"""

    def __init__(
        self,
        min: Union[float, AbstractParameter],
        max: Union[float, AbstractParameter],
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """


        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=clamp, keys=keys, grad=grad, min=min, max=max, property_names=("min", "max"), **kwargs
        )


class NormRange(PerSampleTransform):
    def __init__(
        self,
        min: Union[float, AbstractParameter],
        max: Union[float, AbstractParameter],
        keys: Sequence = ("data",),
        per_channel: bool = True,
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(
            augment_fn=norm_range,
            keys=keys,
            grad=grad,
            min=min,
            max=max,
            per_channel=per_channel,
            property_names=("min", "max"),
            **kwargs
        )


class NormMinMax(PerSampleTransform):
    """Norm to [0, 1]"""

    def __init__(
        self,
        keys: Sequence = ("data",),
        per_channel: bool = True,
        grad: bool = False,
        eps: Optional[float] = 1e-8,
        **kwargs
    ):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad, per_channel=per_channel, eps=eps, **kwargs)


class NormZeroMeanUnitStd(PerSampleTransform):
    """Normalize mean to zero and std to one"""

    def __init__(
        self,
        keys: Sequence = ("data",),
        per_channel: bool = True,
        grad: bool = False,
        eps: Optional[float] = 1e-8,
        **kwargs
    ):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(
            augment_fn=norm_zero_mean_unit_std, keys=keys, grad=grad, per_channel=per_channel, eps=eps, **kwargs
        )


class NormMeanStd(PerSampleTransform):
    """Normalize mean and std with provided values"""

    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
        keys: Sequence[str] = ("data",),
        per_channel: bool = True,
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            mean: used for mean normalization
            std: used for std normalization
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(
            augment_fn=norm_mean_std, keys=keys, grad=grad, mean=mean, std=std, per_channel=per_channel, **kwargs
        )


class Noise(PerChannelTransform):
    """
    Add noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(
        self, noise_type: str, per_channel: bool = False, keys: Sequence = ("data",), grad: bool = False, **kwargs
    ):
        """
        Args:
            noise_type: supports all inplace functions of a
                :class:`torch.Tensor`
            per_channel: enable transformation per channel
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to noise function

        See Also:
            :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(
            augment_fn=add_noise, per_channel=per_channel, keys=keys, grad=grad, noise_type=noise_type, **kwargs
        )


class ExponentialNoise(Noise):
    """
    Add exponential noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(self, lambd: float, keys: Sequence = ("data",), grad: bool = False, **kwargs):
        """
        Args:
            lambd: lambda of exponential distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type="exponential_", lambd=lambd, keys=keys, grad=grad, **kwargs)


class GaussianNoise(Noise):
    """
    Add gaussian noise to data

    .. warning:: This transform will apply different noise patterns to different keys.
    """

    def __init__(self, mean: float, std: float, keys: Sequence = ("data",), grad: bool = False, **kwargs):
        """
        Args:
            mean: mean of normal distribution
            std: std of normal distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type="normal_", mean=mean, std=std, keys=keys, grad=grad, **kwargs)


class GammaCorrection(BaseTransform):
    """Apply Gamma correction"""

    def __init__(
        self, gamma: Union[float, AbstractParameter], keys: Sequence = ("data",), grad: bool = False, **kwargs
    ):
        """
        Args:
            gamma: define gamma
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass
        """
        super().__init__(
            augment_fn=gamma_correction, gamma=gamma, property_names=("gamma",), keys=keys, grad=grad, **kwargs
        )


class RandomValuePerChannel(PerChannelTransform):
    """
    Apply augmentations which take random values as input by keyword :attr:`value`

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        augment_fn: callable,
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            augment_fn: augmentation function
            random_mode: specifies distribution which should be used to
                sample additive value. All function from python's random
                module are supported
            random_args: positional arguments passed for random function
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=augment_fn,
            per_channel=per_channel,
            keys=keys,
            grad=grad,
            random_sampler=random_sampler,
            property_names=("random_sampler",),
            **kwargs
        )

    def forward(self, **data) -> dict:
        """
        Perform Augmentation.

        Args:
            data: dict with data

        Returns:
            dict: augmented data
        """
        if self.per_channel:
            seed = torch.random.get_rng_state()
            for _key in self.keys:
                torch.random.set_rng_state(seed)
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    rand_value = self.random_sampler
                    out[:, _i] = self.augment_fn(data[_key][:, _i], value=rand_value, out=out[:, _i], **self.kwargs)
                data[_key] = out
        else:
            rand_value = self.random_sampler
            for _key in self.keys:
                data[_key] = self.augment_fn(data[_key], value=rand_value, **self.kwargs)
        return data


class RandomAddValue(RandomValuePerChannel):
    """
    Increase values additively

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=add_value, random_sampler=random_sampler, per_channel=per_channel, keys=keys, grad=grad, **kwargs
        )


class RandomScaleValue(RandomValuePerChannel):
    """
    Scale Values

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(
        self,
        random_sampler: AbstractParameter,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=scale_by_value,
            random_sampler=random_sampler,
            per_channel=per_channel,
            keys=keys,
            grad=grad,
            **kwargs
        )


class RandomBezierTransform(BaseTransform):
    """Apply a random 3rd order bezier spline to the intensity values, as proposed in Models Genesis."""

    def __init__(self, maxv: float = 1.0, minv: float = 0.0, keys: Sequence = ("data",), **kwargs):
        super().__init__(augment_fn=bezier_3rd_order, maxv=maxv, minv=minv, keys=keys, grad=False, **kwargs)


class InvertAmplitude(BaseTransform):
    """
    Inverts the amplitude with probability p according to the following formula:
    out = maxv + minv - data
    """

    def __init__(self, prob: float = 0.5, maxv: float = 1.0, minv: float = 0.0, keys: Sequence = ("data",), **kwargs):
        super().__init__(
            augment_fn=random_inversion, prob_inversion=prob, maxv=maxv, minv=minv, keys=keys, grad=False, **kwargs
        )
