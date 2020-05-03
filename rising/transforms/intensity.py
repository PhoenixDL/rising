import random
from typing import Union, Sequence

import torch

from rising.transforms.functional.intensity import norm_range, norm_min_max, norm_mean_std, \
    norm_zero_mean_unit_std, add_noise, gamma_correction, add_value, scale_by_value
from rising.utils import check_scalar
from .abstract import BaseTransform, PerSampleTransform, AbstractTransform, \
    PerChannelTransform, RandomProcess

__all__ = ["Clamp", "NormRange", "NormMinMax",
           "NormZeroMeanUnitStd", "NormMeanStd", "Noise",
           "GaussianNoise", "ExponentialNoise", "GammaCorrection",
           "RandomValuePerChannel", "RandomAddValue", "RandomScaleValue"]


class Clamp(BaseTransform):
    """Clamp Inputs to range"""

    def __init__(self, min: float, max: float, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=torch.clamp, keys=keys, grad=grad,
                         min=min, max=max, **kwargs)


class NormRange(PerSampleTransform):
    """Scale data to provided min and max values"""

    def __init__(self, min: float, max: float, keys: Sequence = ('data',),
                 per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_range, keys=keys, grad=grad,
                         min=min, max=max, per_channel=per_channel, **kwargs)


class NormMinMax(PerSampleTransform):
    """Scale data to [0, 1]"""

    # TODO: Rename to NormUnitRange?
    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """

        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormZeroMeanUnitStd(PerSampleTransform):
    """Normalize mean to zero and std to one"""

    def __init__(self, keys: Sequence = ('data',), per_channel: bool = True, grad: bool = False, **kwargs):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_zero_mean_unit_std,
                         keys=keys, grad=grad,
                         per_channel=per_channel, **kwargs)


class NormMeanStd(PerSampleTransform):
    """Normalize mean and std with provided values"""

    def __init__(self, mean: Union[float, Sequence], std: Union[float, Sequence],
                 keys: Sequence = ('data',), per_channel: bool = True,
                 grad: bool = False, **kwargs):
        """
        Args:
            mean: used for mean normalization
            std: used for std normalization
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_mean_std, keys=keys, grad=grad,
                         mean=mean, std=std, per_channel=per_channel, **kwargs)


class Noise(PerChannelTransform):
    """Add noise to data"""

    def __init__(self, noise_type: str, per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            noise_type: supports all inplace functions of a
                :class:`torch.Tensor`
            per_channel: enable transformation per channel
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to noise function

        See Also
        --------
        :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(augment_fn=add_noise, per_channel=per_channel, keys=keys,
                         grad=grad, noise_type=noise_type, **kwargs)


class ExponentialNoise(Noise):
    """Add exponential noise to data"""

    def __init__(self, lambd: float, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            lambd: lambda of exponential distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type='exponential_', lambd=lambd, keys=keys, grad=grad, **kwargs)


class GaussianNoise(Noise):
    """Add gaussian noise to data"""

    def __init__(self, mean: float, std: float, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            mean: mean of normal distribution
            std: std of normal distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type='normal_', mean=mean, std=std, keys=keys, grad=grad, **kwargs)


class GammaCorrection(AbstractTransform):
    """Apply Gamma correction"""

    def __init__(self, gamma: Union[float, Sequence] = (0.5, 2),
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            gamma: if gamma is float it is always applied.
                if gamma is a sequence it is interpreted as  the minimal and
                maximal value. If the maximal value is greater than one,
                the transform chooses gamma < 1 in 50% of the cases and
                gamma > 1 in the other cases.
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass
        """
        super().__init__(augment_fn=gamma_correction, keys=keys, grad=grad)
        self.kwargs = kwargs
        self.gamma = gamma
        if not check_scalar(self.gamma):
            if not len(self.gamma) == 2:
                raise TypeError(f"Gamma needs to be scalar or a Sequence with two entries "
                                f"(min, max), found {self.gamma}")

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            augmented data
        """
        if check_scalar(self.gamma):
            _gamma = self.gamma
        elif self.gamma[1] < 1:
            _gamma = random.uniform(self.gamma[0], self.gamma[1])
        else:
            if random.random() < 0.5:
                _gamma = _gamma = random.uniform(self.gamma[0], 1)
            else:
                _gamma = _gamma = random.uniform(1, self.gamma[1])

        for _key in self.keys:
            data[_key] = self.augment_fn(data[_key], _gamma, **self.kwargs)
        return data


class RandomValuePerChannel(RandomProcess, PerChannelTransform):
    """
    Apply augmentations which take random values as input by keyword
    :attr:`value`
    """

    def __init__(self, augment_fn: callable, random_mode: str, random_args: Sequence = (),
                 per_channel: bool = False, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
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
        super().__init__(augment_fn=augment_fn, per_channel=per_channel,
                         keys=keys, grad=grad, random_mode=random_mode,
                         random_args=random_args, random_module="random",
                         rand_seq=False, **kwargs)

    def forward(self, **data) -> dict:
        """
        Perform Augmentation.

        Args:
            data: dict with data

        Returns:
            augmented data
        """
        if self.per_channel:
            random_seed = random.random()
            for _key in self.keys:
                random.seed(random_seed)
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    rand_value = self.rand()
                    out[:, _i] = self.augment_fn(data[_key][:, _i], value=rand_value,
                                                 out=out[:, _i], **self.kwargs)
                data[_key] = out
            return data
        else:
            self.kwargs["value"] = self.rand()
            return super().forward(**data)


class RandomAddValue(RandomValuePerChannel):
    """Increase values additively"""

    def __init__(self, random_mode: str, per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            random_mode: specifies distribution which should be used to
                sample additive value (supports all random generators from
                python random package)
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=add_value, random_mode=random_mode,
                         per_channel=per_channel, keys=keys, grad=grad, **kwargs)


class RandomScaleValue(RandomValuePerChannel):
    """Scale Values"""

    def __init__(self, random_mode, per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            random_mode: specifies distribution which should be used to sample
                additive value (supports all random generators from python
                random package)
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=scale_by_value, random_mode=random_mode,
                         per_channel=per_channel, keys=keys, grad=grad, **kwargs)
