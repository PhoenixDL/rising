from typing import Union

import torch
from torch.distributions import Distribution as TorchDistribution

from rising.random.abstract import AbstractParameter

__all__ = ["ContinuousParameter", "NormalParameter", "UniformParameter"]


class ContinuousParameter(AbstractParameter):
    """Class to perform parameter sampling from torch distributions"""

    def __init__(self, distribution: TorchDistribution):
        """
        Args:
            distribution : the distribution to sample from
        """
        super().__init__()
        self.dist = distribution

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the internal distribution

        Args:
            n_samples : the number of elements to sample

        Returns
            torch.Tensor: samples
        """
        return self.dist.sample((n_samples,))


class NormalParameter(ContinuousParameter):
    """
    Samples Parameters from a normal distribution.
    For details have a look at :class:`torch.distributions.Normal`
    """

    def __init__(self, mu: Union[float, torch.Tensor], sigma: Union[float, torch.Tensor]):
        """
        Args:
            mu : the distributions mean
            sigma : the distributions standard deviation
        """
        super().__init__(torch.distributions.Normal(loc=mu, scale=sigma))


class UniformParameter(ContinuousParameter):
    """
    Samples Parameters from a uniform distribution.
    For details have a look at :class:`torch.distributions.Uniform`
    """

    def __init__(self, low: Union[float, torch.Tensor], high: Union[float, torch.Tensor]):
        """
        Args:
            low : the lower range (inclusive)
            high : the higher range (exclusive)
        """
        super().__init__(torch.distributions.Uniform(low=low, high=high))
