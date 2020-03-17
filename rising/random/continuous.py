from typing import Union

import torch
from torch.distributions import Distribution as TorchDistribution

from rising.random.abstract import AbstractParameter

__all__ = [
    'ContinuousParameter',
    'NormalParameter',
    'UniformParameter'
]


class ContinuousParameter(AbstractParameter):
    def __init__(self, distribution: TorchDistribution):
        """
        Class to perform parameter sampling from torch distributions

        Parameters
        ----------
        distribution : torch.distributions.Distribution
            the distribution to sample from
        """
        super().__init__()
        self.dist = distribution

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the internal distribution

        Parameters
        ----------
        n_samples : int
            the number of elements to sample

        Returns
        -------
        torch.Tensor

        """
        return self.dist.sample_n(n_samples)


class NormalParameter(ContinuousParameter):
    def __init__(self, mu: Union[float, torch.Tensor],
                 sigma: Union[float, torch.Tensor]):
        """
        Samples Parameters from a normal distribution.
        For details have a look at :class:`torch.distributions.Normal`

        Parameters
        ----------
        mu : float or torch.Tensor
            the distributions mean
        sigma : float or torch.Tensor
            the distributions standard deviation
        """
        super().__init__(torch.distributions.Normal(loc=mu, scale=sigma))


class UniformParameter(ContinuousParameter):
    def __init__(self, low: Union[float, torch.Tensor],
                 high: Union[float, torch.Tensor]):
        """
        Samples Parameters from a uniform distribution.
        For details have a look at :class:`torch.distributions.Uniform`

        Parameters
        ----------
        low : float or torch.Tensor
            the lower range (inclusive)
        high : float or torch.Tensor
            the higher range (exclusive)
        """
        super().__init__(torch.distributions.Uniform(low=low, high=high))
