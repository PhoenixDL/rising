from typing import Union

import torch
from torch.distributions import Distribution as TorchDistribution

from rising.random.base_parameter import AbstractParameter


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
