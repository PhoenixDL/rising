from typing import Sequence
from random import (sample as sample_without_replacement,
                    choices as sample_with_replacement)

from functools import partial

from rising.random.abstract import AbstractParameter

__all__ = [
    'DiscreteParameter'
]


class DiscreteParameter(AbstractParameter):
    """
    Samples parameters from a discrete population with or without
    replacement
    """

    def __init__(self, population: Sequence,
                 replacement: bool = False, weights: Sequence = None,
                 cum_weights: Sequence = None):
        """
        Args:
            population : the parameter population to sample from
            replacement : whether or not to sample with replacement
            weights : relative sampling weights
            cum_weights : cumulative sampling weights
        """
        super().__init__()
        if replacement:
            sample_fn = partial(sample_with_replacement, weights=weights,
                                cum_weights=cum_weights)
        else:
            if weights is not None or cum_weights is not None:
                raise ValueError(
                    'weights and cum_weights should only be specified if '
                    'replacement is set to True!')

            sample_fn = sample_without_replacement

        self.sample_fn = sample_fn
        self.population = population

    def sample(self, n_samples: int) -> list:
        """
        Samples from the discrete internal population

        Args:
            n_samples : the number of elements to sample

        Returns:
            list: the sampled values

        """
        return self.sample_fn(population=self.population, k=n_samples)
