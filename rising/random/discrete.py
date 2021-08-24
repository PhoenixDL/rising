from functools import partial
from itertools import combinations
from random import choices as sample_with_replacement
from random import sample as sample_without_replacement
from typing import List, Sequence

from rising.random.abstract import AbstractParameter

__all__ = ["DiscreteParameter", "DiscreteCombinationsParameter"]


def combinations_all(data: Sequence) -> List:
    """
    Return all combinations of all length for given sequence

    Args:
        data: sequence to get combinations of

    Returns:
        List: all combinations
    """
    comb = []
    for r in range(1, len(data) + 1):
        comb.extend(combinations(data, r=r))
    return comb


class DiscreteParameter(AbstractParameter):
    """
    Samples parameters from a discrete population with or without
    replacement
    """

    def __init__(
        self, population: Sequence, replacement: bool = False, weights: Sequence = None, cum_weights: Sequence = None
    ):
        """
        Args:
            population : the parameter population to sample from
            replacement : whether or not to sample with replacement
            weights : relative sampling weights
            cum_weights : cumulative sampling weights
        """
        super().__init__()
        if replacement:
            sample_fn = partial(sample_with_replacement, weights=weights, cum_weights=cum_weights)
        else:
            if weights is not None or cum_weights is not None:
                raise ValueError("weights and cum_weights should only be specified if " "replacement is set to True!")

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


class DiscreteCombinationsParameter(DiscreteParameter):
    """
    Sample parameters from an extended population which consists of all
    possible combinations of the given population
    """

    def __init__(self, population: Sequence, replacement: bool = False):
        """
        Args:
            population : population to build combination of
            replacement : whether or not to sample with replacement
        """
        population = combinations_all(population)
        super().__init__(population=population, replacement=replacement)
