import torch
import math
from typing import Union


def comb(n: Union[int, torch.Tensor], k: Union[int, torch.Tensor],
         exact: bool = False,
         repetition: bool = False) -> Union[int, torch.Tensor]:
    """
    The number of xombinations of ``n`` things taken ``k`` at a time

    Args:
        n: number of things
        k: number of elements taken
        exact: then floating point precision is used, otherwise
            exact long integer is computed.. Defaults to False.
        repetition: if True: the number of combinations with
            repetition is computed.. Defaults to False.

    Returns:
        Union[torch.Tensor, int]: number of combinations
    """
    if repetition:
        return comb(n + k - 1, k, exact)

    if exact:
        n = int(n)
        k = int(k)

        if (k > n) or (n < 0) or (k < 0):
            return 0
        val = 1
        for j in range(min(k, n - k)):
            val = (val * (n - j)) // (j + 1)
        return val

    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n)

    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)

    cond = (k <= n) & (n >= 0) & (k >= 0)
    vals = binomial_coefficient(n, k)

    vals[1 - cond] = 0

    return vals


def binomial_coefficient(n: Union[int, torch.Tensor],
                         k: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
    """
    Calculates the binomial coefficient for given :attr`n` and :attr`k`

    Args:
        n: number of things
        k: number of elements taken

    Returns:
        Union[int, torch.Tensor]: binomial coefficient
    """
    return faculty(n) / (faculty(k) * faculty(n - k))


def faculty(k: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
    """
    Calculates the faculty of a given number or set of numbers (elementwise)

    Args:
        k: the number to calculate the faculty for

    Raises:
        ValueError: if k is tensor and contains negative elements

    Returns:
        Union[int, torch.Tensor]: factorial number
    """
    if isinstance(k, int):
        return math.factorial(k)

    elif isinstance(k, torch.Tensor):
        if (k < 0).any():
            raise ValueError('k contans negative elemets')

        k = k.long()

        k = torch.clamp_min(k, 1)

        if k.sum() == k.numel():  # all tensor elements will be 1 in that case
            return k

        return k * faculty(k - 1)
