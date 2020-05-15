import torch
from typing import Union, Any


def check_scalar(x: Union[Any, float, int]) -> bool:
    """
    Provide interface to check for scalars

    Args:
        x: object to check for scalar

    Returns:
        bool" True if input is scalar
    """
    return isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and x.numel() == 1)
