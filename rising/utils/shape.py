from typing import Sequence, Union

import torch


def reshape(value: Union[list, torch.Tensor], size: Union[Sequence, torch.Size]) -> Union[torch.Tensor, list]:
    """
    Reshape sequence (list or tensor) to given size

    Args:
        value: sequence to reshape
        size: size to reshape to

    Returns:
        Union[torch.Tensor, list]: reshaped sequence
    """
    if isinstance(value, torch.Tensor):
        return value.view(size)
    else:
        return reshape_list(value, size)


def reshape_list(flat_list: list, size: Union[torch.Size, tuple]) -> list:
    """
    Reshape a (nested) list to a given shape

    Args:
        flat_list: (nested) list to reshape
        size: shape to reshape to

    Returns:
        list: reshape list
    """
    if len(size) == 1:
        return [flat_list.pop(0) for _ in range(size[0])]
    else:
        return [reshape_list(flat_list, size[1:]) for _ in range(size[0])]
