import torch
from typing import Union, Sequence


def reshape(value: Union[list, torch.Tensor],
            size: Union[Sequence, torch.Size]) -> Union[torch.Tensor, list]:
    if isinstance(value, torch.Tensor):
        return value.view(size)

    else:
        return reshape_list(value, size)


def reshape_list(flat_list: list, size: Union[torch.Size, tuple]) -> list:
    if len(size) == 1:
        return [flat_list.pop(0) for _ in range(size[0])]
    else:
        return [reshape_list(flat_list, size[1:]) for _ in range(size[0])]
