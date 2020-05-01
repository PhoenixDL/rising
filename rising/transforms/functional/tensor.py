from typing import List, Tuple, Union, Mapping, Hashable

import torch
from torch import Tensor

__all__ = ["tensor_op"]

data_type = Union[Tensor, List[Tensor], Tuple[Tensor], Mapping[Hashable, Tensor]]


def tensor_op(data: data_type, fn: str, *args, **kwargs) -> data_type:
    """
    Invokes a function form a tensor

    Args:
    data: data which should be pushed to device. Sequence and mapping items
        are mapping individually to gpu
    fn: tensor function
    *args: positional arguments passed to tensor function
    **kwargs: keyword arguments passed to tensor function

    Returns:
        data which was pushed to device
    """
    if torch.is_tensor(data):
        return getattr(data, fn)(*args, **kwargs)
    elif isinstance(data, Mapping):
        return {key: tensor_op(item, fn, *args, **kwargs)
                for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([tensor_op(item, fn, *args, **kwargs)
                           for item in data])
    else:
        return data
