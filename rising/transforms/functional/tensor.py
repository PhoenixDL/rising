from typing import Hashable, List, Mapping, Tuple, Union

import torch
from torch import Tensor

__all__ = ["tensor_op", "to_device_dtype"]

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
        Union[torch.Tensor, Sequence, Mapping]: data which was pushed to device
    """
    if torch.is_tensor(data):
        return getattr(data, fn)(*args, **kwargs)
    elif isinstance(data, Mapping):
        return {key: tensor_op(item, fn, *args, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([tensor_op(item, fn, *args, **kwargs) for item in data])
    else:
        return data


def to_device_dtype(
    data: data_type, dtype: Union[torch.dtype, str] = None, device: Union[torch.device, str] = None, **kwargs
) -> data_type:
    """
    Pushes data to device

    Args:
        data: data which should be pushed to device. Sequence and mapping
            items are mapping individually to gpu
        device: target device
        kwargs: keyword arguments passed to assigning function

    Returns:
        Union[torch.Tensor, Sequence, Mapping]: data which was pushed to device
    """
    if torch.is_tensor(data):
        return data.to(device=device, dtype=dtype, **kwargs)
    elif isinstance(data, Mapping):
        return {key: to_device_dtype(item, device=device, dtype=dtype, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device_dtype(item, device=device, dtype=dtype, **kwargs) for item in data])
    else:
        return data
