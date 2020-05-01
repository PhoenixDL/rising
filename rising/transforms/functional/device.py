from typing import List, Tuple, Union, Mapping, Hashable

import torch
from torch import Tensor

__all__ = ["to_device_dtype"]

data_type = Union[Tensor, List[Tensor], Tuple[Tensor], Mapping[Hashable, Tensor]]


def to_device_dtype(data: data_type, dtype: Union[torch.dtype, str] = None,
                    device: Union[torch.device, str] = None, **kwargs) -> data_type:
    """
    Pushes data to device

    Args:
        data: data which should be pushed to device. Sequence and mapping
            items are mapping individually to gpu
        device: target device
        kwargs: keyword arguments passed to assigning function

    Returns:
        data which was pushed to device
    """
    if torch.is_tensor(data):
        return data.to(device=device, dtype=dtype, **kwargs)
    elif isinstance(data, Mapping):
        return {key: to_device_dtype(item, device=device, dtype=dtype, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device_dtype(item, device=device, dtype=dtype, **kwargs) for item in data])
    else:
        return data
