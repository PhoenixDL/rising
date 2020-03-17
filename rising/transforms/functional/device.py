import torch
from torch import Tensor
from typing import List, Tuple, Union, Mapping, Hashable

__all__ = ["to_device_dtype"]

data_type = Union[Tensor, List[Tensor], Tuple[Tensor], Mapping[Hashable, Tensor]]


def to_device_dtype(data: data_type, dtype: Union[torch.dtype, str] = None,
                    device: Union[torch.device, str] = None, **kwargs) -> data_type:
    """
    Pushes data to device

    Parameters
    ----------
    data: data_type
        data which should be pushed to device. Sequence and mapping items are
        mapping individually to gpu
    dtype : Union[torch.dtype, str]
        target dtype
    device: Union[torch.device, str]
        target device
    kwargs:
        keyword arguments passed to assiging function

    Returns
    -------
    data_type:
        data which was pushed to device
    """
    if torch.is_tensor(data):
        return data.to(device=device, dtype=dtype, **kwargs)
    elif isinstance(data, Mapping):
        return {key: to_device_dtype(item, device=device, dtype=dtype, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device(item, device=device, dtype=dtype, **kwargs) for item in data])
    else:
        return data
