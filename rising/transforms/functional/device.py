import torch
from typing import Iterable, Union, Mapping

__all__ = ["to_device"]

data_type = Union[torch.Tensor, Iterable[torch.Tensor]]


def to_device(data: data_type, device: Union[torch.device, str], **kwargs) -> data_type:
    if torch.is_tensor(data):
        return data.to(device=device, **kwargs)
    elif isinstance(data, Mapping):
        return {key: to_device(item, device, **kwargs) for key, item in data.items()}
    else:
        return type(data)([to_device(item, device, **kwargs) for item in data])
