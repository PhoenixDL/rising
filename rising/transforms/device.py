import torch
from typing import Union, Sequence
from rising.transforms import BaseTransform
from rising.transforms.functional import to_device


class ToDevice(BaseTransform):
    def __init__(self, device: Union[torch.device, str],
                 non_blocking: bool = False, copy: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        super().__init__(augment_fn=to_device, keys=keys, grad=grad, device=device,
                         non_blocking=non_blocking, copy=copy, **kwargs)
