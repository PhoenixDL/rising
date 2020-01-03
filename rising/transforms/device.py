import torch
from typing import Union, Sequence
from rising.transforms import BaseTransform
from rising.transforms.functional import to_device

__all__ = ["ToDevice"]


class ToDevice(BaseTransform):
    def __init__(self, device: Union[torch.device, str],
                 non_blocking: bool = False, copy: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Pushed data to device

        Parameters
        ----------
        device: Union[torch.device, str]
            target device
        non_blocking: bool
            if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        copy: bool
            create copy of data
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to function
        """
        super().__init__(augment_fn=to_device, keys=keys, grad=grad, device=device,
                         non_blocking=non_blocking, copy=copy, **kwargs)
