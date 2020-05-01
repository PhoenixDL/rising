import torch
from typing import Dict, Union, Sequence
from torch.utils.data._utils.collate import default_convert

from rising.transforms import AbstractTransform, BaseTransform
from rising.transforms.functional import tensor_op, to_device


__all__ = ["ToTensor", "ToDevice", "TensorOp", "Permute"]


class ToTensor(BaseTransform):
    """Transform Input Collection to Collection of :class:`torch.Tensor`"""
    def __init__(self, keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            keys: keys which should be transformed
        grad: enable gradient computation inside transformation
        **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=default_convert, keys=keys, grad=grad, **kwargs)


class ToDevice(BaseTransform):
    """Push data to device"""
    def __init__(self, device: Union[torch.device, str],
                 non_blocking: bool = False, copy: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            device: target device
            non_blocking: if True and this copy is between CPU and GPU,
                the copy may occur asynchronously with respect to the host.
                For other cases, this argument has no effect.
            copy: create copy of data
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to function
        """
        super().__init__(augment_fn=to_device, keys=keys, grad=grad, device=device,
                         non_blocking=non_blocking, copy=copy, **kwargs)


class TensorOp(BaseTransform):
    """Apply function which are supported by the `torch.Tensor` class"""
    def __init__(self, op_name: str, *args, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Args:
            op_name: name of tensor operation
            *args: positional arguments passed to function
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to function
        """
        super().__init__(tensor_op, op_name, *args, keys=keys, grad=grad, **kwargs)


class Permute(BaseTransform):
    """Permute dimensions of tensor"""
    def __init__(self, dims: Dict[str, Sequence[int]], grad: bool = False, **kwargs):
        """
        Args:
            dims: defines permutation sequence for respective key
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to permute function
        """
        super().__init__(grad=grad)
        self.dims = dims
        self.kwargs = kwargs

    def forward(self, **data) -> dict:
        """
        Forward input

        Args:
        data: batch dict

        Returns:
            augmented data
        """
        for key, item in self.dims.items():
            data[key] = tensor_op(data[key], "permute", *item, **self.kwargs)
        return data
