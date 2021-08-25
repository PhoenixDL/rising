from typing import Optional

import torch

from rising.ops import torch_one_hot

__all__ = ["one_hot_batch"]


def one_hot_batch(
    target: torch.Tensor, num_classes: Optional[int] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Compute one hot for input tensor (assumed to a be batch and thus saved
    into first dimension -> input should only have one channel)

    Args:
        target: long tensor to be converted
        num_classes: number of classes.
            If :attr:`num_classes` is None, the maximum of target is used
        dtype: optionally changes the dtype of the onehot encoding

    Returns:
        torch.Tensor: one hot encoded tensor
    """
    if target.dtype != torch.long:
        raise TypeError(f"Target tensor needs to be of type torch.long, found {target.dtype}")

    if target.ndim in [0, 1]:
        return torch_one_hot(target, num_classes)
    else:
        if num_classes is None:
            num_classes = int(target.max().detach().item() + 1)
        _dtype, device, shape = target.dtype, target.device, target.shape
        if dtype is None:
            dtype = _dtype
        target_onehot = torch.zeros(shape[0], num_classes, *shape[2:], dtype=dtype, device=device)
        return target_onehot.scatter_(1, target, 1.0)
