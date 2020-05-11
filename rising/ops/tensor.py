from typing import Optional

import numpy as np
import torch


def torch_one_hot(target: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Compute one hot encoding of input tensor

    Args:
        target: tensor to be converted
        num_classes: number of classes. If :attr:`num_classes` is None,
            the maximum of target is used

    Returns:
        torch.Tensor: one hot encoded tensor
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def np_one_hot(target: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute one hot encoding of input array

    Args:
        target: array to be converted
        num_classes: number of classes

    Returns:
        numpy.ndarray: one hot encoded array
    """
    if num_classes is None:
        num_classes = int(target.max().item() + 1)
    dtype = target.dtype
    target_onehot = np.zeros((*target.shape, num_classes), dtype=dtype)
    for c in range(num_classes):
        target_onehot[..., c] = target == c
    return target_onehot
