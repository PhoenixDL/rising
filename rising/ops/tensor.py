import torch
import numpy as np


def torch_one_hot(target: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """
    Compute one hot encoding of input tensor

    Parameters
    ----------
    target: torch.Tensor
        tensor to be converted
    num_classes: int
        number of classes. If :param:`num_classes` is None, the maximum of target is used

    Returns
    -------
    torch.Tensor
        one hot encoded tensor
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def np_one_hot(target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute one hot encoding of input array

    Parameters
    ----------
    target: np.ndarray
        array to be converted
    num_classes: int
        number of classes

    Returns
    -------
    np.ndarray
        one hot encoded array
    """
    dtype = target.dtype
    target_onehot = np.zeros((*target.shape, num_classes), dtype=dtype)
    for c in range(num_classes):
        target_onehot[..., c] = target == c
    return target_onehot
