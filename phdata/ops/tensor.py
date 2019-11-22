import torch
import numpy as np


def torch_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute one hot encoding of input tensor

    Parameters
    ----------
    target: torch.Tensor
        tensor to be converted
    num_classes: int
        number of classes

    Returns
    -------
    torch.Tensor
        one hot encoded tensor
    """
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes,
                                dtype=dtype, device=device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    return target_onehot.to(dtype=dtype, device=device)


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
