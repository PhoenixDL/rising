import torch
from rising.ops import torch_one_hot

__all__ = ["one_hot_batch"]


def one_hot_batch(target: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """
    Compute one hot for input tensor (assumed to a be batch and thus saved
    into first dimension -> input should only have one channel)

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
    if target.ndim in [0, 1]:
        return torch_one_hot(target, num_classes)
    else:
        if num_classes is None:
            num_classes = int(target.max().detach().item() + 1)
        dtype, device, shape = target.dtype, target.device, target.shape
        target_onehot = torch.zeros(shape[0], num_classes, *shape[2:],
                                    dtype=dtype, device=device)
        return target_onehot.scatter_(1, target, 1.0)
