import torch


def check_scalar(x):
    """
    Provide interface to check for scalars

    Args:
        x: object to check for scalar

    Returns:
        True if input is scalar
    """
    return isinstance(x, (int, float)) or (is instance(x, torch.Tensor) and x.numel() == 1)
        return True
    elif torch.is_tensor(x):
        return x.numel() == 1
    else:
        return False
