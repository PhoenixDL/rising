import torch
from rising.transforms.abstract import AbstractTransform


def chech_data_preservation(trafo: AbstractTransform, batch: dict, key: str = 'data') -> bool:
    """
    Checks for inplace modification of input data

    Parameters
    ----------
    trafo: AbstractTransform
        transforamtion
    batch: dict
        batch with torch.Tensor data
    key: str
        key to examine

    Returns
    -------
    bool
        true if data did not change inplace
    """
    orig = batch[key].clone()
    _ = trafo(**batch)
    return (orig == batch[key]).all()
