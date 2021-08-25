from rising.transforms.abstract import AbstractTransform


def chech_data_preservation(trafo: AbstractTransform, batch: dict, key: str = "data") -> bool:
    """
    Checks for inplace modification of input data

    Args:
        trafo: transforamtion
        batch: batch with torch.Tensor data
        key: key to examine

    Returns:
        bool: true if data did not change inplace
    """
    orig = batch[key].clone()
    _ = trafo(**batch)
    return (orig == batch[key]).all()
