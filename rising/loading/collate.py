import collections.abc
from typing import Any

import numpy as np
import torch

__all__ = ["numpy_collate", "do_nothing_collate"]

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
)


def numpy_collate(batch: Any) -> Any:
    """
    function to collate the samples to a whole batch of numpy arrays.
    PyTorch Tensors, scalar values and sequences will be casted to arrays
    automatically.

    Args:
        batch: a batch of samples. In most cases either sequence,
            mapping or mixture of them

    Returns:
        Any: collated batch with optionally converted type
            (to :class:`numpy.ndarray`)

    Raises:
        TypeError: When batch could not be collated automatically
    """

    elem = batch[0]
    if isinstance(elem, np.ndarray):
        return np.stack(batch, 0)
    elif isinstance(elem, torch.Tensor):
        return numpy_collate([b.detach().cpu().numpy() for b in batch])
    elif isinstance(elem, float) or isinstance(elem, int):
        return np.array(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return type(elem)(*(numpy_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(type(elem)))


def do_nothing_collate(batch: Any) -> Any:
    """
    Returns the batch as is (with out any collation
    Args:
        batch: input batch (typically a sequence, mapping or mixture of those).

    Returns:
        Any: the batch as given to this function

    """

    return batch
