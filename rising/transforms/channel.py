from typing import Optional, Sequence

import torch

from rising.transforms import BaseTransform
from rising.transforms.functional import one_hot_batch

__all__ = ["OneHot", "ArgMax"]


class OneHot(BaseTransform):
    """
    Convert to one hot encoding. One hot encoding is applied in first dimension
    which results in shape N x NumClasses x [same as input] while input is expected to
    have shape N x 1 x [arbitrary additional dimensions]
    """

    def __init__(
        self,
        num_classes: int,
        keys: Sequence = ("seg",),
        dtype: Optional[torch.dtype] = None,
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            num_classes: number of classes. If :attr:`num_classes` is None,
                the number of classes is automatically determined from the
                current batch (by using the max of the current batch and
                assuming a consecutive order from zero)
            dtype: optionally changes the dtype of the onehot encoding
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to :func:`one_hot_batch`

        Warnings:
            Input tensor needs to be of type torch.long. This could
            be achieved by applying `TenorOp("long", keys=("seg",))`.
        """
        super().__init__(augment_fn=one_hot_batch, keys=keys, grad=grad, num_classes=num_classes, dtype=dtype, **kwargs)


class ArgMax(BaseTransform):
    """
    Compute argmax along given dimension.
    Can be used to revert OneHot encoding.
    """

    def __init__(self, dim: int, keepdim: bool = True, keys: Sequence = ("seg",), grad: bool = False, **kwargs):
        """
        Args:
            dim: dimension to apply argmax
            keepdim: whether the output tensor has dim retained or not
            dtype: optionally changes the dtype of the onehot encoding
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to :func:`one_hot_batch`

        Warnings
            The output of the argmax function is always a tensor of dtype long.
        """
        super().__init__(augment_fn=torch.argmax, keys=keys, grad=grad, dim=dim, keepdim=keepdim, **kwargs)
