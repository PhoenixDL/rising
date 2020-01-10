from typing import Sequence

from rising.transforms import BaseTransform
from rising.transforms.functional import one_hot_batch


__all__ = ["OneHot"]


class OneHot(BaseTransform):
    def __init__(self, num_classes: int, keys: Sequence = ('seg',), grad: bool = False, **kwargs):
        """
        Convert to one hot encoding. One hot encoding is applied in first dimension
        which results in shape N x NumClasses x [same as input] while input is expected to
        have shape N x 1 x [arbitrary additional dimensions]

        Parameters
        ----------
        num_classes: int
            number of classes. If :param:`num_classes` is None, the number of classes
            is automatically determined from the current batch (by using the max
            of the current batch and assuming a consecutive order from zero)
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to :func:`one_hot_batch`
        """
        super().__init__(augment_fn=one_hot_batch, keys=keys, grad=grad, num_classes=num_classes,
                         **kwargs)