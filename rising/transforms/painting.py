import torch
from typing import Union, Sequence, Optional

from rising.transforms.abstract import AbstractTransform, BaseTransform
from rising.transforms.functional.painting import (
    local_pixel_shuffle, random_inpainting, random_outpainting
)

from rising.random import AbstractParameter

__all__ = ["RandomInpainting", "RandomOutpainting", "LocalPixelShuffle"]


class LocalPixelShuffle(BaseTransform):
    """Apply augment_fn to keys"""

    def __init__(self, n: int=-1,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """

        """
        super().__init__(augment_fn=local_pixel_shuffle, n=n,
                         keys=keys, grad=grad, **kwargs)


class RandomInpainting(BaseTransform):
    """Apply augment_fn to keys"""

    def __init__(self, n: int = 5,
                 maxv: float=1.0, minv: float = 0.0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_inpainting, n=n, maxv=maxv, minv=minv,
                         keys=keys, grad=grad, **kwargs)


class RandomOutpainting(AbstractTransform):
    """Apply augment_fn to keys"""

    def __init__(self, prob: float = 0.5, maxv: float=1.0, minv: float = 0.0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad, **kwargs)
        self.prob = prob
        self.maxv = maxv
        self.minv = minv
        self.keys = keys

    def forward(self, **data) -> dict:
        if torch.rand(1) < self.prob:
            for key in self.keys:
                data[key] = random_outpainting(data[key], maxv=self.maxv, minv=self.minv)
        return data