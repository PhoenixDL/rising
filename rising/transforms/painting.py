from typing import Sequence

import torch

from rising.transforms.abstract import AbstractTransform, BaseTransform
from rising.transforms.functional.painting import local_pixel_shuffle, random_inpainting, random_outpainting

__all__ = ["RandomInpainting", "RandomOutpainting", "RandomInOrOutpainting", "LocalPixelShuffle"]


class LocalPixelShuffle(BaseTransform):
    """Shuffels Pixels locally in n patches,
    as proposed in Models Genesis"""

    def __init__(self, n: int = -1, keys: Sequence = ("data",), grad: bool = False, **kwargs):
        """
        Args:
            n: number of local patches to shuffle, default = 1000*channels
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=local_pixel_shuffle, n=n, keys=keys, grad=grad, **kwargs)


class RandomInpainting(BaseTransform):
    """In n local areas, the image is replaced by uniform noise in range (minv, maxv),
    as proposed in Models Genesis"""

    def __init__(
        self, n: int = 5, maxv: float = 1.0, minv: float = 0.0, keys: Sequence = ("data",), grad: bool = False, **kwargs
    ):
        """
        Args:
            minv, maxv: range of uniform noise
            n: number of local patches to randomize
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_inpainting, n=n, maxv=maxv, minv=minv, keys=keys, grad=grad, **kwargs)


class RandomOutpainting(AbstractTransform):
    """The border of the images will be replaced by uniform noise,
    as proposed in Models Genesis"""

    def __init__(
        self,
        prob: float = 0.5,
        maxv: float = 1.0,
        minv: float = 0.0,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            minv, maxv: range of uniform noise
            prob: probability of outpainting. For prob<1.0, not all images will be augmented
            keys: the keys corresponding to the values to distort
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


class RandomInOrOutpainting(AbstractTransform):
    """Applies either random inpainting or random outpainting to the image,
    as proposed in Models Genesis"""

    def __init__(
        self,
        prob: float = 0.5,
        n: int = 5,
        maxv: float = 1.0,
        minv: float = 0.0,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            minv, maxv: range of uniform noise
            prob: probability of outpainting, probability of inpainting is 1-prob.
            n: number of local patches to randomize in case of inpainting
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad, **kwargs)
        self.prob = prob
        self.maxv = maxv
        self.minv = minv
        self.keys = keys
        self.n = n

    def forward(self, **data) -> dict:
        if torch.rand(1) < self.prob:
            for key in self.keys:
                data[key] = random_outpainting(data[key], maxv=self.maxv, minv=self.minv)
        else:
            for key in self.keys:
                data[key] = random_inpainting(data[key], n=self.n, maxv=self.maxv, minv=self.minv)
        return data
