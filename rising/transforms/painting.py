import torch
from typing import Sequence

from rising.transforms.abstract import AbstractTransform, BaseTransform
from rising.transforms.functional.painting import (
    local_pixel_shuffle, random_inpainting, random_outpainting
)


__all__ = ["RandomInpainting", "RandomOutpainting", "RandomInOrOutpainting", "LocalPixelShuffle"]


class LocalPixelShuffle(BaseTransform):
    """ Shuffels Pixels locally in n patches,
    as proposed in Models Genesis """

    def __init__(self, n: int=-1, block_size: tuple=(0,0,0), rel_block_size: float = 0.1,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            n: number of local patches to shuffle, default = 1000*channels
            block_size: size of local patches in pixel
            rel_block_size: size of local patches in relation to image size, e.g. image_size=(32,192,192) and rel_block_size=0.25 will result in patches of size (8, 48, 48). If rel_block_size > 0, it will overwrite block_size.
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=local_pixel_shuffle, n=n, block_size=block_size, rel_block_size=rel_block_size,
                         keys=keys, grad=grad, **kwargs)


class RandomInpainting(BaseTransform):
    """ In n local areas, the image is replaced by uniform noise in range (minv, maxv),
    as proposed in Models Genesis """

    def __init__(self, n: int = 5,
                 maxv: float=1.0, minv: float = 0.0,
                 max_size: tuple = (0,0,0), min_size: tuple = (0,0,0), rel_max_size: tuple = (0.25, 0.25, 0.25), rel_min_size: tuple = (0.1, 0.1, 0.1), min_border_distance: tuple = (3, 3, 3),
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            minv, maxv: range of uniform noise
            n: number of local patches to randomize
            max_size: absolute upper bound for the patch size 
            min_size: absolute lower bound for the patch size
            rel_max_size: relative upper bound for the patch size, relative to image_size. Overwrites max_size.
            rel_min_size: relative lower bound for the patch size, relative to image_size. Overwrites min_size.
            min_border_distance: the minimum distance of patches to the border in pixel for each dimension.
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_inpainting, n=n, maxv=maxv, minv=minv, max_size=max_size, min_size=min_size, rel_max_size=rel_max_size, rel_min_size=rel_min_size,
                         keys=keys, grad=grad, **kwargs)


class RandomOutpainting(AbstractTransform):
    """ The border of the images will be replaced by uniform noise,
    as proposed in Models Genesis. (Replaces a patch in an equally sized noise image with the corresponding input image content) """

    def __init__(self, prob: float = 0.5, maxv: float=1.0, minv: float = 0.0,
                 max_size: tuple = (0,0,0), min_size: tuple = (0,0,0), 
                 rel_max_size: tuple = (6/7, 6/7, 6/7), rel_min_size: tuple = (5/7, 5/7, 5/7), min_border_distance: tuple = (3, 3, 3),
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            minv, maxv: range of uniform noise
            prob: probability of outpainting. For prob<1.0, not all images will be augmented
            max_size: absolute upper bound for the patch size. Here the patch is the remaining image
            min_size: absolute lower bound for the patch size. Here the patch is the remaining image
            rel_max_size: relative upper bound for the patch size, relative to image_size. Overwrites max_size.
            rel_min_size: relative lower bound for the patch size, relative to image_size. Overwrites min_size.
            min_border_distance: the minimum thickness of the border in pixel for each dimension.
            keys: the keys corresponding to the values to distort
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad, **kwargs)
        self.prob = prob
        self.maxv = maxv
        self.minv = minv
        self.keys = keys
        self.max_size = max_size
        self.min_size = min_size
        self.rel_min_size = rel_min_size
        self.rel_max_size = rel_max_size
        self.min_border_distance = min_border_distance

    def forward(self, **data) -> dict:
        if torch.rand(1) < self.prob:
            for key in self.keys:
                data[key] = random_outpainting(data[key], maxv=self.maxv, minv=self.minv, max_size=self.max_size, min_size=self.min_size, rel_max_size=self.rel_max_size, rel_min_size=self.rel_min_size,
                min_border_distance=self.min_border_distance)
        return data


class RandomInOrOutpainting(AbstractTransform):
    """Applies either random inpainting or random outpainting to the image,
    as proposed in Models Genesis """

    def __init__(self, prob: float = 0.5, n: int = 5,
                 maxv: float=1.0, minv: float = 0.0,
                 max_size_in: tuple = (0,0,0), min_size_in: tuple = (0,0,0), rel_max_size_in: tuple = (0.25, 0.25, 0.25), rel_min_size_in: tuple = (0.1, 0.1, 0.1),
                 max_size_out: tuple = (0,0,0), min_size_out: tuple = (0,0,0), 
                 rel_max_size_out: tuple = (6/7, 6/7, 6/7), rel_min_size_out: tuple = (5/7, 5/7, 5/7),
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
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
        self.max_size_in = max_size_in
        self.min_size_in = min_size_in
        self.rel_min_size_in = rel_min_size_in
        self.rel_max_size_in = rel_max_size_in
        self.max_size_out = max_size_out
        self.min_size_out = min_size_out
        self.rel_min_size_out = rel_min_size_out
        self.rel_max_size_out = rel_max_size_out

    def forward(self, **data) -> dict:
        if torch.rand(1) < self.prob:
            for key in self.keys:
                data[key] = random_outpainting(data[key], maxv=self.maxv, minv=self.minv, max_size=self.max_size_out, min_size=self.min_size_out, rel_max_size=self.rel_max_size_out, rel_min_size=self.rel_min_size_out)
        else:
            for key in self.keys:
                data[key] = random_inpainting(data[key], n=self.n, maxv=self.maxv, minv=self.minv, max_size = self.max_size_in, min_size=self.min_size_in, rel_max_size=self.rel_max_size_in, rel_min_size=self.rel_min_size_in)
        return data