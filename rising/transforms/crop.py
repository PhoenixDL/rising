from typing import Sequence, Union
from rising.transforms.abstract import BaseTransform, RandomProcess
from rising.transforms.functional.crop import random_crop, center_crop

__all__ = ["CenterCrop", "RandomCrop", "CenterCropRandomSize", "RandomCropRandomSize"]


class CenterCrop(BaseTransform):
    """Crop from image center"""

    def __init__(self, size: Union[int, Sequence[int]],
                 keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Args:
            size: size of crop
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, size=size, keys=keys,
                         grad=grad, **kwargs)


class RandomCrop(BaseTransform):
    """Perform a random crop"""

    def __init__(self, size: Union[int, Sequence[int]],
                 dist: Union[int, Sequence[int]] = 0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            size: size of crop
            dist: minimum distance to border. By default zero
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_crop, size=size, dist=dist,
                         keys=keys, grad=grad, **kwargs)


class CenterCropRandomSize(RandomProcess, BaseTransform):
    """Crop a random sized part out of the image center"""

    def __init__(self, random_args: Union[Sequence, Sequence[Sequence]],
                 random_mode: str = "randrange", keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Args:
            random_args: positional arguments passed for random function.
                If Sequence[Sequence] is provided, a random value for each
                item in the outer. This can be used to set different ranges
                for different axis.
            random_mode: specifies distribution which should be used to
                sample additive value
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, random_mode=random_mode,
                         random_args=random_args, keys=keys, grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Augment data

        Args:
            **data: input batch

        Returns:
            dict: augmented data
        """
        self.kwargs["size"] = self.rand()
        return super().forward(**data)


class RandomCropRandomSize(RandomProcess, BaseTransform):
    """Crop a random sized part of the image out of a random location"""

    def __init__(self, random_args: Union[Sequence, Sequence[Sequence]],
                 random_mode: str = "randrange", dist: Union[int, Sequence[int]] = 0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Args:
            random_mode: specifies distribution which should be used to sample
                additive value
            random_args: positional arguments passed for random function.
                If Sequence[Sequence] is provided, a random value for each
                item in the outer. This can be used to set different ranges
                for different axis.
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_crop, random_mode=random_mode,
                         random_args=random_args, dist=dist,
                         keys=keys, grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Augment data

        Args:
            data: input batch

        Returns:
            dict: augmented data
        """
        self.kwargs["size"] = self.rand()
        return super().forward(**data)
