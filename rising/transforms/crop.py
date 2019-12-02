from typing import Sequence, Union
from rising.transforms.abstract import BaseTransform, RandomProcess
from rising.transforms.functional.crop import random_crop, center_crop

__all__ = ["CenterCrop", "RandomCrop", "CenterCropRandomSize", "RandomCropRandomSize"]


class CenterCrop(BaseTransform):
    def __init__(self, size: Union[int, Sequence[int]], keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        size: Union[int, Sequence[int]]
            size of crop
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, size=size, keys=keys,
                         grad=grad, **kwargs)


class RandomCrop(BaseTransform):
    def __init__(self, size: Union[int, Sequence[int]], dist: Union[int, Sequence[int]] = 0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        size: Union[int, Sequence[int]]
            size of crop
        dist: Union[int, Sequence[int]]
            minimum distance to border. By default zero
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_crop, size=size, dist=dist,
                         keys=keys, grad=grad, **kwargs)


class CenterCropRandomSize(RandomProcess, BaseTransform):
    def __init__(self, random_args: Union[Sequence, Sequence[Sequence]],
                 random_mode: str = "randrange", keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        random_args: Union[Sequence, Sequence[Sequence]]
            positional arguments passed for random function. If Sequence[Sequence]
            is provided, a random value for each item in the outer. This can be
            used to set different ranges for different axis.
        random_mode: str
            specifies distribution which should be used to sample additive value
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, random_mode=random_mode,
                         random_args=random_args, keys=keys, grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Augment data

        Parameters
        ----------
        data: dict
            input batch

        Returns
        -------
        dict
            augmented data
        """
        self.kwargs["size"] = self.rand()
        return super().forward(**data)


class RandomCropRandomSize(RandomProcess, BaseTransform):
    def __init__(self, random_args: Union[Sequence, Sequence[Sequence]],
                 random_mode: str = "randrange", dist: Union[int, Sequence[int]] = 0,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        random_mode: str
            specifies distribution which should be used to sample additive value
        random_args: Union[Sequence, Sequence[Sequence]]
            positional arguments passed for random function. If Sequence[Sequence]
            is provided, a random value for each item in the outer. This can be
            used to set different ranges for different axis.
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_crop, random_mode=random_mode,
                         random_args=random_args, dist=dist,
                         keys=keys, grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Augment data

        Parameters
        ----------
        data: dict
            input batch

        Returns
        -------
        dict
            augmented data
        """
        self.kwargs["size"] = self.rand()
        return super().forward(**data)
