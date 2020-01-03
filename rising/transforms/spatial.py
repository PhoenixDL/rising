from __future__ import annotations

import torch
import random
from torch.multiprocessing import Value
from .abstract import RandomDimsTransform, AbstractTransform, BaseTransform, RandomProcess
from typing import Union, Sequence, Callable
from itertools import permutations

from .functional.spatial import *

__all__ = ["Mirror", "Rot90", "Resize",
           "Zoom", "ProgressiveResize", "SizeStepScheduler"]

schduler_type = Callable[[int], Union[int, Sequence[int]]]


class Mirror(RandomDimsTransform):
    def __init__(self, dims: Sequence, keys: Sequence = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Random mirror transform

        Parameters
        ----------
        dims: tuple
            axes which should be mirrored
        keys: tuple
            keys which should be mirrored
        prob: typing.Union[float, tuple]
            probability for mirror. If float value is provided, it is used
            for all dims
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass
        """
        super().__init__(augment_fn=mirror, dims=dims, keys=keys, prob=prob, grad=grad, **kwargs)


class Rot90(AbstractTransform):
    def __init__(self, dims: tuple, keys: tuple = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Randomly rotate 90 degree around dims

        Parameters
        ----------
        dims: tuple
            dims which should be rotated. If more than two dims are provided,
            two dimensions are randomly chosen at each call
        keys: tuple
            keys which should be rotated
        prob: typing.Union[float, tuple]
            probability for rotation
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass

        See Also
        --------
        :func:`torch.Tensor.rot90`
        """
        super().__init__(grad=grad, **kwargs)
        self.dims = dims
        self.keys = keys
        self.prob = prob

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Parameters
        ----------
        data: dict
            dict with tensors

        Returns
        -------
        dict
            dict with augmented data
        """
        if torch.rand(1) < self.prob:
            k = random.randrange(0, 4)
            rand_dims = self._permutations[random.randrange(0, len(self._permutations))]

            for key in self.keys:
                data[key] = rot90(data[key], k=k, dims=rand_dims)
        return data

    @property
    def dims(self) -> Sequence:
        """
        Number of dimensions
        """
        return self._dims

    @dims.setter
    def dims(self, dims: Sequence):
        """
        Set number of dimensions and compute new set of permutations

        Parameters
        ----------
        dims: Sequence
            number of dimensions
        """
        self._dims = dims
        self._permutations = tuple(permutations(dims, 2))


class Resize(BaseTransform):
    def __init__(self, size: Union[int, Sequence[int]], mode: str = 'nearest',
                 align_corners: bool = None, preserve_range: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Resize data to given size

        Parameters
        ----------
        size: Union[int, Sequence[int]]
            output size (with channel and batch dim)
        mode: str
            one of :param:`nearest`, :param:`linear`, :param:`bilinear`, :param:`bicubic`,
            :param:`trilinear`, :param:`area` (for more inforamtion see :func:`torch.nn.functional.interpolate`
        align_corners: bool
            input and output tensors are aligned by the center points of their corners pixels,
            preserving the values at the corner pixels.
        preserve_range: bool
            output tensor has same range as input tensor
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=resize, size=size, mode=mode,
                         align_corners=align_corners, preserve_range=preserve_range,
                         keys=keys, grad=grad, **kwargs)


class Zoom(RandomProcess, BaseTransform):
    def __init__(self, random_args: Union[Sequence, Sequence[Sequence]] = (0.75, 1.25),
                 random_mode: str = "uniform", mode: str = 'nearest',
                 align_corners: bool = None, preserve_range: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys. By default the scaling factor is sampled from a uniform
        distribution with the range specified by :param:`random_args`

        Parameters
        ----------
        random_args: Union[Sequence, Sequence[Sequence]]
            positional arguments passed for random function. If Sequence[Sequence]
            is provided, a random value for each item in the outer
            Sequence is generated. This can be
            used to set different ranges for different axis.
        random_mode: str
            specifies distribution which should be used to sample additive value
        mode: str
            one of :param:`nearest`, :param:`linear`, :param:`bilinear`, :param:`bicubic`,
            :param:`trilinear`, :param:`area` (for more inforamtion see :func:`torch.nn.functional.interpolate`)
        align_corners: bool
            input and output tensors are aligned by the center points of their corners pixels,
            preserving the values at the corner pixels.
        preserve_range: bool
            output tensor has same range as input tensor
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn

        See Also
        --------
        :func:`random.uniform`, :func:`torch.nn.functional.interpolate`
        """
        super().__init__(augment_fn=resize, random_args=random_args,
                         random_mode=random_mode, mode=mode,
                         align_corners=align_corners, preserve_range=preserve_range,
                         keys=keys, grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Augment data

        Parameters
        ----------
        data: dict
            input data

        Returns
        -------
        dict
            augmented data
        """
        self.kwargs["scale_factor"] = self.rand()
        return super().forward(**data)


class ProgressiveResize(Resize):
    def __init__(self, scheduler: schduler_type, mode: str = 'nearest',
                 align_corners: bool = None, preserve_range: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Resize data to sizes specified by scheduler

        Parameters
        ----------
        scheduler: Callable[[], Union[int, Sequence[int]]]
            scheduler which determined the current size. The scheduler is called
            with the current iteration of the transform
        mode: str
            one of :param:`nearest`, :param:`linear`, :param:`bilinear`, :param:`bicubic`,
            :param:`trilinear`, :param:`area` (for more inforamtion see :func:`torch.nn.functional.interpolate`
        align_corners: bool
            input and output tensors are aligned by the center points of their corners pixels,
            preserving the values at the corner pixels.
        preserve_range: bool
            output tensor has same range as input tensor
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn

        Warnings
        --------
        When this transformations is used in combination with multiprocessing
        the step counter is not perfectly synchronized between multiple
        processes. As a result the step count my jump between values
        in a range of the number of processes used.
        """
        super().__init__(size=0, mode=mode, align_corners=align_corners,
                         preserve_range=preserve_range,
                         keys=keys, grad=grad, **kwargs)
        self.scheduler = scheduler
        self._step = Value('i', 0)

    def reset_step(self) -> ProgressiveResize:
        """
        Reset step to 0

        Returns
        -------
        ProgressiveResize
            returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value = 0
        return self

    def increment(self) -> ProgressiveResize:
        """
        Increment step by 1

        Returns
        -------
        ProgressiveResize
            returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value += 1
        return self

    @property
    def step(self) -> int:
        """
        Current step

        Returns
        -------
        int
            number of steps
        """
        return self._step.value

    def forward(self, **data) -> dict:
        """
        Resize data

        Parameters
        ----------
        data: dict
            input batch

        Returns
        -------
        dict
            augmented batch
        """
        self.kwargs["size"] = self.scheduler(self.step)
        self.increment()
        return super().forward(**data)


class SizeStepScheduler:
    def __init__(self, milestones: Sequence[int],
                 sizes: Union[Sequence[int], Sequence[Sequence[int]]]):
        """
        Scheduler return size when milestone is reached

        Parameters
        ----------
        milestones: Sequence
            contains number of iterations where size should be changed
        sizes: Union[Sequence[int], Sequence[Sequence[int]]]
            sizes corresponding to milestones
        """
        if len(milestones) != len(sizes) - 1:
            raise TypeError("Sizes must include initial size and thus "
                            "has one element more than miltstones.")
        self.targets = sorted(zip((0, *milestones), sizes), key=lambda x: x[0], reverse=True)

    def __call__(self, step) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """
        Return size with regard to milestones

        Parameters
        ----------
        step: int
            current step

        Returns
        -------
        Union[Sequence[int], Sequence[Sequence[int]]]
            current size
        """
        for t in self.targets:
            if step >= t[0]:
                return t[1]
        return self.targets[-1][1]
