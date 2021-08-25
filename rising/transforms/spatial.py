# from __future__ import annotations
from itertools import combinations
from typing import Callable, Optional, Sequence, Union

import torch
from torch.multiprocessing import Value

from rising.random import AbstractParameter, DiscreteParameter
from rising.transforms.abstract import AbstractTransform, BaseTransform

__all__ = ["Mirror", "Rot90", "ResizeNative", "Zoom", "ProgressiveResize", "SizeStepScheduler"]

from rising.transforms.functional import mirror, resize_native, rot90

scheduler_type = Callable[[int], Union[int, Sequence[int]]]


class Mirror(AbstractTransform):
    """Random mirror transform"""

    def __init__(
        self,
        dims: Union[int, DiscreteParameter, Sequence[Union[int, DiscreteParameter]]],
        keys: Sequence[str] = ("data",),
        prob: float = 0.5,
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            dims: axes which should be mirrored
            keys: keys which should be mirrored
            prob: probability for mirror. If float value is provided,
                it is used for all dims
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass

        Examples:
            >>> # Use mirror transform for augmentations
            >>> from rising.random import DiscreteCombinationsParameter
            >>> # We sample from all possible mirror combination for
            >>> # volumetric data
            >>> trafo = Mirror(DiscreteCombinationsParameter((0, 1, 2)))
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys
        self.prob = prob
        if not isinstance(dims, DiscreteParameter):
            if len(dims) > 2:
                dims = list(combinations(dims, 2))
            else:
                dims = (dims,)
            dims = DiscreteParameter(dims)
        self.register_sampler("dims", dims)

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors
        Returns:
            dict: dict with augmented data
        """
        if torch.rand(1) < self.prob:
            for key in self.keys:
                data[key] = mirror(data[key], self.dims)
        return data


class Rot90(AbstractTransform):
    """Rotate 90 degree around dims"""

    def __init__(
        self,
        dims: Union[Sequence[int], DiscreteParameter],
        keys: Sequence[str] = ("data",),
        num_rots: Sequence[int] = (0, 1, 2, 3),
        prob: float = 0.5,
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            dims: dims/axis ro rotate. If more than two dims are
                provided, 2 dimensions are randomly chosen at each call
            keys: keys which should be rotated
            num_rots: possible values for number of rotations
            prob: probability for rotation
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.Tensor.rot90`
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys
        self.prob = prob
        if not isinstance(dims, DiscreteParameter):
            if len(dims) > 2:
                dims = list(combinations(dims, 2))
            else:
                dims = (dims,)
            dims = DiscreteParameter(dims)
        self.register_sampler("dims", dims)
        self.register_sampler("num_rots", DiscreteParameter(num_rots))

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if torch.rand(1) < self.prob:
            num_rots = self.num_rots
            rand_dims = self.dims

            for key in self.keys:
                data[key] = rot90(data[key], k=num_rots, dims=rand_dims)
        return data


class ResizeNative(BaseTransform):
    """Resize data to given size"""

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        preserve_range: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            size: spatial output size (excluding batch size and
                number of channels)
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                ``trilinear``, ``area`` (for more inforamtion see
                :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center \
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(
            augment_fn=resize_native,
            size=size,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            keys=keys,
            grad=grad,
            **kwargs
        )


class Zoom(BaseTransform):
    """Apply augment_fn to keys. By default the scaling factor is sampled
    from a uniform distribution with the range specified by
    :attr:`random_args`
    """

    def __init__(
        self,
        scale_factor: Union[Sequence, AbstractParameter] = (0.75, 1.25),
        mode: str = "nearest",
        align_corners: bool = None,
        preserve_range: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            scale_factor: positional arguments passed for random function.
                If Sequence[Sequence] is provided, a random value for each item
                in the outer Sequence is generated. This can be used to set
                different ranges for different axis.
            mode: one of `nearest`, `linear`, `bilinear`,
                `bicubic`, `trilinear`, `area` (for more
                inforamtion see :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        See Also:
            :func:`random.uniform`, :func:`torch.nn.functional.interpolate`
        """
        super().__init__(
            augment_fn=resize_native,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            keys=keys,
            grad=grad,
            property_names=("scale_factor",),
            **kwargs
        )


class ProgressiveResize(ResizeNative):
    """Resize data to sizes specified by scheduler"""

    def __init__(
        self,
        scheduler: scheduler_type,
        mode: str = "nearest",
        align_corners: bool = None,
        preserve_range: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            scheduler: scheduler which determined the current size.
                The scheduler is called with the current iteration of the
                transform
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                    ``trilinear``, ``area`` (for more inforamtion see
                    :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        Warnings:
            When this transformations is used in combination with
            multiprocessing, the step counter is not perfectly synchronized
            between multiple processes.
            As a result the step count my jump between values
            in a range of the number of processes used.
        """
        super().__init__(
            size=0,
            mode=mode,
            align_corners=align_corners,
            preserve_range=preserve_range,
            keys=keys,
            grad=grad,
            **kwargs
        )
        self.scheduler = scheduler
        self._step = Value("i", 0)

    def reset_step(self) -> ResizeNative:
        """
        Reset step to 0

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value = 0
        return self

    def increment(self) -> ResizeNative:
        """
        Increment step by 1

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value += 1
        return self

    @property
    def step(self) -> int:
        """
        Current step

        Returns:
            int: number of steps
        """
        return self._step.value

    def forward(self, **data) -> dict:
        """
        Resize data

        Args:
            **data: input batch

        Returns:
            dict: augmented batch
        """
        self.kwargs["size"] = self.scheduler(self.step)
        self.increment()
        return super().forward(**data)


class SizeStepScheduler:
    """Scheduler return size when milestone is reached"""

    def __init__(self, milestones: Sequence[int], sizes: Union[Sequence[int], Sequence[Sequence[int]]]):
        """
        Args:
            milestones: contains number of iterations where size should be changed
            sizes: sizes corresponding to milestones
        """
        if len(milestones) != len(sizes) - 1:
            raise TypeError("Sizes must include initial size and thus " "has one element more than miltstones.")
        self.targets = sorted(zip((0, *milestones), sizes), key=lambda x: x[0], reverse=True)

    def __call__(self, step) -> Union[int, Sequence[int], Sequence[Sequence[int]]]:
        """
        Return size with regard to milestones

        Args:
            step: current step

        Returns:
            Union[int, Sequence[int], Sequence[Sequence[int]]]: current size
        """
        for t in self.targets:
            if step >= t[0]:
                return t[1]
        return self.targets[-1][1]
