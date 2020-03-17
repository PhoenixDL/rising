import torch
import typing
import importlib
from typing import Callable, Union, Sequence, Any

from rising import AbstractMixin
from rising.utils import check_scalar
from rising.random import AbstractParameter, DiscreteParameter

__all__ = ["AbstractTransform", "BaseTransform", "PerSampleTransform",
           "PerChannelTransform"]

augment_callable = Callable[[torch.Tensor], Any]
augment_axis_callable = Callable[[torch.Tensor, Union[float, Sequence]], Any]


class AbstractTransform(torch.nn.Module):
    def __init__(self, grad: bool = False, **kwargs):
        """
        Base class for all transforms

        Parameters
        ----------
        grad: bool
            enable gradient computation inside transformation
        """
        super().__init__()
        self.grad = grad
        for key, item in kwargs.items():
            setattr(self, key, item)

    def register_sampler(self, name: str, sampler: AbstractParameter,
                         *args, **kwargs):
        """
        Registers a parameter sampler to the transform.
        Internally a property is created to forward calls to the attribute to
        calls of the sampler.

        Parameters
        ----------
        name : str
            the property name
        sampler : AbstractParameter
            the sampler. Will be wrapped to a sampler always returning the
            same element if not already a sampler
        *args :
            additional positional arguments (will be forwarded to sampler call)
        **kwargs :
            additional keyword arguments (will be forwarded to sampler call)

        """
        if not isinstance(sampler, AbstractParameter):
            sampler = DiscreteParameter([sampler], replacement=True)

        # value_name = '_' + name
        # while hasattr(self, value_name):
        #     value_name = '_' + value_name
        #
        # setattr(self, value_name, sampler)

        if hasattr(self, name):
            raise NameError('Name %s already exists' % name)
        setattr(self, name, property(
            lambda self: sampler(*args, **kwargs)))

    def __call__(self, *args, **kwargs) -> typing.Any:
        """
        Call super class with correct torch context

        Parameters
        ----------
        args:
            forwarded positional arguments
        kwargs:
            forwarded keyword arguments

        Returns
        -------

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            return super().__call__(*args, **kwargs)

    def forward(self, **data) -> dict:
        """
        Implement transform functionality here

        Parameters
        ----------
        data: dict
            dict with data

        Returns
        -------
        dict
            dict with transformed data
        """
        raise NotImplementedError


class BaseTransform(AbstractTransform):
    def __init__(self, augment_fn: augment_callable, *args,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad)
        self.augment_fn = augment_fn
        self.keys = keys
        self.args = args
        self.kwargs = kwargs

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
        for _key in self.keys:
            data[_key] = self.augment_fn(data[_key], *self.args, **self.kwargs)
        return data


class PerSampleTransform(BaseTransform):
    def forward(self, **data) -> dict:
        """
        Apply transformation to each sample in batch individually
        :param:`augment_fn` must be callable with option :param:`out`
        where results are saved in

        Parameters
        ----------
        data: dict
            dict with tensors

        Returns
        -------
        dict
            dict with augmented data
        """
        for _key in self.keys:
            out = torch.empty_like(data[_key])
            for _i in range(data[_key].shape[0]):
                out[_i] = self.augment_fn(data[_key][_i], out=out[_i], **self.kwargs)
            data[_key] = out
        return data


class PerChannelTransform(BaseTransform):
    def __init__(self, augment_fn: augment_callable, per_channel: bool = False,
                 keys: Sequence = ('data',), grad: bool = False, **kwargs):
        """
        Apply transformation per channel (but still to whole batch)

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        per_channel: bool
            enable transformation per channel
        keys: Sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=augment_fn, keys=keys, grad=grad, **kwargs)
        self.per_channel = per_channel

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
        if self.per_channel:
            for _key in self.keys:
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    out[:, _i] = self.augment_fn(data[_key][:, _i], out=out[:, _i], **self.kwargs)
                data[_key] = out
            return data
        else:
            return super().forward(**data)
