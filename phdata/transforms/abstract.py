import torch
import typing
from typing import Callable, Union, Sequence, Any

from phdata.utils import check_scalar


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
    def __init__(self, augment_fn: augment_callable, keys: Sequence = ('data',),
                 grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        dims: tuple
            axes which should be mirrored
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad)
        self.augment_fn = augment_fn
        self.keys = keys
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
            data[_key] = self.augment_fn(data[_key], **self.kwargs)
        return data


class PerSampleTransform(BaseTransform):
    def forward(self, **data) -> dict:
        """
        Apply transformation to each sample in batch individually

        Parameters
        ----------
        data: dict
            dict with tensors

        Returns
        -------
        dict
            dict with augmented data
        """
        # TODO: check if data must be cloned first
        for _key in self.keys:
            for _i in range(data[_key].shape[0]):
                data[_key][_i] = self.augment_fn(data[_key][_i], **self.kwargs)
        return data


class RandomDimsTransform(AbstractTransform):
    def __init__(self, augment_fn: augment_axis_callable, dims: Sequence, keys: Sequence = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Randomly choose axes to perform augmentation on

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        dims: tuple
            possible axes
        keys: tuple
            keys which should be mirrored
        prob: typing.Union[float, tuple]
            probability for mirror. If float value is provided, it is used
            for all dims
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to augment_fn
        """
        super().__init__(grad=grad)
        self.augment_fn = augment_fn
        self.dims = dims
        self.keys = keys
        if check_scalar(prob):
            prob = (prob,) * len(dims)
        self.prob = prob
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
        rand_val = torch.rand(len(self.dims), requires_grad=False)
        dims = [_dim for _dim, _prob in zip(self.dims, self.prob) if rand_val[_dim] < _prob]

        for key in self.keys:
            data[key] = self.augment_fn(data[key], dims=dims, **self.kwargs)
        return data
