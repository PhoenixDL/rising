import torch
import typing
import importlib
from typing import Callable, Union, Sequence, Any

from rising import AbstractMixin
from rising.utils import check_scalar

__all__ = ["AbstractTransform", "BaseTransform", "PerSampleTransform",
           "PerChannelTransform", "RandomDimsTransform", "RandomProcess"]


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
            keys which should be augmented
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

        if dims:
            for key in self.keys:
                data[key] = self.augment_fn(data[key], dims=dims, **self.kwargs)
        return data


class RandomProcess(AbstractMixin):
    def __init__(self, *args, random_mode: str,
                 random_args: Union[Sequence, Sequence[Sequence]] = (),
                 random_module: str = "random", rand_seq: bool = True,
                 **kwargs):
        """
        Saves specified function to generate random values to current class.
        Function is saved inside :param:`random_fn`.

        Parameters
        ----------
        random_mode: str
            specifies distribution which should be used to sample additive value
        random_args: Union[Sequence, Sequence[Sequence]]
            positional arguments passed for random function. If Sequence[Sequence]
            is provided, a random value for each item in the outer
            Sequence is generated
        random_module: str
            module from where function random function should be imported
        rand_seq: bool
            if enabled, multiple random values are generated if :param:`random_args`
            is of type Sequence[Sequence]
        """
        super().__init__(*args, **kwargs)
        self.random_module = random_module
        self.random_mode = random_mode
        self.random_args = random_args
        self.rand_seq = rand_seq

    def rand(self, **kwargs):
        """
        Return random value

        Returns
        -------
        Any
            object generated from function
        """
        if (self.rand_seq and len(self.random_args) > 0 and
                isinstance(self.random_args[0], Sequence)):
            val = tuple(self.random_fn(*args, **kwargs) for args in self.random_args)
        else:
            val = self.random_fn(*self.random_args, **kwargs)
        return val

    @property
    def random_mode(self) -> str:
        """
        Get random mode

        Returns
        -------
        str
            random mode
        """
        return self._random_mode

    @random_mode.setter
    def random_mode(self, mode) -> None:
        """
        Set random mode

        Parameters
        ----------
        mode: str
            specifies distribution which should be used to sample additive value (supports all
            random generators from python random package)
        """
        module = importlib.import_module(self.random_module)
        self._random_mode = mode
        self.random_fn = getattr(module, mode)
