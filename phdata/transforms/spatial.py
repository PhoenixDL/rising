import torch
from .abstract import AbstractTransform
from typing import Callable, Union, Sequence, Any

from .functional.spatial import mirror, rot90
from phdata.utils import check_scalar


augment_callable = Callable[[torch.Tensor, Union[float, Sequence]], Any]


class RandomAxisTransform(AbstractTransform):
    def __init__(self, augment_fn: augment_callable, dims: Sequence, keys: Sequence = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Random mirror transform

        Parameters
        ----------
        augment_fn: callable
            function for augmentation
        dims: tuple
            axes which should be mirrored
        keys: tuple
            keys which should be mirrored
        prob: typing.Union[float, tuple]
            probability for mirror. If float value is provided, it is used
            for all dims
        grad: bool
            enables differentiation through the transform
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
        rand_val = torch.rand(data[self.keys[0]].ndim - 2, requires_grad=False)
        dims = [_dim for _dim, _prob in zip(self.dims, self.prob) if rand_val[_dim] < _prob]

        for key in self.keys:
            data[key] = self.augment_fn(data[key], dims=dims, **self.kwargs)
        return data


class MirrorTransform(RandomAxisTransform):
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
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to superclass
        """
        super().__init__(augment_fn=mirror, dims=dims, keys=keys, prob=prob, grad=grad, **kwargs)


class Rot90Transform(AbstractTransform):
    def __init__(self, dims: tuple, keys: tuple = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Randomly rotate 90 degree around dims

        Parameters
        ----------
        dims: tuple
            dims which should be mirrored
        keys: tuple
            keys which should be mirrored
        prob: typing.Union[float, tuple]
            probability for mirror. If float value is provided, it is used
            for all dims
        grad: bool
            enables differentiation through the transform
        kwargs:
            keyword arguments passed to superclass
        """
        super().__init__(augment_fn=rot90, dims=dims, keys=keys, prob=prob, grad=grad, **kwargs)

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
        self.kwargs["k"] = torch.randint(0, 3, (1,), requires_grad=False).item()
        return super().forward(**data)
