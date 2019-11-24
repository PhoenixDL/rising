import torch
import random
from .abstract import RandomDimsTransform, AbstractTransform
from typing import Union, Sequence
from itertools import permutations

from .functional.spatial import mirror, rot90


class MirrorTransform(RandomDimsTransform):
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


class Rot90Transform(AbstractTransform):
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
            k = random.randint(0, 4)
            rand_dims = self._permutations[random.randint(0, len(self._permutations))]

            for key in self.keys:
                data[key] = rot90(data[key], k=k, dims=rand_dims)
        return data

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        self._dims = dims
        self._permutations = tuple(permutations(dims, 2))
