import typing
import torch
from .abstract import AbstractTransform

from .functional.spatial import mirror


class MirrorTransform(AbstractTransform):
    def __int__(self, axis: typing.Iterable, keys: tuple = ('data',),
                prob: float = 0.5, grad: bool = False, **kwargs):
        super().__init__(grad=grad, **kwargs)
        self.axis = axis
        self.keys = keys
        self.prob = prob

    def forward(self, **data):
        rand_val = torch.rand(data[self.keys[0]].ndim - 2)
        for ax in self.axis:
            if rand_val[ax] < self.prob:
                for key in self.keys:
                    data[key] = mirror(data[key], ax)


class Rot90Transform(AbstractTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        pass
