import torch
from .abstract import RandomDimsTransform
from typing import Union, Sequence

from .functional.spatial import mirror, rot90


# TODO: Crops
# TODO: Rotation
# TODO: Deformable
# TODO: Zoom/Scale


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


class Rot90Transform(RandomDimsTransform):
    def __init__(self, dims: tuple, keys: tuple = ('data',),
                 prob: Union[float, Sequence] = 0.5, grad: bool = False, **kwargs):
        """
        Randomly rotate 90 degree around dims

        Parameters
        ----------
        dims: tuple
            dims which should be rotated
        keys: tuple
            keys which should be rotated
        prob: typing.Union[float, tuple]
            probability for rotation. If float value is provided, it is used
            for all dims
        grad: bool
            enable gradient computation inside transformation
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
