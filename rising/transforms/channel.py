from typing import Sequence, Dict

from rising.transforms import AbstractTransform

from rising.transforms.functional import tensor_op

__all__ = ["Permute"]


class Permute(AbstractTransform):
    def __init__(self, dims: Dict[str, Sequence[int]], grad: bool = False, **kwargs):
        """
        Apply augment_fn to keys

        Parameters
        ----------
        dims: Dict[str, Sequence[int]]
            defines permutation sequence for respective key
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to permute function
        """
        super().__init__(grad=grad)
        self.dims = dims
        self.kwargs = kwargs

    def forward(self, **data) -> dict:
        """
        Forward input

        Parameters
        ----------
        data: dict
            batch dict

        Returns
        -------
        dict
            augmented data
        """
        for key, item in self.dims.items():
            data[key] = tensor_op(data[key], "permute", *item, **self.kwargs)
        return data