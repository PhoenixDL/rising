from typing import Sequence, Union
import random
from rising.utils import check_scalar
from .abstract import AbstractTransform


class Compose(AbstractTransform):
    def __init__(self, *transforms, grad: bool = False):
        """
        Compose multiple transforms to one

        Parameters
        ----------
        transforms: Union[AbstractTransform, Sequence[AbstractTransform]]
            one or multiple transformations which are applied in consecutive order
        grad: bool
            enable gradient computation inside transformation
        """
        super().__init__(grad=grad)
        if isinstance(transforms[0], Sequence):
            transforms = transforms[0]
        self.transforms = transforms

    def forward(self, **data) -> dict:
        """
        Apply transforms in a consecutive order

        Parameters
        ----------
        data: dict
            dict with data

        Returns
        -------
        dict
            dict with transformed data
        """
        for trafo in self.transforms:
            data = trafo(**data)
        return data


class DropoutCompose(Compose):
    def __init__(self, *transforms, dropout: Union[float, Sequence[float]] = 0.5,
                 grad: bool = False):
        """
        Compose multiple transforms to one

        Parameters
        ----------
        transforms: Union[AbstractTransform, Sequence[AbstractTransform]]
            one or multiple transformations which are applied in consecutive order
        dropout: Union[float, Sequence[float]]
            if provided as float, each transform is skipped with the given probability
            if :param:`dropout` is a sequence, it needs to specify the dropout
            probability for each given transform
        grad: bool
            enable gradient computation inside transformation

        Raises
        ------
        TypeError
            if dropout is a sequence it must have the same length as transforms
        """
        super().__init__(*transforms, grad=grad)
        if check_scalar(dropout):
            dropout = [dropout] * len(self.transforms)
        if len(dropout) != len(self.transforms):
            raise TypeError(f"If dropout is a sequence it must specify the dropout probability "
                            f"for each transform, found {len(dropout)} probabilities "
                            f"and {len(self.transforms)} transforms.")
        self.dropout = dropout

    def forward(self, **data) -> dict:
        """
        Apply transforms in a consecutive order

        Parameters
        ----------
        data: dict
            dict with data

        Returns
        -------
        dict
            dict with transformed data
        """
        for trafo, drop in zip(self.transforms, self.dropout):
            if random.random() > drop:
                data = trafo(**data)
        return data
