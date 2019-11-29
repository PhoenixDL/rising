from typing import Sequence, Union
from rising.utils import check_scalar
from .abstract import AbstractTransform, RandomProcess


class Compose(AbstractTransform):
    def __init__(self, *transforms):
        """
        Compose multiple transforms

        Parameters
        ----------
        transforms: Union[AbstractTransform, Sequence[AbstractTransform]]
            one or multiple transformations which are applied in consecutive order
        """
        super().__init__(grad=True)
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


class DropoutCompose(RandomProcess, Compose):
    def __init__(self, *transforms, dropout: Union[float, Sequence[float]] = 0.5,
                 random_mode: str = "random", random_args: Sequence = (),
                 random_kwargs: dict = None, random_module: str = "random", **kwargs):
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
        random_mode: str
            specifies distribution which should be used to sample additive value
        random_args: Sequence
            positional arguments passed for random function
        random_kwargs: dict
            keyword arguments for random function
        random_module: str
            module from where function random function should be imported

        Raises
        ------
        TypeError
            if dropout is a sequence it must have the same length as transforms
        """
        super().__init__(*transforms, random_mode=random_mode,
                         random_kwargs=random_kwargs, random_args=random_args,
                         random_module=random_module, **kwargs)
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
            if self.rand() > drop:
                data = trafo(**data)
        return data
