from typing import Sequence, Union, Callable, Any, Mapping
from rising.utils import check_scalar
from rising.transforms import AbstractTransform, RandomProcess


__all__ = ["Compose", "DropoutCompose"]


def dict_call(batch: dict, transform: Callable) -> Any:
    """
    Unpacks the dict for every transformation

    Parameters
    ----------
    batch: dict
        current batch which is passed to transform
    transform: Callable
        transform to perform

    Returns
    -------
    Any
        transformed batch
    """
    return transform(**batch)


class Compose(AbstractTransform):
    def __init__(self, *transforms,
                 transform_call: Callable[[Any, Callable], Any] = dict_call):
        """
        Compose multiple transforms

        Parameters
        ----------
        transforms: Union[AbstractTransform, Sequence[AbstractTransform]]
            one or multiple transformations which are applied in consecutive order
        transform_call: Callable[[Any, Callable], Any], optional
            function which determines how transforms are called. By default
            Mappings and Sequences are unpacked during the transform.
        """
        super().__init__(grad=True)
        if isinstance(transforms[0], Sequence):
            transforms = transforms[0]
        self.transforms = transforms
        self.transform_call = transform_call

    def forward(self, *seq_like, **map_like) -> Union[Sequence, Mapping]:
        """
        Apply transforms in a consecutive order. Can either handle
        Sequence like or Mapping like data.

        Parameters
        ----------
        seq_like: tuple
            data which is unpacked like a Sequence
        map_like: dict
            data which is unpacked like a dict

        Returns
        -------
        dict
            dict with transformed data
        """
        assert not (seq_like and map_like)
        data = seq_like if seq_like else map_like

        for trafo in self.transforms:
            data = self.transform_call(data, trafo)
        return data


class DropoutCompose(RandomProcess, Compose):
    def __init__(self, *transforms, dropout: Union[float, Sequence[float]] = 0.5,
                 random_mode: str = "random", random_args: Sequence = (), random_module: str = "random",
                 transform_call: Callable[[Any, Callable], Any] = dict_call,
                 **kwargs):
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
        random_module: str
            module from where function random function should be imported
        transform_call: Callable[[Any, Callable], Any], optional
            function which determines how transforms are called. By default
            Mappings and Sequences are unpacked during the transform.

        Raises
        ------
        TypeError
            if dropout is a sequence it must have the same length as transforms
        """
        super().__init__(*transforms, random_mode=random_mode,
                         random_args=random_args, random_module=random_module,
                         rand_seq=False, transform_call=transform_call, **kwargs)
        if check_scalar(dropout):
            dropout = [dropout] * len(self.transforms)
        if len(dropout) != len(self.transforms):
            raise TypeError(f"If dropout is a sequence it must specify the dropout probability "
                            f"for each transform, found {len(dropout)} probabilities "
                            f"and {len(self.transforms)} transforms.")
        self.dropout = dropout

    def forward(self, *seq_like, **map_like) -> Union[Sequence, Mapping]:
        """
        Apply transforms in a consecutive order. Can either handle
        Sequence like or Mapping like data.

        Parameters
        ----------
        seq_like: tuple
            data which is unpacked like a Sequence
        map_like: dict
            data which is unpacked like a dict

        Returns
        -------
        dict
            dict with transformed data
        """
        assert not (seq_like and map_like)
        data = seq_like if seq_like else map_like

        for trafo, drop in zip(self.transforms, self.dropout):
            if self.rand() > drop:
                data = self.transform_call(data, trafo)
        return data
