from .abstract import AbstractTransform

__all__ = ["MapToSeq", "SeqToMap"]


class MapToSeq(AbstractTransform):
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Convert dict to sequence

        Parameters
        ----------
        keys: tuple
            keys which are mapped into sequence.
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, **data) -> tuple:
        """
        Convert input

        Parameters
        ----------
        data: dict
            input dict

        Returns
        -------
        tuple
            mapped data
        """
        return tuple(data[_k] for _k in self.keys)


class SeqToMap(AbstractTransform):
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Convert sequence to dict

        Parameters
        ----------
        keys: tuple
            keys which are mapped into dict.
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, *data, **kwargs) -> dict:
        """
        Convert input

        Parameters
        ----------
        data: tuple
            input tuple

        Returns
        -------
        dict
            mapped data
        """
        return {_key: data[_idx] for _idx, _key in enumerate(self.keys)}
