from .abstract import AbstractTransform

__all__ = ["MapToSeq", "SeqToMap"]


class MapToSeq(AbstractTransform):
    """
    Convert dict to sequence
    """
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Args:
            keys: keys which are mapped into sequence.
            grad: enable gradient computation inside transformation
            ** kwargs: additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, **data) -> tuple:
        """
        Convert input

        Args:
            data: input dict

        Returns:
            mapped data
        """
        return tuple(data[_k] for _k in self.keys)


class SeqToMap(AbstractTransform):
    """Convert sequence to dict"""
    def __init__(self, *keys, grad: bool = False, **kwargs):
        """
        Args:
            keys: keys which are mapped into dict.
            grad: enable gradient computation inside transformation
            **kwargs: additional keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)
        if isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        self.keys = keys

    def forward(self, *data, **kwargs) -> dict:
        """
        Convert input

        Args:
            data: input tuple

        Args:
            mapped data
        """
        return {_key: data[_idx] for _idx, _key in enumerate(self.keys)}
