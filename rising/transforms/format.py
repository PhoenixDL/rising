from typing import Callable, Dict, Hashable, Mapping, Sequence, Tuple, Union

from rising.transforms.functional.utility import filter_keys, pop_keys

from .abstract import AbstractTransform

__all__ = ["MapToSeq", "SeqToMap", "PopKeys", "FilterKeys", "RenameKeys"]


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
            tuple: mapped data
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

        Returns:
            dict: mapped data
        """
        return {_key: data[_idx] for _idx, _key in enumerate(self.keys)}


class PopKeys(AbstractTransform):
    """
    Pops keys from a given data dict
    """

    def __init__(self, keys: Union[Callable, Sequence], return_popped: bool = False):
        """
        Args:
            keys : if callable it must return a boolean for each key
                indicating whether it should be popped from the dict.
                if sequence of strings, the strings shall be the keys to be
                poppedAbstractTransform,
            return_popped: whether to also return the popped values
                (default: False)
        """
        super().__init__(grad=False)
        self.keys = keys
        self.return_popped = return_popped

    def forward(self, **data) -> Union[dict, Tuple[dict, dict]]:
        return pop_keys(data=data, keys=self.keys, return_popped=self.return_popped)


class FilterKeys(AbstractTransform):
    """
    Filters keys from a given data dict
    """

    def __init__(self, keys: Union[Callable, Sequence], return_popped: bool = False):
        """
        Args:
            keys: if callable it must return a boolean for each key
                indicating whether it should be retained in the dict.
                if sequence of strings, the strings shall be the keys to be
                retained
            return_popped: whether to also return the popped values
                (default: False)
        """
        super().__init__(grad=False)
        self.keys = keys
        self.return_popped = return_popped

    def forward(self, **data) -> Union[dict, Tuple[dict, dict]]:
        return filter_keys(data=data, keys=self.keys, return_popped=self.return_popped)


class RenameKeys(AbstractTransform):
    """Rename keys inside batch"""

    def __init__(self, keys: Mapping[Hashable, Hashable]):
        """
        Args:
            keys: keys of mapping define current name and items define the
                new names
        """
        super().__init__(grad=False)
        self.keys = keys

    def forward(self, **data) -> Dict:
        for old_key, new_key in self.keys.items():
            data[new_key] = data.pop(old_key)
        return data
