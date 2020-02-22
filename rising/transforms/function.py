from rising.transforms.abstract import AbstractTransform
from typing import Callable, Sequence

__all__ = [
    'LambdaKeyTransform',
    'LambdaTransform'
]


class LambdaTransform(AbstractTransform):
    def __init__(self, trafo_fn: Callable, grad: bool = False):
        """
        Transform which accepts a function and applies this function to all
        batches it gets.

        Parameters
        ----------
        trafo_fn : Callable
            the function which will be applied to all batches.
        grad : bool
            whether to trace the gradient for this transformation (default: False)

        Notes
        -----
        Since there are no sanity checks computed here, the user is
        completely responsible for correct data shapes and types.

        """
        super().__init__(grad=grad)
        if not callable(trafo_fn):
            raise ValueError('The given transfomation function must be callable')
        self.trafo_fn = trafo_fn

    def forward(self, **data) -> dict:
        return self.trafo_fn(data)


class LambdaKeyTransform(LambdaTransform):
    def __init__(self, trafo_fn: Callable, keys: Sequence = ('data', ), grad: bool = False):
        """
        Transform which accepts a function and applies this function to all given keys of all
        batches it gets.

        Parameters
        ----------
        trafo_fn : Callable
            the function which will be applied to all batches.
        keys : Sequence
            sequence of string specifying to which parts of the batch the function shall be applied.
        grad : bool
            whether to trace the gradient for this transformation (default: False)

        Notes
        -----
        Since there are no sanity checks computed here, the user is
        completely responsible for correct data shapes and types.

        """
        super().__init__(trafo_fn=trafo_fn, grad=grad)
        self.keys = keys

    def forward(self, **data) -> dict:
        for key in self.keys:
            data[key] = self.trafo_fn(key, data.get(key, None))
        return data
