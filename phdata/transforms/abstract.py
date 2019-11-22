import torch
import typing


class AbstractTransform(torch.nn.Module):
    def __init__(self, grad: bool = False, **kwargs):
        """
        Base class for all transforms

        Parameters
        ----------
        grad: bool
            enables differentiation through the transform
        """
        super().__init__()
        self.grad = grad
        for key, item in kwargs.items():
            setattr(self, key, item)

    def __call__(self, *args, **kwargs) -> typing.Any:
        """
        Call super class with correct torch context

        Parameters
        ----------
        args:
            forwarded positional arguments
        kwargs:
            forwarded keyword arguments

        Returns
        -------

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            return super().__call__(*args, **kwargs)
