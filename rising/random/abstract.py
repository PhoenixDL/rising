from abc import abstractmethod
from typing import Optional, Sequence, Union

import torch

from rising.utils.shape import reshape

__all__ = ["AbstractParameter"]


class AbstractParameter(torch.nn.Module):
    """
    Abstract Parameter class to inject randomness to transforms
    """

    @staticmethod
    def _get_n_samples(size: Union[Sequence, torch.Size] = (1,)):
        """
        Calculates the number of elements in the given size

        Args:
            size: Sequence or torch.Size

        Returns:
            int: the number of elements
        """
        if not isinstance(size, torch.Size):
            size = torch.Size(size)
        return size.numel()

    @abstractmethod
    def sample(self, n_samples: int) -> Union[torch.Tensor, list]:
        """
        Abstract sampling function

        Args:
            n_samples : the number of samples to return

        Returns:
            torch.Tensor or list: the sampled values
        """
        raise NotImplementedError

    def forward(
        self,
        size: Optional[Union[Sequence, torch.Size]] = None,
        device: Union[torch.device, str] = None,
        dtype: Union[torch.dtype, str] = None,
        tensor_like: torch.Tensor = None,
    ) -> Union[None, list, torch.Tensor]:
        """
        Forward function (will also be called if the module is called).
        Calculates the number of samples from the given shape, performs the
        sampling and converts it back to the correct shape.

        Args:
            size: the size of the sampled values. If None, it samples one value
                without reshaping
            device : the device the result value should be set to, if it is a tensor
            dtype : the dtype, the result value should be casted to, if it is a tensor
            tensor_like: the tensor, having the correct dtype and device.
                The result will be pushed onto this device and casted to this
                dtype if this is specified.

        Returns:
            list or torch.Tensor: the sampled values

        Notes:
            if the parameter ``tensor_like`` is given,
            it overwrites the parameters ``dtype`` and ``device``
        """
        n_samples = self._get_n_samples(size if size is not None else (1,))
        samples = self.sample(n_samples)

        if any([s is None for s in samples]):
            return None

        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples).flatten()

        if size is not None:
            samples = reshape(samples, size)

        if isinstance(samples, torch.Tensor):
            if tensor_like is not None:
                samples = samples.to(tensor_like)
            else:
                samples = samples.to(device=device, dtype=dtype)
        return samples
