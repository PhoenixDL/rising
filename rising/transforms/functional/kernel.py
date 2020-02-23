import math
import torch

from typing import Sequence, Union

from rising.utils import check_scalar

__all__ = ["gaussian_kernel"]


def gaussian_kernel(dim: int, kernel_size: Union[int, Sequence[int]],
                    std: Union[float, Sequence[float]], in_channels: int = 1) -> torch.Tensor:
    if check_scalar(kernel_size):
        kernel_size = [kernel_size] * dim
    if check_scalar(std):
        std = [std] * dim
    # The gaussian kernel is the product of the gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32)
                                for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, std, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / kernel.sum()

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(in_channels, *[1] * (kernel.dim() - 1))
    kernel.requires_grad = False
    return kernel.contiguous()
