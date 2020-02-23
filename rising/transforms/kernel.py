import math
import torch
from typing import Sequence, Union, Callable

from .abstract import AbstractTransform
from rising.utils import check_scalar
from rising.transforms.functional import gaussian_kernel

__all__ = ["KernelTransform", "GaussianSmoothing"]


class KernelTransform(AbstractTransform):
    def __init__(self, in_channels: int, kernel_size: Union[int, Sequence], dim: int = 2,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 0,
                 padding_mode: str = 'zero', keys: Sequence = ('data',), grad: bool = False,
                 **kwargs):
        """
        Baseclass for kernel based transformations (kernel is applied to each channel individually)

        Parameters
        ----------
        in_channels: int
            number of input channels
        kernel_size: int or sequence
            size of kernel
        dim: int
            number of spatial dimensions
        stride: int or sequence
            stride of convolution
        padding: int or sequence
            padding size for input
        padding_mode: str
            padding mode for input. Supports all modes from :func:`torch.functional.pad` except
            `circular`
        keys: sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass

        See Also
        --------
        :func:`torch.functional.pad`
        """
        super().__init__(grad=grad, **kwargs)
        self.in_channels = in_channels

        if check_scalar(kernel_size):
            kernel_size = [kernel_size] * dim
        self.kernel_size = kernel_size

        if check_scalar(stride):
            stride = [stride] * dim
        self.stride = stride

        if check_scalar(padding):
            padding = [padding] * dim * 2
        self.padding = padding

        self.padding_mode = padding_mode
        self.keys = keys

        kernel = self.create_kernel()
        self.register_buffer('weight', kernel)
        self.groups = in_channels
        self.conv = self.get_conv(dim)

    @staticmethod
    def get_conv(dim) -> Callable:
        """
        Select convolution with regard to dimension

        Parameters
        ----------
        dim: int
            spatial dimension of data
        """
        if dim == 1:
            return torch.nn.functional.conv1d
        elif dim == 2:
            return torch.nn.functional.conv2d
        elif dim == 3:
            return torch.nn.functional.conv3d
        else:
            raise TypeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def create_kernel(self) -> torch.Tensor:
        """
        Create kernel for convolution
        """
        raise NotImplementedError

    def forward(self, **data) -> dict:
        """
        Apply kernel to selected keys

        Parameters
        ----------
        data: dict
            dict with input data

        Returns
        -------
        dict
            dict with transformed data
        """
        for key in self.keys:
            inp_pad = torch.nn.functional.pad(data[key], self.padding, mode=self.padding_mode)
            data[key] = self.conv(inp_pad, weight=self.weight, groups=self.groups, stride=self.stride)
        return data


class GaussianSmoothing(KernelTransform):
    def __init__(self,
                 in_channels: int,
                 kernel_size: Union[int, Sequence],
                 std: Union[int, Sequence], dim: int = 2,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 0,
                 padding_mode: str = 'constant', keys: Sequence = ('data',), grad: bool = False,
                 **kwargs):
        """
        Perform Gaussian Smoothing.
        Filtering is performed seperately for each channel in the input using a depthwise convolution.
        This code is adapted from: ('https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian'
        '-filtering-for-an-image-2d-3d-in-pytorch/12351/10')

        Parameters
        ----------
        in_channels: int
            number of input channels
        kernel_size: int or sequence
            size of kernel
        std: int or sequence
            standard deviation of gaussian
        dim: int
            number of spatial dimensions
        stride: int or sequence
            stride of convolution
        padding: int or sequence
            padding size for input
        padding_mode: str
            padding mode for input. Supports all modes from :func:`torch.functional.pad` except
            `circular`
        keys: sequence
            keys which should be augmented
        grad: bool
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass

        See Also
        --------
        :func:`torch.functional.pad`
        """
        self.std = std
        self.spatial_dim = dim
        super().__init__(in_channels=in_channels, kernel_size=kernel_size, dim=dim, stride=stride,
                         padding=padding, padding_mode=padding_mode, keys=keys, grad=grad, **kwargs)

    def create_kernel(self) -> torch.Tensor:
        """
        Create gaussian blur kernel
        """
        return gaussian_kernel(kernel_size=self.kernel_size,
                               std=self.std, in_channels=self.in_channels,
                               dim=self.spatial_dim)
