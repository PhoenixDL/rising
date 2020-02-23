from typing import Sequence, Union, Dict, Tuple

import torch

from abc import abstractmethod
from torch import Tensor

from rising.transforms import AbstractTransform
from rising.utils.affine import get_batched_eye, matrix_to_homogeneous
from rising.transforms.functional import center_crop, random_crop, add_noise


__all__ = ["GridTransform", "StackedGridTransform",
           "CenterCropGrid", "RandomCropGrid", "RandomDistortion", "RadialDistortion"]


class GridTransform(AbstractTransform):
    def __init__(self,
                 keys: Sequence[str] = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False,
                 **kwargs,
                 ):
        super().__init__(grad=grad)
        self.keys = keys
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.kwargs = kwargs

        self.grid: Dict[Tuple, Tensor] = None

    def forward(self, **data) -> dict:
        if self.grid is None:
            self.grid = self.create_grid([data[key].shape for key in self.keys])

        self.grid = self.augment_grid(self.grid)

        for key in self.keys:
            _grid = self.grid[tuple(data[key].shape)]
            _grid = _grid.to(data[key])

            data[key] = torch.nn.functional.grid_sample(
                data[key], _grid, mode=self.interpolation_mode,
                padding_mode=self.padding_mode, align_corners=self.align_corners)
        self.grid = None
        return data

    @abstractmethod
    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:
        raise NotImplementedError

    def create_grid(self, input_size: Sequence[Sequence[int]],
                    matrix: Tensor = None) -> Dict[Tuple, Tensor]:
        if matrix is None:
            matrix = get_batched_eye(batchsize=input_size[0][0], ndim=len(input_size[0]) - 2)
            matrix = matrix_to_homogeneous(matrix)[:, :-1]

        grid = {}
        for size in input_size:
            if tuple(size) not in grid:
                grid[tuple(size)] = torch.nn.functional.affine_grid(
                    matrix, size=size, align_corners=self.align_corners)
        return grid

    def __add__(self, other):
        if not isinstance(other, GridTransform):
            raise ValueError("Concatenation is only supported for grid transforms.")
        return StackedGridTransform(self, other)

    def __radd__(self, other):
        if not isinstance(other, GridTransform):
            raise ValueError("Concatenation is only supported for grid transforms.")
        return StackedGridTransform(other, self)


class StackedGridTransform(GridTransform):
    def __init__(self, *transforms: Union[GridTransform, Sequence[GridTransform]]):
        super().__init__(keys=None, interpolation_mode=None, padding_mode=None,
                         align_corners=None)
        if isinstance(transforms, (tuple, list)):
            if isinstance(transforms[0], (tuple, list)):
                transforms = transforms[0]
        self.transforms = transforms

    def create_grid(self, input_size: Sequence[Sequence[int]], matrix: Tensor = None) -> \
            Dict[Tuple, Tensor]:
        return self.transforms[0].create_grid(input_size=input_size, matrix=matrix)

    def augment_grid(self, grid: Tensor) -> Tensor:
        for transform in self.transforms:
            grid = transform.augment_grid(grid)
        return grid


class CenterCropGrid(GridTransform):
    def __init__(self,
                 size: Union[int, Sequence[int]],
                 keys: Sequence[str] = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False,
                 **kwargs,):
        super().__init__(keys=keys, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         grad=grad, **kwargs)
        self.size = size

    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:
        return {key: center_crop(item, size=self.size, grid_crop=True)
                for key, item in grid.items()}


class RandomCropGrid(GridTransform):
    def __init__(self,
                 size: Union[int, Sequence[int]],
                 dist: Union[int, Sequence[int]] = 0,
                 keys: Sequence[str] = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False,
                 **kwargs,):
        super().__init__(keys=keys, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         grad=grad, **kwargs)
        self.size = size
        self.dist = dist

    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:
        return {key: random_crop(item, size=self.size, dist=self.dist, grid_crop=True)
                for key, item in grid.items()}


class RandomDistortion(GridTransform):
    def __init__(self,
                 noise_type: str,
                 noise_kwargs: dict = None,
                 keys: Sequence[str] = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False,
                 **kwargs,):
        super().__init__(keys=keys, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         grad=grad, **kwargs)
        self.noise_type = noise_type
        self.noise_kwargs = noise_kwargs if noise_kwargs is not None else {}

    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:
        return {key: add_noise(item, noise_type=self.noise_type, **self.noise_kwargs)
                for key, item in grid.items()}


class RadialDistortion(GridTransform):
    def __init__(self,
                 scale: float,
                 keys: Sequence[str] = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False,
                 **kwargs,):
        super().__init__(keys=keys, interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode, align_corners=align_corners,
                         grad=grad, **kwargs)
        self.scale = scale

    def augment_grid(self, grid: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:

        new_grid = {key: radial_distortion_grid(item, scale=self.scale)
                    for key, item in grid.items()}
        print(new_grid)
        return new_grid


def radial_distortion_grid(grid: Tensor, scale: float) -> Tensor:
    # spatial_shape = grid.shape[1:-1]
    # new_grid = torch.stack([torch.meshgrid(
    #     *[torch.linspace(-1, 1, i) for i in spatial_shape])], dim=-1).to(grid)
    # print(new_grid.shape)
    #
    # distortion =

    dist = torch.norm(grid, 2, dim=-1, keepdim=True)
    dist = dist / dist.max()
    distortion = (scale[0] * dist.pow(3) + scale[1] * dist.pow(2) + scale[2] * dist) / 3
    print(distortion.max())
    print(distortion.min())
    return grid * (1 - distortion)
