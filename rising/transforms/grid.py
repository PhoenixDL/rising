from typing import Sequence, Union, Dict, Tuple

import torch

from abc import abstractmethod
from torch import Tensor

from rising.transforms import AbstractTransform
from rising.utils.affine import get_batched_eye, matrix_to_homogeneous


__all__ = ["GridTransform", "StackedGridTransform"]


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
                padding_mode=self.padding_mode, align_corners=self.align_corners,
                **self.kwargs)
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
                    matrix, size=input_size, align_corners=self.align_corners)
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
