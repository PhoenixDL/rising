import torch
from rising.utils.affine import to_cartesian, matrix_to_homogeneous, \
    points_to_homogeneous, matrix_permute_coordinate_order


def affine_point_transform(point_batch: torch.Tensor,
                           matrix_batch: torch.Tensor) -> torch.Tensor:
    point_batch = points_to_homogeneous(point_batch)
    matrix_batch = matrix_to_homogeneous(matrix_batch)

    matrix_batch = matrix_permute_coordinate_order(matrix_batch)

    transformed_points = torch.bmm(point_batch,
                                   matrix_batch.permute(0, 2, 1))

    return to_cartesian(transformed_points)

