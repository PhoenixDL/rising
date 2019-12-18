import unittest
from rising.utils.affine import points_to_homogeneous, matrix_to_homogeneous, \
    matrix_to_cartesian, points_to_cartesian, matrix_revert_coordinate_order, \
    get_batched_eye, _format_scale, _format_translation, deg_to_rad, \
    _format_rotation, parametrize_matrix, assemble_matrix_if_necessary
import torch


class AffineHelperTests(unittest.TestCase):
    def test_points_to_homogeneous(self):
        inputs = [
            torch.tensor([[[0, 0]]]),  # single element, one point, 2d
            torch.tensor([[[0, 0, 0]]]),  # single element, one point, 3d
            torch.tensor([[[3, 3], [4, 4]]]),  # single element, multiple points, 2d
            torch.tensor([[[3, 3, 3], [4, 4, 4]]]),  # single element, multiple points 3d
            torch.tensor([[[0, 0]], [[2, 2]]]),  # multiple elements, one point, 2d
            torch.tensor([[[0, 0, 0]], [[2, 2, 2]]]),  # multiple elements, one point, 3d
            torch.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]]]),  # multiple elements, multiple points, 2d
            torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])  # multiple elements, multiple points, 2d
        ]
        expectations = [
            torch.tensor([[[0, 0, 1]]]),  # single element, one point, 2d
            torch.tensor([[[0, 0, 0, 1]]]),  # single element, one point, 3d
            torch.tensor([[[3, 3, 1], [4, 4, 1]]]),  # single element, multiple points, 2d
            torch.tensor([[[3, 3, 3, 1], [4, 4, 4, 1]]]),  # single element, multiple points 3d
            torch.tensor([[[0, 0, 1]], [[2, 2, 1]]]),  # multiple elements, one point, 2d
            torch.tensor([[[0, 0, 0, 1]], [[2, 2, 2, 1]]]),  # multiple elements, one point, 3d
            torch.tensor([[[0, 0, 1], [1, 1, 1]], [[2, 2, 1], [3, 3, 1]]]),  # multiple elements, multiple points, 2d
            torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]], [[2, 2, 2, 1], [3, 3, 3, 1]]])  # multiple elements,
            # multiple points, 2d
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue((points_to_homogeneous(inp) == exp).all())

    def test_points_to_cartesian(self):
        expectations = [
            torch.tensor([[[0, 0]]]),  # single element, one point, 2d
            torch.tensor([[[0, 0, 0]]]),  # single element, one point, 3d
            torch.tensor([[[3, 3], [4, 4]]]),  # single element, multiple points, 2d
            torch.tensor([[[3, 3, 3], [4, 4, 4]]]),  # single element, multiple points 3d
            torch.tensor([[[0, 0]], [[2, 2]]]),  # multiple elements, one point, 2d
            torch.tensor([[[0, 0, 0]], [[2, 2, 2]]]),  # multiple elements, one point, 3d
            torch.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]]]),  # multiple elements, multiple points, 2d
            torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])  # multiple elements, multiple points, 2d
        ]
        inputs = [
            torch.tensor([[[0, 0, 1]]]),  # single element, one point, 2d
            torch.tensor([[[0, 0, 0, 1]]]),  # single element, one point, 3d
            torch.tensor([[[3, 3, 1], [4, 4, 1]]]),  # single element, multiple points, 2d
            torch.tensor([[[3, 3, 3, 1], [4, 4, 4, 1]]]),  # single element, multiple points 3d
            torch.tensor([[[0, 0, 1]], [[2, 2, 1]]]),  # multiple elements, one point, 2d
            torch.tensor([[[0, 0, 0, 1]], [[2, 2, 2, 1]]]),  # multiple elements, one point, 3d
            torch.tensor([[[0, 0, 1], [1, 1, 1]], [[2, 2, 1], [3, 3, 1]]]),  # multiple elements, multiple points, 2d
            torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]], [[2, 2, 2, 1], [3, 3, 3, 1]]])  # multiple elements,
            # multiple points, 2d
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue((points_to_cartesian(inp) == exp).all())


if __name__ == '__main__':
    unittest.main()