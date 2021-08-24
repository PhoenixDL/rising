import math
import unittest

import torch

from rising.utils.affine import (
    deg_to_rad,
    get_batched_eye,
    matrix_revert_coordinate_order,
    matrix_to_cartesian,
    matrix_to_homogeneous,
    points_to_cartesian,
    points_to_homogeneous,
    unit_box,
)


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
            torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]),  # multiple elements, multiple points, 2d
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
                self.assertTrue(torch.allclose(points_to_homogeneous(inp), exp))

    def test_points_to_cartesian(self):
        expectations = [
            torch.tensor([[[0, 0]]]),  # single element, one point, 2d
            torch.tensor([[[0, 0, 0]]]),  # single element, one point, 3d
            torch.tensor([[[3, 3], [4, 4]]]),  # single element, multiple points, 2d
            torch.tensor([[[3, 3, 3], [4, 4, 4]]]),  # single element, multiple points 3d
            torch.tensor([[[0, 0]], [[2, 2]]]),  # multiple elements, one point, 2d
            torch.tensor([[[0, 0, 0]], [[2, 2, 2]]]),  # multiple elements, one point, 3d
            torch.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]]]),  # multiple elements, multiple points, 2d
            torch.tensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]),  # multiple elements, multiple points, 2d
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
                self.assertTrue(torch.allclose(points_to_cartesian(inp), exp))

    def test_matrix_to_homogeneous(self):
        inputs = [
            torch.tensor([[[1, 2], [3, 4]]]),  # single sample, no trans, 2d
            torch.tensor([[[1, 2, 5], [3, 4, 6]]]),  # single sample, trans, 2d
            torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),  # single sample, no trans, 3d
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]]),  # single sample, trans, 3d
            torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # multiple samples, no trans, 2d
            torch.tensor([[[1, 2, 10], [3, 4, 11]], [[5, 6, 12], [7, 8, 13]]]),  # multiple samples, trans, 2d
            # multiple samples, trans, 3d
            torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]),
            # multiple samples, trans, 3d
            torch.tensor(
                [[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23]], [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26]]]
            ),
        ]

        expectations = [
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 5], [3, 4, 6], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]], [[5, 6, 0], [7, 8, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 10], [3, 4, 11], [0, 0, 1]], [[5, 6, 12], [7, 8, 13], [0, 0, 1]]]),
            torch.tensor(
                [
                    [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]],
                    [[10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0], [0, 0, 0, 1]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23], [0, 0, 0, 1]],
                    [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26], [0, 0, 0, 1]],
                ]
            ),
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(matrix_to_homogeneous(inp), exp))

    def test_matrix_to_cartesian(self):
        expectations = [
            torch.tensor([[[1, 2], [3, 4]]]),  # single sample, no trans, 2d
            torch.tensor([[[1, 2, 5], [3, 4, 6]]]),  # single sample, trans, 2d
            torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),  # single sample, no trans, 3d
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]]),  # single sample, trans, 3d
            torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # multiple samples, no trans, 2d
            torch.tensor([[[1, 2, 10], [3, 4, 11]], [[5, 6, 12], [7, 8, 13]]]),  # multiple samples, trans, 2d
            # multiple samples, trans, 3d
            torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]),
            # multiple samples, trans, 3d
            torch.tensor(
                [[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23]], [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26]]]
            ),
        ]

        inputs = [
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 5], [3, 4, 6], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]], [[5, 6, 0], [7, 8, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 10], [3, 4, 11], [0, 0, 1]], [[5, 6, 12], [7, 8, 13], [0, 0, 1]]]),
            torch.tensor(
                [
                    [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]],
                    [[10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0], [0, 0, 0, 1]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23], [0, 0, 0, 1]],
                    [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26], [0, 0, 0, 1]],
                ]
            ),
        ]

        keep_square = True
        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(matrix_to_cartesian(inp, keep_square=keep_square), exp))
                keep_square = not keep_square

    def test_batched_eye(self):
        for dtype in [torch.float, torch.long]:
            for ndim in range(10):
                for batchsize in range(10):
                    with self.subTest(batchsize=batchsize, ndim=ndim, dtype=dtype):
                        batched_eye = get_batched_eye(batchsize=batchsize, ndim=ndim, dtype=dtype)
                        self.assertTupleEqual(batched_eye.size(), (batchsize, ndim, ndim))
                        self.assertEqual(dtype, batched_eye.dtype)

                        non_batched_eye = torch.eye(ndim, dtype=dtype)
                        for _eye in batched_eye:
                            self.assertTrue(torch.allclose(_eye, non_batched_eye, atol=1e-6))

    def test_deg_to_rad(self):
        inputs = [
            torch.tensor([tmp * 45.0 for tmp in range(9)]),
        ]

        expectations = [torch.tensor([tmp * math.pi / 4 for tmp in range(9)])]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(deg_to_rad(inp), exp, atol=1e-6))

    def test_unit_box_2d(self):
        curr_img_size = torch.tensor([2, 3])
        box = torch.tensor([[0.0, 0.0], [0.0, curr_img_size[1]], [curr_img_size[0], 0], curr_img_size])
        created_box = unit_box(2, curr_img_size).to(box)
        self.compare_points_unordered(box, created_box)

    def compare_points_unordered(self, points0: torch.Tensor, points1: torch.Tensor):
        self.assertEqual(tuple(points0.shape), tuple(points1.shape))
        for point in points0:
            comp = point[None] == points1
            comp = comp.sum(dim=1) == comp.shape[1]
            self.assertTrue(comp.any())

    def test_unit_box_3d(self):
        curr_img_size = torch.tensor([2, 3, 4])
        box = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, curr_img_size[2]],
                [0.0, curr_img_size[1], 0],
                [0.0, curr_img_size[1], curr_img_size[2]],
                [curr_img_size[0], 0.0, 0.0],
                [curr_img_size[0], 0.0, curr_img_size[2]],
                [curr_img_size[0], curr_img_size[1], 0.0],
                curr_img_size,
            ]
        )
        created_box = unit_box(3, curr_img_size).to(box)
        self.compare_points_unordered(box, created_box)

    def test_matrix_coordinate_order(self):
        inputs = [torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])]

        expectations = [torch.tensor([[[5, 4, 6], [2, 1, 3], [7, 8, 9]]])]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(matrix_revert_coordinate_order(inp), exp))
                # self.assertTrue(torch.allclose(inp, matrix_revert_coordinate_order(exp)))


if __name__ == "__main__":
    unittest.main()
