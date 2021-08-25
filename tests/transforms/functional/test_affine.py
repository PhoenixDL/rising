import unittest

import torch

from rising.transforms.functional.affine import (
    _check_new_img_size,
    affine_image_transform,
    affine_point_transform,
    create_rotation,
    create_scale,
    create_translation,
    parametrize_matrix,
)
from rising.utils.affine import matrix_to_cartesian, matrix_to_homogeneous


class AffineTestCase(unittest.TestCase):
    def test_check_image_size(self):
        images = [
            torch.rand(11, 2, 3, 4, 5),
            torch.rand(11, 2, 3, 4),
            torch.rand(11, 2, 3, 3),
        ]
        img_sizes = [[3, 4, 5], [3, 4], 3]

        scales = [torch.tensor([2.0, 3.0, 4.0]), torch.tensor([2.0, 3.0]), torch.tensor([2.0, 3.0])]
        rots = [[45.0, 90.0, 135.0], [45.0], [45.0]]
        trans = [[0.0, 10.0, 20.0], [10.0, 20.0], [10.0, 20.0]]

        edges = [
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 5.0, 1.0],
                [0.0, 4.0, 0.0, 1.0],
                [0.0, 4.0, 5.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 5.0, 1.0],
                [3.0, 4.0, 0.0, 1.0],
                [3.0, 4.0, 5.0, 1.0],
            ],
            [[0.0, 0.0, 1.0], [0.0, 4.0, 1.0], [3.0, 0.0, 1.0], [3.0, 4.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 3.0, 1.0], [3.0, 0.0, 1.0], [3.0, 3.0, 1.0]],
        ]

        for img, size, scale, rot, tran, edge_pts in zip(images, img_sizes, scales, rots, trans, edges):
            ndim = scale.size(-1)
            with self.subTest(ndim=ndim):
                affine = matrix_to_homogeneous(
                    parametrize_matrix(
                        scale=scale,
                        rotation=rot,
                        translation=tran,
                        degree=True,
                        batchsize=1,
                        ndim=ndim,
                        dtype=torch.float,
                        image_transform=False,
                    )
                )

                edge_pts = torch.tensor(edge_pts, dtype=torch.float)
                img = img.to(torch.float)
                new_edges = torch.bmm(edge_pts.unsqueeze(0), affine.clone().permute(0, 2, 1))

                img_size_zero_border = new_edges.max(dim=1)[0][0]
                img_size_non_zero_border = (new_edges.max(dim=1)[0] - new_edges.min(dim=1)[0])[0]

                fn_result_zero_border = _check_new_img_size(
                    size,
                    matrix_to_cartesian(affine.expand(img.size(0), -1, -1).clone()),
                    zero_border=True,
                )
                fn_result_non_zero_border = _check_new_img_size(
                    size,
                    matrix_to_cartesian(affine.expand(img.size(0), -1, -1).clone()),
                    zero_border=False,
                )

                self.assertTrue(torch.allclose(img_size_zero_border[:-1], fn_result_zero_border))
                self.assertTrue(torch.allclose(img_size_non_zero_border[:-1], fn_result_non_zero_border))

    def test_affine_point_transform(self):
        points = [
            [[[0, 1], [1, 0]]],
            [[[1, 1, 1]]],
        ]
        matrices = [
            torch.tensor([[[1.0, 0.0], [0.0, 5.0]]]),
            parametrize_matrix(
                scale=1,
                translation=0,
                rotation=[0, 0, 90],
                degree=True,
                batchsize=1,
                ndim=3,
                dtype=torch.float,
                device="cpu",
                image_transform=False,
            ),
        ]
        expected = [[[0, 5], [1, 0]], [[-1, 1, 1]]]

        for input_pt, matrix, expected_pt in zip(points, matrices, expected):
            input_pt = torch.tensor(input_pt, device="cpu", dtype=torch.float)
            if not torch.is_tensor(matrix):
                matrix = torch.tensor(matrix, device="cpu", dtype=torch.float)

            expected_pt = torch.tensor(expected_pt, device="cpu", dtype=torch.float)

            while len(expected_pt.shape) < 3:
                expected_pt = expected_pt[None]

            with self.subTest(input=input_pt, matrix=matrix, expected=expected_pt):
                trafo_result = affine_point_transform(input_pt, matrix)
                self.assertTrue(torch.allclose(trafo_result, expected_pt, atol=1e-7))

    def test_affine_image_trafo(self):
        matrix = torch.tensor([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        image_batch = torch.zeros(10, 3, 25, 25, dtype=torch.float, device="cpu")

        target_sizes = [(100, 125), image_batch.shape[2:], (50, 50), (50, 50), (45, 50), (45, 50)]

        for output_size in [None, 50, (45, 50)]:
            for adjust_size in [True, False]:
                target_size = target_sizes.pop(0)

                with self.subTest(adjust_size=adjust_size, target_size=target_size, output_size=output_size):
                    if output_size is not None and adjust_size:
                        with self.assertWarns(UserWarning):
                            result = affine_image_transform(
                                image_batch=image_batch,
                                matrix_batch=matrix,
                                output_size=output_size,
                                adjust_size=adjust_size,
                            )
                    else:
                        result = affine_image_transform(
                            image_batch=image_batch,
                            matrix_batch=matrix,
                            output_size=output_size,
                            adjust_size=adjust_size,
                        )

                    self.assertTupleEqual(result.shape[2:], target_size)

    def test_create_scale(self):
        inputs = [
            {"scale": None, "batchsize": 2, "ndim": 2},
            {"scale": 2, "batchsize": 2, "ndim": 2},
            {"scale": [2, 3], "batchsize": 3, "ndim": 2},
            {"scale": [2, 3, 4], "batchsize": 3, "ndim": 2},
        ]

        expectations = [
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                    [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
                    [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = create_scale(**inp, image_transform=False).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            create_scale([4, 5, 6, 7], batchsize=3, ndim=2)

    def test_create_translation(self):
        inputs = [
            {"offset": None, "batchsize": 2, "ndim": 2},
            {"offset": 2, "batchsize": 2, "ndim": 2},
            {"offset": [2, 3], "batchsize": 3, "ndim": 2},
            {"offset": [2, 3, 4], "batchsize": 3, "ndim": 2},
        ]

        expectations = [
            torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 0, 2], [0, 1, 2], [0, 0, 1]], [[1, 0, 2], [0, 1, 2], [0, 0, 1]]]),
            torch.tensor(
                [
                    [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                    [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                    [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                    [[1, 0, 3], [0, 1, 3], [0, 0, 1]],
                    [[1, 0, 4], [0, 1, 4], [0, 0, 1]],
                ]
            ),
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = create_translation(**inp, image_transform=False).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            create_translation([4, 5, 6, 7], batchsize=3, ndim=2)

    def test_format_rotation(self):
        inputs = [
            {"rotation": None, "batchsize": 2, "ndim": 3},
            {"rotation": 0, "degree": True, "batchsize": 2, "ndim": 2},
        ]
        expectations = [
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = create_rotation(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            create_rotation([4, 5, 6, 7], batchsize=1, ndim=2)

    def test_matrix_parametrization(self):
        inputs = [
            {
                "scale": None,
                "translation": None,
                "rotation": None,
                "batchsize": 2,
                "ndim": 2,
                "dtype": torch.float,
                "image_transform": False,
            },
            {
                "scale": [4, 5],
                "translation": [2, 3],
                "rotation": 90,
                "batchsize": 1,
                "ndim": 2,
                "dtype": torch.float,
                "degree": True,
                "image_transform": False,
            },
            {
                "scale": [4, 5],
                "translation": [2, 3],
                "rotation": 90,
                "batchsize": 1,
                "ndim": 2,
                "dtype": torch.float,
                "degree": True,
                "image_transform": True,
            },
            {
                "scale": [2, 5],
                "translation": [9, 18, 27],
                "rotation": [180, 0, 180],
                "degree": True,
                "batchsize": 3,
                "ndim": 2,
                "dtype": torch.float,
                "image_transform": False,
            },
        ]

        expectations = [
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor([[[0.0, -4.0, -12.0], [5.0, 0.0, 10.0], [0.0, 0.0, 1.0]]]),
            torch.tensor([[[0.0, 1 / 5.0, -2.0], [-1 / 4.0, 0.0, -3.0], [0.0, 0.0, 1.0]]]),
            torch.bmm(
                torch.bmm(
                    torch.tensor(
                        [
                            [[2.0, 0.0, 0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]],
                            [[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]],
                            [[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]],
                        ]
                    ),
                    torch.tensor(
                        [
                            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                        ]
                    ),
                ),
                torch.tensor(
                    [
                        [[1.0, 0.0, 9.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 18.0], [0.0, 1.0, 18.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 27.0], [0.0, 1.0, 27.0], [0.0, 0.0, 1.0]],
                    ]
                ),
            ),
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = parametrize_matrix(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, matrix_to_cartesian(exp), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
