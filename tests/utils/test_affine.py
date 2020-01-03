import unittest
from rising.utils.affine import points_to_homogeneous, matrix_to_homogeneous, \
    matrix_to_cartesian, points_to_cartesian, matrix_revert_coordinate_order, \
    get_batched_eye, _format_scale, _format_translation, deg_to_rad, \
    _format_rotation, parametrize_matrix, assemble_matrix_if_necessary
import torch
import math


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
            torch.tensor([[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23]],
                          [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26]]])
        ]

        expectations = [
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 5], [3, 4, 6], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]], [[5, 6, 0], [7, 8, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 10], [3, 4, 11], [0, 0, 1]], [[5, 6, 12], [7, 8, 13], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]],
                          [[10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23], [0, 0, 0, 1]],
                          [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26], [0, 0, 0, 1]]])
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
            torch.tensor([[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23]],
                          [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26]]])
        ]

        inputs = [
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 5], [3, 4, 6], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 0], [3, 4, 0], [0, 0, 1]], [[5, 6, 0], [7, 8, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 10], [3, 4, 11], [0, 0, 1]], [[5, 6, 12], [7, 8, 13], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]],
                          [[10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0], [0, 0, 0, 1]]]),
            torch.tensor([[[1, 2, 3, 21], [4, 5, 6, 22], [7, 8, 9, 23], [0, 0, 0, 1]],
                          [[10, 11, 12, 24], [13, 14, 15, 25], [16, 17, 18, 26], [0, 0, 0, 1]]])
        ]

        keep_square = True
        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(matrix_to_cartesian(inp, keep_square=keep_square), exp))
                keep_square = not keep_square

    def test_matrix_coordinate_order(self):
        inputs = [
            torch.tensor([[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]])
        ]

        expectations = [
            torch.tensor([[[5, 4, 6],
                           [2, 1, 3],
                           [7, 8, 9]]])
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(matrix_revert_coordinate_order(inp), exp))
                # self.assertTrue(torch.allclose(inp, matrix_revert_coordinate_order(exp)))

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

    def test_format_scale(self):
        inputs = [
            {'scale': None, 'batchsize': 2, 'ndim': 2},
            {'scale': 2, 'batchsize': 2, 'ndim': 2},
            {'scale': [2, 3], 'batchsize': 3, 'ndim': 2},
            {'scale': [2, 3, 4], 'batchsize': 3, 'ndim': 2},
            {'scale': [[2, 3], [4, 5]], 'batchsize': 3, 'ndim': 2},
        ]

        expectations = [
            torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),
            torch.tensor([[[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]],
                          [[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]]]),
            torch.tensor([[[2., 0., 0.], [0., 3., 0.], [0., 0., 1.]],
                          [[2., 0., 0.], [0., 3., 0.], [0., 0., 1.]],
                          [[2., 0., 0.], [0., 3., 0.], [0., 0., 1.]]]),
            torch.tensor([[[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]],
                          [[3., 0., 0.], [0., 3., 0.], [0., 0., 1.]],
                          [[4., 0., 0.], [0., 4., 0.], [0., 0., 1.]]]),
            torch.tensor([[[2, 3, 0], [4, 5, 0], [0, 0, 1]],
                          [[2, 3, 0], [4, 5, 0], [0, 0, 1]],
                          [[2, 3, 0], [4, 5, 0], [0, 0, 1]]])

        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = _format_scale(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            _format_scale([4, 5, 6, 7], batchsize=3, ndim=2)

    def test_format_translation(self):
        inputs = [
            {'offset': None, 'batchsize': 2, 'ndim': 2},
            {'offset': 2, 'batchsize': 2, 'ndim': 2},
            {'offset': [2, 3], 'batchsize': 3, 'ndim': 2},
            {'offset': [2, 3, 4], 'batchsize': 3, 'ndim': 2},
            {'offset': [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],
             'batchsize': 3, 'ndim': 2},
            {'offset': [[[1, 2, 3], [4, 5, 6]],
                        [[10, 11, 12], [13, 14, 15]],
                        [[19, 20, 21], [22, 23, 24]]],
             'batchsize': 3, 'ndim': 2}

        ]

        expectations = [
            torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
            torch.tensor([[[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                          [[1, 0, 2], [0, 1, 2], [0, 0, 1]]]),
            torch.tensor([[[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                          [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                          [[1, 0, 2], [0, 1, 3], [0, 0, 1]]]),
            torch.tensor([[[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                          [[1, 0, 3], [0, 1, 3], [0, 0, 1]],
                          [[1, 0, 4], [0, 1, 4], [0, 0, 1]]]),
            torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                          [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]),
            torch.tensor([[[1, 2, 3], [4, 5, 6], [0, 0, 1]],
                          [[10, 11, 12], [13, 14, 15], [0, 0, 1]],
                          [[19, 20, 21], [22, 23, 24], [0, 0, 1]]])

        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = _format_translation(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            _format_translation([4, 5, 6, 7], batchsize=3, ndim=2)

    def test_deg_to_rad(self):
        inputs = [
            torch.tensor([tmp * 45. for tmp in range(9)]),
        ]

        expectations = [
            torch.tensor([tmp * math.pi / 4 for tmp in range(9)])
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                self.assertTrue(torch.allclose(deg_to_rad(inp), exp, atol=1e-6))

    def test_format_rotation(self):
        inputs = [
            {'rotation': None, 'batchsize': 2, 'ndim': 3},
            {'rotation': 0, 'degree': True, 'batchsize': 2, 'ndim': 2},
            {'rotation': [180, 0, 180], 'degree': True, 'batchsize': 2, 'ndim': 3},
            {'rotation': [180, 0, 180], 'degree': True, 'batchsize': 3, 'ndim': 2},
            {'rotation': [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18]]],
             'batchsize': 2, 'ndim': 2},
            {'rotation': [[[1, 2, 3], [4, 5, 6]],
                          [[10, 11, 12], [13, 14, 15]]],
             'batchsize': 2, 'ndim': 2},
            {'rotation': [[1, 2], [3, 4]], 'batchsize': 3, 'ndim': 2, 'degree': False}

        ]
        expectations = [
            torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]],
                          [[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]]]),
            torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),
            torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]],
                          [[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]]]),
            torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                          [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                          [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]),
            torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]),
            torch.tensor([[[1., 2., 3.], [4., 5., 6.], [0., 0., 1.]],
                          [[10., 11., 12.], [13., 14., 15.], [0., 0., 1.]]]),
            torch.tensor([[[1., 2., 0.], [3., 4., 0.], [0., 0., 1.]],
                          [[1., 2., 0.], [3., 4., 0.], [0., 0., 1.]],
                          [[1., 2., 0.], [3., 4., 0.], [0., 0., 1.]]])
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = _format_rotation(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            _format_rotation([4, 5, 6, 7], batchsize=1, ndim=2)

    def test_matrix_parametrization(self):
        inputs = [
            {'scale': None, 'translation': None, 'rotation': None, 'batchsize': 2, 'ndim': 2,
             'dtype': torch.float},
            {'scale': [[2, 3], [4, 5]], 'translation': [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                                                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],
             'rotation': [180, 0, 180], 'degree': True, 'batchsize': 3,
             'ndim': 2, 'dtype':torch.float}
        ]

        expectations = [
            torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),

            torch.bmm(torch.bmm(torch.tensor([[[2., 3., 0], [4., 5., 0.], [0., 0., 1.]],
                                              [[2., 3., 0.], [4., 5., 0.], [0., 0., 1.]],
                                              [[2., 3., 0.], [4., 5., 0.], [0., 0., 1.]]]),
                                torch.tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
                                              [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
                                              [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]])),
                      torch.tensor([[[1., 2., 3.], [4., 5., 6.], [0., 0., 1.]],
                                    [[10., 11., 12.], [13., 14., 15.], [0., 0., 1.]],
                                    [[19., 20., 21.], [22., 23., 24.], [0., 0., 1.]]]))

        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = parametrize_matrix(**inp).to(exp.dtype)
                self.assertTrue(torch.allclose(res, matrix_to_cartesian(exp), atol=1e-6))

    def test_necessary_assembly(self):
        inputs = [
            {'matrix': None, 'translation': [2, 3], 'ndim':2, 'batchsize': 3,
             'dtype': torch.float},
            {'matrix': [[1., 0., 4.], [0., 1., 5.], [0., 0., 1.]], 'translation': [2, 3], 'ndim': 2, 'batchsize': 3,
             'dtype': torch.float},
            {'matrix': [[1., 0., 4.], [0., 1., 5.]], 'translation': [2, 3], 'ndim': 2, 'batchsize': 3,
             'dtype': torch.float}

        ]
        expectations = [
            torch.tensor([[[1., 0., 2.], [0., 1., 3.]],
                          [[1., 0., 2.], [0., 1., 3.]],
                          [[1., 0., 2.], [0., 1., 3.]]]),
            torch.tensor([[[1., 0., 4.], [0., 1., 5.]],
                          [[1., 0., 4.], [0., 1., 5.]],
                          [[1., 0., 4.], [0., 1., 5.]]]),
            torch.tensor([[[1., 0., 4.], [0., 1., 5.]],
                          [[1., 0., 4.], [0., 1., 5.]],
                          [[1., 0., 4.], [0., 1., 5.]]])
        ]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expected=exp):
                res = assemble_matrix_if_necessary(**inp, degree=False,
                                                   device='cpu', scale=None, rotation=None).to(exp.dtype)
                self.assertTrue(torch.allclose(res, exp, atol=1e-6))

        with self.assertRaises(ValueError):
            assemble_matrix_if_necessary(matrix=[1, 2, 3, 4, 5], scale=None,
                                         rotation=None, translation=None,
                                         degree=False, dtype=torch.float,
                                         device='cpu', batchsize=1, ndim=2)


if __name__ == '__main__':
    unittest.main()
