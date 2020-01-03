import unittest
import torch
from rising.transforms.functional.affine import _check_new_img_size, \
    affine_point_transform, affine_image_transform
from rising.utils.affine import parametrize_matrix, matrix_to_homogeneous, matrix_to_cartesian, \
    matrix_revert_coordinate_order
from rising.utils.checktype import check_scalar


class AffineTestCase(unittest.TestCase):

    def test_check_image_size(self):
        images = [torch.rand(11, 2, 3, 4, 5), torch.rand(11, 2, 3, 4), torch.rand(11, 2, 3, 3)]

        img_sizes = [
            [3, 4, 5], [3, 4], 3
        ]
        scales = [
            torch.tensor([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]]),
            torch.tensor([[2., 0.], [0., 3.]]),
            torch.tensor([[2., 0.], [0., 3.]])
        ]

        rots = [[45., 90., 135.], [45.], [45.]]

        trans = [[0., 10., 20.], [10., 20.], [10., 20.]]

        edges = [
            [
                [0., 0., 0., 1.], [0., 0., 5., 1.], [0., 4., 0., 1.], [0., 4., 5., 1.],
                [3., 0., 0., 1.], [3., 0., 5., 1.], [3., 4., 0., 1.], [3., 4., 5., 1.]
            ],
            [
                [0., 0., 1.], [0., 4., 1.], [3., 0., 1.], [3., 4., 1.]
            ],
            [
                [0., 0., 1.], [0., 3., 1.], [3., 0., 1.], [3., 3., 1.]
            ]
        ]

        for img, size, scale, rot, tran, edge_pts in zip(images, img_sizes,
                                                         scales, rots, trans,
                                                         edges):
            ndim = scale.size(-1)
            with self.subTest(ndim=ndim):
                affine = matrix_to_homogeneous(
                    parametrize_matrix(scale=scale, rotation=rot,
                                       translation=tran, degree=True,
                                       batchsize=1, ndim=ndim, dtype=torch.float))

                edge_pts = torch.tensor(edge_pts, dtype=torch.float)
                edge_pts[edge_pts > 1] = edge_pts[edge_pts > 1] - 1
                img = img.to(torch.float)

                new_edges = torch.bmm(edge_pts.unsqueeze(0),
                                      matrix_revert_coordinate_order(affine.clone()).permute(0, 2, 1))

                img_size = (new_edges.max(dim=1)[0] - new_edges.min(dim=1)[0])[0]

                fn_result = _check_new_img_size(size,
                                                matrix_to_cartesian(
                                                    affine.expand(img.size(0), -1, -1).clone()))

                self.assertTrue(torch.allclose(img_size[:-1] + 1,
                                               fn_result))

        with self.assertRaises(ValueError):
            _check_new_img_size([2, 3, 4, 5], torch.rand(11, 2, 2, 3, 4, 5))

    def test_affine_point_transform(self):
        points = [
            [[[0, 1], [1, 0]]],
            [[[0, 0, 1]]]
        ]
        matrices = [
            torch.tensor([[[1., 0.], [0., 5.]]]),
            parametrize_matrix(scale=1,
                               translation=0,
                               rotation=[0, 0, 90],
                               degree=True, batchsize=1,
                               ndim=3, dtype=torch.float,
                               device='cpu')
        ]
        expected = [
            [[0, 1], [5, 0]],
            [[0, 1, 0]]
        ]

        for input_pt, matrix, expected_pt in zip(points, matrices, expected):
            input_pt = torch.tensor(input_pt, device='cpu', dtype=torch.float)
            if not torch.is_tensor(matrix):
                matrix = torch.tensor(matrix, device='cpu', dtype=torch.float)

            expected_pt = torch.tensor(expected_pt, device='cpu',
                                       dtype=torch.float)

            while len(expected_pt.shape) < 3:
                expected_pt = expected_pt[None]

            with self.subTest(input=input_pt, matrix=matrix,
                              expected=expected_pt):
                trafo_result = affine_point_transform(input_pt, matrix)
                self.assertTrue(torch.allclose(trafo_result, expected_pt,
                                               atol=1e-7))

    def test_affine_image_trafo(self):

        matrix = torch.tensor([[4., 0., 0.], [0., 5., 0.]])
        image_batch = torch.zeros(10, 3, 25, 25, dtype=torch.float,
                                  device='cpu')

        target_sizes = [(121, 97), image_batch.shape[2:], (50, 50), (50, 50),
                        (45, 50), (45, 50)]

        for output_size in [None, 50, (45, 50)]:
            for adjust_size in [True, False]:
                target_size = target_sizes.pop(0)

                with self.subTest(adjust_size=adjust_size,
                                  target_size=target_size,
                                  output_size=output_size):
                    if output_size is not None and adjust_size:
                        with self.assertWarns(UserWarning):
                            result = affine_image_transform(
                                image_batch=image_batch,
                                matrix_batch=matrix,
                                output_size=output_size,
                                adjust_size=adjust_size)
                    else:
                        result = affine_image_transform(
                            image_batch=image_batch,
                            matrix_batch=matrix,
                            output_size=output_size,
                            adjust_size=adjust_size)

                    self.assertTupleEqual(result.shape[2:], target_size)


if __name__ == '__main__':
    unittest.main()
