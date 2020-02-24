import unittest
from rising.transforms.affine import Affine, StackedAffine, Translate, Rotate, \
    Scale, Resize
import torch
from copy import deepcopy
from rising.utils.affine import matrix_to_cartesian, matrix_to_homogeneous


class AffineTestCase(unittest.TestCase):
    def test_affine(self):
        matrix = torch.tensor([[4., 0., 0.], [0., 5., 0.]])
        image_batch = torch.zeros(10, 3, 25, 25, dtype=torch.float,
                                  device='cpu')
        matrix = matrix.expand(image_batch.size(0), -1, -1).clone()

        target_sizes = [(100, 125), image_batch.shape[2:], (50, 50), (50, 50),
                        (45, 50), (45, 50)]

        for output_size in [None, 50, (45, 50)]:
            for adjust_size in [True, False]:
                target_size = target_sizes.pop(0)

                with self.subTest(adjust_size=adjust_size,
                                  target_size=target_size,
                                  output_size=output_size):
                    trafo = Affine(matrix=matrix, adjust_size=adjust_size,
                                   output_size=output_size)
                    sample = {'data': image_batch, 'label': 4}
                    if output_size is not None and adjust_size:
                        with self.assertWarns(UserWarning):
                            result = trafo(**sample)
                    else:
                        result = trafo(**sample)
                    self.assertTupleEqual(result['data'].shape[2:],
                                          target_size)

                    self.assertEqual(sample['label'], result['label'])

    def test_affine_assemble_matrix(self):
        matrices = [
            [[1., 0.], [0., 1.]],
            [[1., 0., 1.], [0., 1., 1.]],
            [[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]],
            None,
            [0., 1., 1., 0.]
        ]
        expected_matrices = [
            torch.tensor([[1., 0., 0.], [0., 1., 0.]])[None],
            torch.tensor([[1., 0., 1.], [0., 1., 1.]])[None],
            torch.tensor([[1., 0., 1.], [0., 1., 1.]])[None],
            None,
            None,
        ]
        value_error = [False, False, False, True, True]
        batch = {"data": torch.zeros(1, 1, 10, 10)}

        for matrix, expected, ve in zip(matrices, expected_matrices, value_error):
            with self.subTest(matrix=matrix, expected=expected):
                trafo = Affine(matrix=matrix)
                if ve:
                    with self.assertRaises(ValueError):
                        assembled = trafo.assemble_matrix(**batch)
                else:
                    assembled = trafo.assemble_matrix(**batch)
                    self.assertTrue(expected.allclose(assembled))

    def test_affine_stacking(self):
        affines = [
            Affine(scale=1),
            [[1., 0., 0.], [0., 1., 0.]],
            torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
            StackedAffine(Affine(scale=1), Affine(scale=1))
        ]

        for first_affine in deepcopy(affines):
            for second_affine in deepcopy(affines):
                if not isinstance(first_affine, Affine) and not isinstance(second_affine, Affine):
                    continue

                if torch.is_tensor(first_affine):
                    # TODO: Remove this, once this has been fixed in PyTorch:
                    #  PR: https://github.com/pytorch/pytorch/pull/31769
                    continue

                with self.subTest(first_affine=first_affine,
                                  second_affine=second_affine):
                    result = first_affine + second_affine
                    self.assertIsInstance(result, StackedAffine)

    def test_stacked_transformation_assembly(self):
        first_matrix = torch.tensor([[[2., 0., 1.], [0., 3., 2.]]])
        second_matrix = torch.tensor([[[4., 0., 3.], [0., 5., 4.]]])
        trafo = StackedAffine([first_matrix, second_matrix])

        sample = {'data': torch.rand(1, 3, 25, 25)}

        matrix = trafo.assemble_matrix(**sample)

        target_matrix = matrix_to_cartesian(
            torch.bmm(
                matrix_to_homogeneous(first_matrix),
                matrix_to_homogeneous(second_matrix)
            )
        )

        self.assertTrue(torch.allclose(matrix, target_matrix))

    def test_affine_subtypes(self):
        sample = {'data': torch.rand(1, 3, 25, 30)}

        trafos = [
            Scale([5, 3], adjust_size=True),
            Resize([50, 90]),
            Rotate([90], adjust_size=True, degree=True),
        ]

        expected_sizes = [
            (5, 10),
            (50, 90),
            (30, 25),
        ]

        for trafo, expected_size in zip(trafos, expected_sizes):
            with self.subTest(trafo=trafo, exp_size=expected_size):
                result = trafo(**sample)['data']
                self.assertIsInstance(result, torch.Tensor)
                self.assertTupleEqual(expected_size, result.shape[-2:])

    def test_translation_assemble_matrix_with_pixel(self):
        trafo = Translate([1, 10, 100], unit='pixel')
        sample = {'data': torch.rand(3, 3, 100, 100)}
        expected = torch.tensor([[1., 0., -0.01], [0., 1., -0.01],
                                 [1., 0., -0.1], [0., 1., -0.1],
                                 [1., 0., -1.], [0., 1., -1.]])

        trafo.assemble_matrix(**sample)
        self.assertTrue(expected.allclose(expected))


if __name__ == '__main__':
    unittest.main()
