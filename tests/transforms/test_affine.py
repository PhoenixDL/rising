import unittest
from rising.transforms.affine import Affine, StackedAffine, Translate, Rotate, \
    Scale
import torch
from copy import deepcopy
from rising.utils.affine import matrix_to_cartesian, matrix_to_homogeneous


class AffineTestCase(unittest.TestCase):
    def test_affine(self):
        matrix = torch.tensor([[4., 0., 0.], [0., 5., 0.]])
        image_batch = torch.zeros(10, 3, 25, 25, dtype=torch.float,
                                  device='cpu')
        matrix = matrix.expand(image_batch.size(0), -1, -1).clone()

        target_sizes = [(121, 97), image_batch.shape[2:], (50, 50), (50, 50),
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

        sample = {'data': torch.rand(10, 3, 25, 25)}
        trafos = [
            Scale(5),
            Rotate(45),
            Translate(10)
        ]

        for trafo in trafos:
            with self.subTest(trafo=trafo):
                self.assertIsInstance(trafo(**sample)['data'], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
