import unittest

import torch

from rising.random import AbstractParameter, DiscreteParameter


class TestAbstract(unittest.TestCase):
    def test_none(self):
        param = DiscreteParameter([None], replacement=True)
        res = param((1,))
        self.assertIsNone(res)

    def test_tensor_like(self):
        param = DiscreteParameter([0.0, 1.0, 2.0], replacement=True)
        tensor_like = torch.tensor([1, 1, 1]).long()
        res = param(size=(10, 10), tensor_like=tensor_like)
        self.assertEqual(res.dtype, torch.long)
        self.assertTupleEqual(tuple(res.shape), (10, 10))

    def test_abstract_error(self):
        param = AbstractParameter()
        with self.assertRaises(NotImplementedError):
            param.sample((1,))


if __name__ == "__main__":
    unittest.main()
