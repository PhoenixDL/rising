import torch
import unittest
from rising.random import DiscreteParameter, AbstractParameter


class TestAbstract(unittest.TestCase):
    def test_none(self):
        param = DiscreteParameter([None], replacement=True)
        res = param((1,))
        self.assertIsNone(res)

    def test_tensor_like(self):
        param = DiscreteParameter([0., 1., 2.], replacement=True)
        tensor_like = torch.tensor([1, 1, 1]).long()
        res = param(size=(10, 10), tensor_like=tensor_like)
        self.assertEqual(res.dtype, torch.long)
        self.assertTupleEqual(tuple(res.shape), (10, 10))

    def test_abstract_error(self):
        param = AbstractParameter()
        with self.assertRaises(NotImplementedError):
            param.sample((1,))


if __name__ == '__main__':
    unittest.main()
