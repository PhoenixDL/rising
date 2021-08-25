import unittest

import torch

from rising.transforms.functional import tensor_op


class TestSpatialFunctional(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_2d = torch.arange(1, 10).reshape(3, 3)[None, None]

    def test_tensor_op_tensor(self):
        outp = tensor_op(self.batch_2d, "permute", 2, 3, 0, 1)
        self.assertEqual(tuple(outp.shape), (3, 3, 1, 1))

    def test_tensor_op_seq(self):
        outp = tensor_op([self.batch_2d], "permute", 2, 3, 0, 1)
        self.assertEqual(tuple(outp[0].shape), (3, 3, 1, 1))

    def test_tensor_op_map(self):
        outp = tensor_op({"a": self.batch_2d}, "permute", 2, 3, 0, 1)
        self.assertEqual(tuple(outp["a"].shape), (3, 3, 1, 1))

    def test_tensor_op_str(self):
        outp = tensor_op("str", "permute", 2, 3, 0, 1)
        self.assertIsInstance(outp, str)


if __name__ == "__main__":
    unittest.main()
