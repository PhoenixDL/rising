import unittest

import torch

from rising.transforms.functional import mirror, resize_native, rot90


class TestSpatialFunctional(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_2d = torch.arange(1, 10).reshape(3, 3)[None, None]

    def test_mirror_dim0(self):
        outp = mirror(self.batch_2d, 0)
        expected = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        self.assertTrue((outp == expected).all())

    def test_mirror_dim1(self):
        inp = self.batch_2d.clone()
        outp = mirror(inp, 1)
        expected = torch.tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
        self.assertTrue((outp == expected).all())

    def test_rot90(self):
        inp = self.batch_2d.clone()
        outp = rot90(inp, 1, (0, 1))
        expected = torch.tensor([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
        self.assertTrue((outp == expected).all())

    def test_resize(self):
        out = resize_native(self.batch_2d.float(), (2, 2), preserve_range=True)
        expected = torch.tensor([[1, 2], [4, 5]])
        self.assertTrue((out == expected).all())


if __name__ == "__main__":
    unittest.main()
