import unittest

import torch
from rising.transforms.functional.channel import one_hot_batch


class TestChannel(unittest.TestCase):
    def test_one_hot_batch_1dim(self):
        inp = torch.tensor([0, 1, 2]).long()
        outp = one_hot_batch(inp)
        exp = torch.eye(3)
        self.assertTrue((outp == exp).all())
        self.assertTrue(outp.dtype == torch.long)

    def test_one_hot_batch_2dim(self):
        inp = torch.zeros(1, 1, 3, 3).long()
        inp[0, 0, 0, 0] = 1
        # inp[0, 0, 1, 1] = 2
        # inp[0, 0, 2, 2] = 3
        outp = one_hot_batch(inp)
        exp = torch.zeros(1, 2, 3, 3).long()
        exp[0, 1, 0, 0] = 1
        # exp[0, 2, 1, 1] = 1
        # exp[0, 3, 2, 2] = 1
        self.assertTrue((outp == exp).all())


if __name__ == '__main__':
    unittest.main()
