import unittest

import numpy as np
import torch

from rising.ops import np_one_hot, torch_one_hot


class TestOneHot(unittest.TestCase):
    def test_torch_one_hot(self):
        target = torch.tensor([0, 1, 2])
        target_onehot = torch_one_hot(target, 3)
        expected_onehot = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue((expected_onehot == target_onehot).all())

    def test_torch_one_hot_auto(self):
        target = torch.tensor([0, 1, 2])
        target_onehot = torch_one_hot(target)
        expected_onehot = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue((expected_onehot == target_onehot).all())

    def test_np_one_hot(self):
        target = np.array([0, 1, 2])
        target_onehot = np_one_hot(target, 3)
        expected_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue((expected_onehot == target_onehot).all())


if __name__ == "__main__":
    unittest.main()
