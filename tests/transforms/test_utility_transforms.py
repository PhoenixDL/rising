import unittest
import torch
from rising.transforms.utility import *


class TestUtilityTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_do_nothing_transform(self):
        inp = {"data": 0, "seg": 1}
        trafo = DoNothingTransform()
        outp = trafo(**inp)
        self.assertEqual(outp["data"], 0)
        self.assertEqual(outp["seg"], 1)


if __name__ == '__main__':
    unittest.main()
