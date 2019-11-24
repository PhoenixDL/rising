import torch
import random
import unittest


from rising.transforms.kernel import *


# TODO: Test: !!!!Implement sensitive tests!!!!

class TestKernelTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": torch.zeros(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3)
        }

    def test_gaussian_smoothing_transform(self):
        trafo = GaussianSmoothingTransform(in_channels=1, kernel_size=3, std=1,
                                           dim=2, stride=1, padding=1)
        self.batch_dict["data"][0, 0, 1] = 1
        outp = trafo(**self.batch_dict)
        print(outp["data"].shape)


if __name__ == '__main__':
    unittest.main()
