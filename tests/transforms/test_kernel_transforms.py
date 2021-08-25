import random
import unittest

import torch

from rising.transforms import GaussianSmoothing, KernelTransform


class TestKernelTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": torch.zeros(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3),
        }

    def test_kernel_transform_get_conv(self):
        conv = KernelTransform.get_conv(1)
        self.assertEqual(conv, torch.nn.functional.conv1d)

        conv = KernelTransform.get_conv(2)
        self.assertEqual(conv, torch.nn.functional.conv2d)

        conv = KernelTransform.get_conv(3)
        self.assertEqual(conv, torch.nn.functional.conv3d)

        with self.assertRaises(TypeError):
            conv = KernelTransform.get_conv(4)

    def test_kernel_transform_error(self):
        with self.assertRaises(NotImplementedError):
            trafo = KernelTransform(in_channels=1, kernel_size=3, std=1, dim=2, stride=1, padding=1)

    def test_gaussian_smoothing_transform(self):
        # TODO: Test: !!!!Implement sensitive tests!!!!
        trafo = GaussianSmoothing(in_channels=1, kernel_size=3, std=1, dim=2, stride=1, padding=1)
        self.batch_dict["data"][0, 0, 1] = 1
        outp = trafo(**self.batch_dict)


if __name__ == "__main__":
    unittest.main()
