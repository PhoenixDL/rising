import unittest
import torch
import random
from math import isclose

from tests.test_transforms import chech_data_preservation
from phdata.transforms.intensity import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3)
        }

    def test_clamp_transform(self):
        trafo = ClampTransform(0, 1)

        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue((outp["data"] == torch.ones_like(outp["data"])).all())

    def test_norm_range_transform(self):
        trafo = NormRangeTransform(0.1, 0.2, per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormRangeTransform(0.1, 0.2, per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].min().item(), 0.1, abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].max().item(), 0.2, abs_tol=1e-6))

    def test_norm_min_max_transform(self):
        trafo = NormMinMaxTransform(per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormMinMaxTransform(per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].min().item(), 0., abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].max().item(), 1., abs_tol=1e-6))

    def test_norm_zero_mean_transform(self):
        trafo = NormZeroMeanUnitStdTransform(per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormZeroMeanUnitStdTransform(per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].mean().item(), 0., abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].std().item(), 1., abs_tol=1e-6))

    def test_norm_std_transform(self):
        mean = self.batch_dict["data"].mean().item()
        std = self.batch_dict["data"].std().item()
        trafo = NormMeanStdTransform(mean=mean, std=std, per_channel=False)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = NormMeanStdTransform(mean=mean, std=std, per_channel=True)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        outp = trafo(**self.batch_dict)
        self.assertTrue(isclose(outp["data"].mean().item(), 0., abs_tol=1e-6))
        self.assertTrue(isclose(outp["data"].std().item(), 1., abs_tol=1e-6))

    def test_noise_transform(self):
        trafo = NoiseTransform('normal', mean=75, std=1)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))
        self.check_noise_distance(trafo)

    def test_expoential_noise_transform(self):
        trafo = ExponentialNoiseTransform(lambd=0.0001)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))
        self.check_noise_distance(trafo)

    def test_gaussian_noise_transform(self):
        trafo = GaussianNoiseTransform(mean=75, std=1)
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))
        self.check_noise_distance(trafo)

    def check_noise_distance(self, trafo, min_diff=50):
        outp = trafo(**self.batch_dict)
        comp_diff = (outp["data"] - self.batch_dict["data"]).mean().item()
        self.assertTrue(comp_diff > min_diff)


if __name__ == '__main__':
    unittest.main()
