import unittest
from math import isclose

from phdata.transforms.functional.intensity import *


class TestIntensityFunctional(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_2d = torch.rand(3, 3)[None, None]

    def test_norm_range(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_range(inp, 2, 3, per_channel=False)

        self.assertEqual(outp.min().item(), 2)
        self.assertEqual(outp.max().item(), 3)

    def test_norm_range_per_channel(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_range(inp, 2, 3, per_channel=True)

        for c in range(inp.shape[0]):
            self.assertEqual(outp[c].min().item(), 2)
            self.assertEqual(outp[c].max().item(), 3)

    def test_norm_min_max(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_min_max(inp, per_channel=False)

        self.assertEqual(outp.min().item(), 0)
        self.assertEqual(outp.max().item(), 1)

    def test_norm_min_max_per_channels(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_min_max(inp, per_channel=True)

        for c in range(inp.shape[0]):
            self.assertEqual(outp[c].min().item(), 0)
            self.assertEqual(outp[c].max().item(), 1)

    def test_zero_mean_unit_std(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_zero_mean_unit_std(inp, per_channel=False)

        self.assertTrue(isclose(outp.mean().item(), 0, abs_tol=1e-06))
        self.assertTrue(isclose(outp.std().item(), 1, rel_tol=1e-08))

    def test_zero_mean_unit_std_per_channel(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_zero_mean_unit_std(inp, per_channel=True)

        for c in range(inp.shape[0]):
            self.assertTrue(isclose(outp[c].mean().item(), 0, abs_tol=1e-06))
            self.assertTrue(isclose(outp[c].std().item(), 1, rel_tol=1e-08))

    def test_mean_std(self):
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_mean_std(inp, inp.mean().item(), inp.std().item(), per_channel=False)

        self.assertTrue(isclose(outp.mean().item(), 0, abs_tol=1e-06))
        self.assertTrue(isclose(outp.std().item(), 1, rel_tol=1e-08))

    def test_mean_std_per_channel(self):
        inp = (self.batch_2d[0] * 10) + 1
        channel_mean = [inp[c].mean().item() for c in range(inp.shape[0])]
        channel_std = [inp[c].std().item() for c in range(inp.shape[0])]
        outp = norm_mean_std(inp, channel_mean, channel_std, per_channel=True)

        for c in range(inp.shape[0]):
            self.assertTrue(isclose(outp[c].mean().item(), 0, abs_tol=1e-06))
            self.assertTrue(isclose(outp[c].std().item(), 1, rel_tol=1e-08))

    def test_mean_std_per_channel_scalar(self):
        # TEST: add error sensitive test to check correct behavior
        inp = (self.batch_2d[0] * 10) + 1
        outp = norm_mean_std(inp, inp.mean().item(), inp.std().item(), per_channel=True)

    def test_add_noise(self):
        outp = add_noise(self.batch_2d.clone(), 'normal', mean=75, std=1)
        diff = (outp - self.batch_2d).abs().mean()
        self.assertTrue(diff > 50)


if __name__ == '__main__':
    unittest.main()
