import random
import unittest

import torch

from rising.transforms.functional import center_crop, random_crop


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.zeros(1, 1, 10, 10)
        self.data[:, :, 4:7, 4:7] = 1

    def test_center_crop(self):
        for s in range(1, 4):
            crop = center_crop(self.data, size=float(s))
            expected = torch.ones(1, 1, s, s)
            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_random_crop(self):
        torch.manual_seed(0)
        h = torch.randint(0, 7, (1,)).item()
        w = torch.randint(0, 7, (1,)).item()
        expected = self.data[:, :, h : h + 3, w : w + 3]
        torch.manual_seed(0)
        crop = random_crop(self.data, size=3.0)
        self.assertTrue((crop == expected).all())
        self.assertTrue(all([_s == 3 for _s in crop.shape[2:]]))

    def test_random_crop_error(self):
        with self.assertRaises(TypeError):
            random_crop(self.data, size=13)

        with self.assertRaises(TypeError):
            random_crop(self.data, size=3, dist=7)

    def test_random_crop_random(self):
        # dummy test to check a variety of sizes and random values
        checks = 100
        for _ in range(checks):
            s = random.randrange(0, 10)
            crop = random_crop(self.data, size=s)
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))


if __name__ == "__main__":
    unittest.main()
