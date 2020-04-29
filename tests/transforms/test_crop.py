import unittest
import torch
import random

from rising.random import DiscreteParameter
from rising.transforms.crop import *
from rising.transforms.functional.crop import random_crop, center_crop


class TestCrop(unittest.TestCase):
    def setUp(self) -> None:
        data = torch.zeros(1, 1, 10, 10)
        data[:, :, 4:7, 4:7] = 1
        self.batch = {"data": data}

    def test_center_crop_transform(self):
        for s in range(1, 10):
            trafo = CenterCrop(s)
            crop = trafo(**self.batch)["data"]

            expected = center_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_random_crop_transform(self):
        for s in range(9):
            random.seed(0)
            trafo = RandomCrop(s)
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            _ = random.choices([0])  # internally sample size
            _ = random.choices([0])  # internally sample dist
            expected = random_crop(self.batch["data"], size=s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_center_crop_random_size_transform(self):
        for _ in range(10):
            random.seed(0)
            trafo = CenterCrop(DiscreteParameter([3, 4, 5, 6, 7, 8]))
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            s = random.randrange(3, 8)
            expected = center_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_center_crop_random_size_2_transform(self):
        for _ in range(10):
            random.seed(0)
            trafo = CenterCrop([DiscreteParameter([3, 4, 5]),
                                DiscreteParameter([6, 7, 8])])
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            s = (random.randrange(3, 5), random.randrange(6, 8))
            expected = center_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertSequenceEqual(crop.shape[2:], s)


if __name__ == '__main__':
    unittest.main()
