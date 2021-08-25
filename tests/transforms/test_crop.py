import random
import unittest

import torch

from rising.random import DiscreteParameter
from rising.transforms import CenterCrop, RandomCrop
from rising.transforms.functional import center_crop, random_crop


class TestCrop(unittest.TestCase):
    def setUp(self) -> None:
        data = torch.zeros(1, 1, 10, 10)
        data[:, :, 4:7, 4:7] = 1
        self.batch = {"data": data, "seg": data.clone()}

    def test_center_crop_transform(self):
        for s in range(1, 10):
            trafo = CenterCrop(s, keys=("data", "seg"))
            crop = trafo(**self.batch)

            expected = center_crop(self.batch["data"], s)

            self.assertTrue(expected.allclose(crop["data"]))
            self.assertTrue(expected.allclose(crop["seg"]))
            self.assertTrue(all([_s == s for _s in crop["data"].shape[2:]]))
            self.assertTrue(all([_s == s for _s in crop["seg"].shape[2:]]))

    def test_random_crop_transform(self):
        for s in range(1, 10):
            torch.manual_seed(s)
            trafo = RandomCrop(s, keys=("data", "seg"))
            crop = trafo(**self.batch)

            random.seed(0)
            _ = random.choices([0])  # internally sample size in transform
            _ = random.choices([0])  # internally sample dist in transform
            torch.manual_seed(s)  # seed random_crop
            expected = random_crop(self.batch["data"], size=s)

            self.assertTrue(expected.allclose(crop["data"]))
            self.assertTrue(expected.allclose(crop["seg"]))
            self.assertTrue(all([_s == s for _s in crop["data"].shape[2:]]))
            self.assertTrue(all([_s == s for _s in crop["seg"].shape[2:]]))

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
            trafo = CenterCrop([DiscreteParameter([3, 4, 5]), DiscreteParameter([6, 7, 8])])
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            s = (random.randrange(3, 5), random.randrange(6, 8))
            expected = center_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertSequenceEqual(crop.shape[2:], s)


if __name__ == "__main__":
    unittest.main()
