import unittest
import torch
import random

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
            expected = random_crop(self.batch["data"], size=s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_center_crop_random_size_transform(self):
        for _ in range(10):
            random.seed(0)
            trafo = CenterCropRandomSize((3, 8))
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            s = random.randrange(3, 8)
            expected = center_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))

    def test_random_crop_random_size_transform(self):
        for _ in range(10):
            random.seed(0)
            trafo = RandomCropRandomSize((3, 8))
            crop = trafo(**self.batch)["data"]

            random.seed(0)
            s = random.randrange(3, 8)
            expected = random_crop(self.batch["data"], s)

            self.assertTrue((crop == expected).all())
            self.assertTrue(all([_s == s for _s in crop.shape[2:]]))


if __name__ == '__main__':
    unittest.main()
