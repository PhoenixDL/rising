import torch
import random
import unittest

from tests.test_transforms import chech_data_preservation
from rising.transforms.spatial import *


class TestSpatialTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.randint(0, 3, (1, 1, 3, 3)).long(),
            "label": torch.arange(3)
        }

    def test_mirror_transform(self):
        trafo = MirrorTransform((0, 1), prob=1)
        outp = trafo(**self.batch_dict)

        self.assertTrue((outp["data"][0, 0] == torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]])).all())
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = MirrorTransform((0, 1), prob=0)
        data_orig = self.batch_dict["data"].clone()
        outp = trafo(**self.batch_dict)
        self.assertTrue((outp["data"] == data_orig).all())

    def test_rot90_transform(self):
        random.seed(0)
        trafo = Rot90Transform((0, 1), prob=1)
        outp = trafo(**self.batch_dict)
        self.assertTrue((outp["data"][0, 0] == torch.tensor([[3, 6, 9], [2, 5, 8], [1, 4, 7]])).all())
        self.assertEqual(trafo.dims, (0, 1))
        self.assertTrue(chech_data_preservation(trafo, self.batch_dict))

        trafo = Rot90Transform((0, 1), prob=0)
        data_orig = self.batch_dict["data"].clone()
        outp = trafo(**self.batch_dict)
        self.assertTrue((outp["data"] == data_orig).all())

    def test_resize_transform(self):
        trafo = ResizeTransform((2, 2))
        out = trafo(**self.batch_dict)
        expected = torch.tensor([[1, 2], [4, 5]])
        self.assertTrue((out["data"] == expected).all())

    def test_zoom_transform(self):
        _range = (2., 3.)
        random.seed(0)
        scale_factor = random.uniform(*_range)

        trafo = ZoomTransform(random_args=_range)
        random.seed(0)
        out = trafo(**self.batch_dict)

        expected = resize(self.batch_dict["data"], mode="nearest", scale_factor=scale_factor)
        self.assertTrue((out["data"] == expected).all())


if __name__ == '__main__':
    unittest.main()
