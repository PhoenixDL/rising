import unittest
import torch

from rising.transforms import Permute, OneHot


class TestChannel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.rand(3, 3)
        }

    def test_permute(self):
        trafo = Permute({"data": (2, 3, 0, 1), "seg": (2, 0, 3, 1)})
        outp = trafo(**self.batch_dict)
        self.assertEqual(tuple(outp["data"].shape), (3, 3, 1, 1))
        self.assertEqual(tuple(outp["seg"].shape), (3, 1, 3, 1))
        self.assertEqual(tuple(outp["label"].shape), (3, 3))


if __name__ == '__main__':
    unittest.main()
