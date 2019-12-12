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

    def test_one_hot(self):
        inp = torch.zeros(1, 1, 3, 3).long()
        inp[0, 0, 0, 0] = 1
        batch = {"seg": inp}
        trafo = OneHot(num_classes=2)
        outp = trafo(**batch)
        exp = torch.zeros(1, 2, 3, 3).long()
        exp[0, 0] = 1
        exp[0, 0, 0, 0] = 0
        exp[0, 1, 0, 0] = 1
        self.assertTrue((outp["seg"] == exp).all())


if __name__ == '__main__':
    unittest.main()
