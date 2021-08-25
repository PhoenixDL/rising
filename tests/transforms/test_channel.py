import unittest

import torch

from rising.transforms import ArgMax, OneHot, Permute


class TestChannel(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.rand(3, 3),
        }

    def test_permute(self):
        trafo = Permute({"data": (2, 3, 0, 1), "seg": (2, 0, 3, 1)})
        outp = trafo(**self.batch_dict)
        self.assertEqual(tuple(outp["data"].shape), (3, 3, 1, 1))
        self.assertEqual(tuple(outp["seg"].shape), (3, 1, 3, 1))
        self.assertEqual(tuple(outp["label"].shape), (3, 3))

    def test_onehot(self):
        for dtype in [torch.long, torch.float]:
            with self.subTest(dtype=dtype):
                seg = torch.ones(1, 1, 10, 10, dtype=torch.long)
                trafo = OneHot(num_classes=2, dtype=dtype)
                seg_oh = trafo(**{"seg": seg})
                seg_expected = torch.zeros(1, 2, 10, 10, dtype=dtype)
                seg_expected[:, 1] = 1
                self.assertTrue(seg_oh["seg"].allclose(seg_expected))

    def test_argmax(self):
        for dtype in [torch.long, torch.float]:
            with self.subTest(dtype=dtype):
                seg_onehot = torch.zeros(1, 2, 10, 10, dtype=dtype)
                seg_onehot[:, 1] = 1
                trafo = ArgMax(dim=1)
                out = trafo(**{"seg": seg_onehot})
                expected_out = torch.ones(1, 2, 10, 10, dtype=torch.long)
                self.assertTrue(out["seg"].allclose(expected_out))


if __name__ == "__main__":
    unittest.main()
