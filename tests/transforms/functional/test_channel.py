import unittest

import torch

from rising.transforms.functional.channel import one_hot_batch


class TestChannel(unittest.TestCase):
    def test_one_hot_batch_1dim(self):
        inp = torch.tensor([0, 1, 2]).long()
        outp = one_hot_batch(inp)
        exp = torch.eye(3)
        self.assertTrue((outp == exp).all())
        self.assertTrue(outp.dtype == torch.long)

    def test_one_hot_batch_2dim(self):
        for dtype, expected_dtype in zip([None, torch.float], [torch.long, torch.float]):
            with self.subTest(dtype=dtype, expected_dtype=expected_dtype):
                inp = torch.zeros(1, 1, 3, 3).long()
                inp[0, 0, 0, 0] = 1
                outp = one_hot_batch(inp, dtype=dtype)
                exp = torch.zeros(1, 2, 3, 3).long()
                exp[0, 0] = 1
                exp[0, 0, 0, 0] = 0
                exp[0, 1, 0, 0] = 1
                self.assertTrue((outp == exp).all())
                self.assertEqual(outp.dtype, expected_dtype)

    def test_one_hot_batch_float_error(self):
        with self.assertRaises(TypeError):
            inp = torch.zeros(1, 1, 3, 3).float()
            one_hot_batch(inp)


if __name__ == "__main__":
    unittest.main()
