import unittest

import torch

from rising.transforms.functional import to_device_dtype


class TestSpatialFunctional(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda:0"
        self.batch_2d = torch.arange(1, 10).reshape(3, 3)[None, None]

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device_tensor(self):
        outp = to_device_dtype(self.batch_2d, device=self.device)
        self.assertEqual(outp.device, torch.device(self.device))

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device_mapping(self):
        outp = to_device_dtype({"a": self.batch_2d}, device=self.device)
        self.assertEqual(outp["a"].device, torch.device(self.device))
        self.assertIsInstance(outp, dict)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device_iterable(self):
        outp = to_device_dtype((self.batch_2d,), device=self.device)
        self.assertEqual(outp[0].device, torch.device(self.device))
        self.assertIsInstance(outp, tuple)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device_str(self):
        outp = to_device_dtype("test", device=self.device)
        self.assertIsInstance(outp, str)


if __name__ == "__main__":
    unittest.main()
