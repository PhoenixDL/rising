import unittest
import torch

from rising.transforms import ToDevice


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda:0"
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3, device="cpu"),
            "id": "str"
        }

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device(self):
        trafo = ToDevice(self.device, keys=("data", "seg"))
        outp = trafo(**self.batch_dict)
        for key in ["data", "seg"]:
            self.assertEqual(outp[key].device, torch.device(self.device))
        self.assertEqual(outp["label"].device, torch.device("cpu"))
        self.assertIsInstance(outp["id"], str)


if __name__ == '__main__':
    unittest.main()
