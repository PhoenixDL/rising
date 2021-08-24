import unittest

import numpy as np
import torch

from rising.transforms import OneHot, TensorOp, ToDevice, ToDtype, ToTensor


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda:0"
        self.batch_dict = {
            "data": torch.arange(1, 10).reshape(1, 1, 3, 3).float(),
            "seg": torch.rand(1, 1, 3, 3),
            "label": torch.arange(3, device="cpu"),
            "id": "str",
        }

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_to_device(self):
        trafo = ToDevice(self.device, keys=("data", "seg"))
        outp = trafo(**self.batch_dict)
        for key in ["data", "seg"]:
            self.assertEqual(outp[key].device, torch.device(self.device))
        self.assertEqual(outp["label"].device, torch.device("cpu"))
        self.assertIsInstance(outp["id"], str)

    def test_to_dtype(self):
        trafo = ToDtype(dtype=torch.long, keys=("data",))
        outp = trafo(**self.batch_dict)
        self.assertEqual(outp["data"].dtype, torch.long)

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

    def test_to_tensor(self):
        inp = {"data": np.random.rand(10, 10), "seg": np.random.rand(10, 10), "bbox": (0, 1, 2)}
        trafo = ToTensor(("data", "seg"))
        outp = trafo(**inp)
        self.assertTrue(torch.is_tensor(outp["data"]))
        self.assertTrue(torch.is_tensor(outp["seg"]))
        self.assertIsInstance(outp["bbox"], tuple)
        self.assertIsInstance(outp["bbox"][0], int)

    def test_tensor_op(self):
        trafo = TensorOp("transpose_", 1, 2)
        outp = trafo(**self.batch_dict)
        self.assertEqual(tuple(outp["data"].shape), (1, 3, 1, 3))


if __name__ == "__main__":
    unittest.main()
