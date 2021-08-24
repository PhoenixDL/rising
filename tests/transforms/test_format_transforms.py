import unittest

import torch

from rising.transforms.format import MapToSeq, RenameKeys, SeqToMap


class TestFormat(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_map_to_seq(self):
        trafo = MapToSeq(("data", "seg", "label"))
        out = trafo(**{"data": 0, "seg": 1, "label": 2})
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 1)
        self.assertEqual(out[2], 2)

    def test_seq_to_map(self):
        trafo = SeqToMap(("data", "seg", "label"))
        out = trafo(0, 1, 2)
        self.assertEqual(out["data"], 0)
        self.assertEqual(out["seg"], 1)
        self.assertEqual(out["label"], 2)

    def test_rename(self):
        trafo = RenameKeys({"data": "new_data", "seg": "new_seg"})
        out = trafo(**{"data": 0, "seg": 1, "label": 2})
        self.assertIn("new_data", out)
        self.assertIn("new_seg", out)


if __name__ == "__main__":
    unittest.main()
