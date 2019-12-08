import unittest
import torch

from rising.transforms.functional import *


class TestSegBox(unittest.TestCase):
    def setUp(self) -> None:
        self.seg = torch.zeros(10, 10, 10, dtype=torch.long)
        self.seg[1:3, 1:3, 1:3] = 1
        self.seg[5:8, 5:8, 1:3] = 2
        self.boxes = [torch.tensor([1, 1, 2, 2, 1, 2]), torch.tensor([5, 5, 7, 7, 1, 2])]
        self.cls = [2, 1]

    def test_box_to_seg(self):
        seg = box_to_seg(self.boxes, self.seg.shape, self.seg.dtype, self.seg.device)
        self.assertTrue((self.seg == seg).all())

    def test_seg_to_box(self):
        boxes = seg_to_box(self.seg, 3)
        for box_gt, box_out in zip(self.boxes, boxes):
            self.assertTrue((box_gt == box_out).all())

    def test_instance_to_semantic(self):
        semantic = instance_to_semantic(self.seg, self.cls)
        expected = torch.zeros_like(self.seg)
        expected[1:3, 1:3, 1:3] = 2
        expected[5:8, 5:8, 1:3] = 1
        self.assertTrue((semantic == expected).all())


if __name__ == '__main__':
    unittest.main()
