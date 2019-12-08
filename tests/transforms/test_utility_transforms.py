import unittest
import torch
from rising.transforms.utility import *


class TestUtilityTransforms(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_do_nothing_transform(self):
        inp = {"data": 0, "seg": 1}
        trafo = DoNothing()
        outp = trafo(**inp)
        self.assertEqual(outp["data"], 0)
        self.assertEqual(outp["seg"], 1)


class TestSegBoxTransforms(unittest.TestCase):
    def setUp(self) -> None:
        self.seg = torch.zeros(10, 10, 10, dtype=torch.long)
        self.seg[1:3, 1:3, 1:3] = 1
        self.seg[5:8, 5:8, 1:3] = 2
        self.boxes = [torch.tensor([1, 1, 2, 2, 1, 2]), torch.tensor([5, 5, 7, 7, 1, 2])]
        self.cls = [2, 1]
        self.batch = {"seg": self.seg[None, None], "boxes": [self.boxes], "cls": [self.cls]}

    def test_seg_to_box_transform(self):
        trafo = SegToBox(keys={"seg": "box"})
        boxes = trafo(**self.batch)["box"][0]
        for box_gt, box_out in zip(self.boxes, boxes):
            self.assertTrue((box_gt == box_out).all())

    def test_box_to_seg_transform(self):
        trafo = BoxToSeg(keys={"boxes": "seg"}, shape=self.seg.shape, dtype=torch.long, device="cpu")
        seg = trafo(**self.batch)["seg"][0, 0]
        self.assertTrue((self.seg == seg).all())

    def test_instance_to_semantic_transform(self):
        trafo = InstanceToSemantic(keys={"seg": "semantic"}, cls_key="cls")
        semantic = trafo(**self.batch)["semantic"][0, 0]
        expected = torch.zeros_like(self.seg)
        expected[1:3, 1:3, 1:3] = 2
        expected[5:8, 5:8, 1:3] = 1
        self.assertTrue((semantic == expected).all())


if __name__ == '__main__':
    unittest.main()
