import unittest
from copy import deepcopy

import torch

from rising.transforms.functional import box_to_seg, filter_keys, instance_to_semantic, pop_keys, seg_to_box


class TestSegBox(unittest.TestCase):
    def setUp(self) -> None:
        self.seg = torch.zeros(10, 10, 10, dtype=torch.long)
        self.seg[1:3, 1:3, 1:3] = 1
        self.seg[5:8, 5:8, 1:3] = 2
        self.boxes = [torch.tensor([1, 1, 2, 2, 1, 2]), torch.tensor([5, 5, 7, 7, 1, 2])]
        self.cls = [2, 1]

    def test_box_to_seg_error(self):
        self.boxes = [b[1:] for b in self.boxes]
        with self.assertRaises(TypeError):
            box_to_seg(self.boxes, self.seg.shape, self.seg.dtype, self.seg.device)

    def test_box_to_seg_2d(self):
        boxes_2d = [torch.tensor([2, 2, 2, 2])]
        expected = torch.zeros(10, 10, dtype=torch.long)
        expected[2, 2] = 1
        seg = box_to_seg(boxes_2d, expected.shape, expected.dtype, expected.device)
        self.assertTrue((expected == seg).all())

    def test_box_to_seg_3d(self):
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

    def test_pop_keys(self):
        data = {str(idx): idx for idx in range(10)}
        keys_to_pop_list = [str(idx) for idx in range(0, 10, 2)]

        def keys_to_pop_fn(key):
            return key in [str(idx) for idx in range(0, 10, 2)]

        for return_pop in [True, False]:
            for _pop_keys in [keys_to_pop_list, keys_to_pop_fn]:
                with self.subTest(return_pop=return_pop, pop_keys=_pop_keys):
                    if isinstance(_pop_keys, list):
                        __pop_keys = deepcopy(_pop_keys)
                    else:
                        __pop_keys = _pop_keys
                    result = pop_keys(data=deepcopy(data), keys=__pop_keys, return_popped=return_pop)

                    if return_pop:
                        result, popped = result
                        for k in popped.keys():
                            self.assertIn(k, keys_to_pop_list)

                        for k in result.keys():
                            self.assertNotIn(k, keys_to_pop_list)

    def test_filter_keys(self):
        data = {str(idx): idx for idx in range(10)}
        keys_to_filter_list = [str(idx) for idx in range(0, 10, 2)]

        def keys_to_filter_fn(key):
            return key in [str(idx) for idx in range(0, 10, 2)]

        for return_pop in [True, False]:
            for _filter_keys in [keys_to_filter_list, keys_to_filter_fn]:
                with self.subTest(return_pop=return_pop, filter_keys=_filter_keys):
                    if isinstance(_filter_keys, list):
                        __filter_keys = deepcopy(_filter_keys)
                    else:
                        __filter_keys = _filter_keys
                    result = filter_keys(data=deepcopy(data), keys=__filter_keys, return_popped=return_pop)

                    if return_pop:
                        result, popped = result
                        for k in popped.keys():
                            self.assertNotIn(k, keys_to_filter_list)

                        for k in result.keys():
                            self.assertIn(k, keys_to_filter_list)


if __name__ == "__main__":
    unittest.main()
