import unittest
from collections import namedtuple

import torch

try:
    import numpy as np
except ImportError:
    np = None

from rising.loading.collate import numpy_collate


class TestCollate(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_int(self):
        arr = [1, 2, -1]
        collated = numpy_collate(arr)
        expected = np.array(arr)
        self.assertTrue((collated == expected).all())
        self.assertEqual(collated.dtype, expected.dtype)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_float(self):
        arr = [1.1, 2.3, -0.9]
        collated = numpy_collate(arr)
        expected = np.array(arr)
        self.assertTrue((collated == expected).all())
        self.assertEqual(collated.dtype, expected.dtype)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_bool(self):
        arr = [True, False]
        collated = numpy_collate(arr)
        self.assertTrue(all(collated == np.array(arr)))
        self.assertEqual(collated.dtype, np.bool)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_str(self):
        # Should be a no-op
        arr = ["a", "b", "c"]
        self.assertTrue((arr == numpy_collate(arr)))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_ndarray(self):
        arr = [np.array(0), np.array(1), np.array(2)]
        collated = numpy_collate(arr)
        expected = np.array([0, 1, 2])
        self.assertTrue((collated == expected).all())

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_tensor(self):
        arr = [torch.tensor(0), torch.tensor(1), torch.tensor(2)]
        collated = numpy_collate(arr)
        expected = np.array([0, 1, 2])
        self.assertTrue((collated == expected).all())

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_mapping(self):
        arr = [{"a": np.array(0), "b": np.array(1)}] * 2
        collated = numpy_collate(arr)
        expected = {"a": np.array([0, 0]), "b": np.array([1, 1])}
        for key in expected.keys():
            self.assertTrue((collated[key] == expected[key]).all())
        self.assertEqual(len(expected.keys()), len(collated.keys()))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_sequence(self):
        arr = [[np.array(0), np.array(1)], [np.array(0), np.array(1)]]
        collated = numpy_collate(arr)
        expected = [np.array([0, 0]), np.array([1, 1])]
        for i in range(len(collated)):
            self.assertTrue((collated[i] == expected[i]).all())
        self.assertEqual(len(expected), len(collated))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_error(self):
        with self.assertRaises(TypeError):
            collated = numpy_collate([{"a", "b"}, {"a", "b"}])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_numpy_collate_named_tuple(self):
        Point = namedtuple("Point", ["x", "y"])
        arr = [Point(0, 1), Point(2, 3)]
        collated = numpy_collate(arr)
        expected = Point(np.array([0, 2]), np.array([1, 3]))
        self.assertTrue((collated.x == expected.x).all())
        self.assertTrue((collated.y == expected.y).all())


if __name__ == "__main__":
    unittest.main()
