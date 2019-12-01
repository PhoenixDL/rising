import torch
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from rising.loading.collate import numpy_collate


# TODO: Add more collate test cases
class TestCollate(unittest.TestCase):
    @unittest.skipIf(np is None, 'numpy is not available')
    def test_default_collate_int(self):
        arr = [1, 2, -1]
        collated = numpy_collate(arr)
        expected = np.array(arr)
        self.assertTrue((collated == expected).all())
        self.assertEqual(collated.dtype, expected.dtype)

    @unittest.skipIf(np is None, 'numpy is not available')
    def test_default_collate_float(self):
        arr = [1.1, 2.3, -0.9]
        collated = numpy_collate(arr)
        expected = np.array(arr)
        self.assertTrue((collated == expected).all())
        self.assertEqual(collated.dtype, expected.dtype)

    @unittest.skipIf(np is None, 'numpy is not available')
    def test_default_collate_bool(self):
        arr = [True, False]
        collated = numpy_collate(arr)
        self.assertTrue(all(collated == np.array(arr)))
        self.assertEqual(collated.dtype, np.bool)

    @unittest.skipIf(np is None, 'numpy is not available')
    def test_default_collate_str(self):
        # Should be a no-op
        arr = ['a', 'b', 'c']
        self.assertTrue((arr == numpy_collate(arr)))


if __name__ == '__main__':
    unittest.main()