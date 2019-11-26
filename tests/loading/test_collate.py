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
    def test_default_collate_dtype(self):
        arr = [1, 2, -1]
        collated = numpy_collate(arr)
        self.assertEqual(collated, np.array(arr))
        self.assertEqual(collated.dtype, np.int32)

        arr = [1.1, 2.3, -0.9]
        collated = numpy_collate(arr)
        self.assertEqual(collated, np.array(arr))
        self.assertEqual(collated.dtype, np.float32)

        arr = [True, False]
        collated = numpy_collate(arr)
        self.assertEqual(collated, np.array(arr))
        self.assertEqual(collated.dtype, np.bool)

        # Should be a no-op
        arr = ['a', 'b', 'c']
        self.assertEqual(arr, numpy_collate(arr))


if __name__ == '__main__':
    unittest.main()