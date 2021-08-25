import unittest

import torch

from rising.utils.checktype import check_scalar


class TypeCheckTestCase(unittest.TestCase):
    def test_scalar_check(self):
        expectations = [True, True, False, False, False, True, False]
        inputs = [0.0, 1, None, "123", [1, 1], torch.tensor(1), torch.tensor([1, 2])]

        for inp, exp in zip(inputs, expectations):
            with self.subTest(input=inp, expectation=exp):
                self.assertEqual(check_scalar(inp), exp)


if __name__ == "__main__":
    unittest.main()
