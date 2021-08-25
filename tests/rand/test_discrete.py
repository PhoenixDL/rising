import unittest

from rising.random import DiscreteCombinationsParameter, DiscreteParameter
from rising.random.discrete import combinations_all


class TestDiscrete(unittest.TestCase):
    def test_discrete_error(self):
        with self.assertRaises(ValueError):
            param = DiscreteParameter((1.0, 2.0), replacement=False, weights=(0.3, 0.7))

    def test_discrete_parameter(self):
        param = DiscreteParameter((1,))
        sample = param()
        self.assertEqual(sample, 1)

    def test_discrete_combinations_parameter(self):
        param = DiscreteCombinationsParameter((1,))
        sample = param()
        self.assertEqual(sample, 1)

    def test_combination_all(self):
        combs = combinations_all((0, 1))
        self.assertIn((0,), combs)
        self.assertIn((1,), combs)
        self.assertIn((0, 1), combs)
        self.assertEqual(len(combs), 3)


if __name__ == "__main__":
    unittest.main()
