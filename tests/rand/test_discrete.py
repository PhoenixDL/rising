import unittest
from rising.random import DiscreteParameter


class TestDiscrete(unittest.TestCase):
    def test_discrete_error(self):
        with self.assertRaises(ValueError):
            param = DiscreteParameter((1., 2.), replacement=False, weights=(0.3, 0.7))


if __name__ == '__main__':
    unittest.main()
