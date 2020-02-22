import unittest
from rising.transforms.function import LambdaTransform, LambdaKeyTransform


class TestLambdaTransforms(unittest.TestCase):
    def test_lambda_trafo(self):
        def test_fn(data):
            data['abc'] = 5
            return data

        trafo = LambdaTransform(trafo_fn=test_fn)
        result = trafo(**{})
        self.assertDictEqual({'abc': 5}, result)

        trafo = LambdaTransform(trafo_fn=lambda x: {**x, 'abc': 5})
        result = trafo(**{'cde': 6})
        self.assertDictEqual({'cde': 6, 'abc': 5}, result)

    def test_lambda_key_trafo(self):
        def test_fn(key, value):
            if value is not None:
                return value + 5
            else:
                return None

        trafo = LambdaKeyTransform(test_fn, keys=['abc', 'def'])
        result = trafo(**{'abc': 4, 'hjk': 4})
        self.assertDictEqual(result, {'abc': 9, 'def': None, 'hjk': 4})

        trafo = LambdaKeyTransform(lambda x,y: y + 5 if y is not None else None,
                                   keys=['abc', 'def'])
        result = trafo(**{'abc': 4, 'hjk': 4})
        self.assertDictEqual(result, {'abc': 9, 'def': None, 'hjk': 4})


if __name__ == '__main__':
    unittest.main()
