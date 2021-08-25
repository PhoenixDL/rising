import unittest

from rising import AbstractMixin


class Abstract(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.abstract = True


class AbstractForward(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.abstract = True


class PreMix(AbstractMixin, Abstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PostMix(AbstractForward, AbstractMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MyTestCase(unittest.TestCase):
    def test_pre_mix(self):
        obj = PreMix(a=True)
        self.assertFalse(hasattr(obj, "a"))
        self.assertTrue(obj.abstract)

    def test_post_mix(self):
        obj = PostMix(a=True)
        self.assertTrue(obj.a)
        self.assertTrue(obj.abstract)


if __name__ == "__main__":
    unittest.main()
