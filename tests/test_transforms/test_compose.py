import unittest
import torch

from rising.transforms.spatial import MirrorTransform
from rising.transforms.compose import Compose, DropoutCompose


class TestCompose(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = {"data": torch.rand(1, 1, 10, 10)}
        self.transforms = [
            MirrorTransform(dims=(0,), prob=1.),
            MirrorTransform(dims=(0,), prob=1.)
        ]

    def test_compose_single(self):
        single_compose = Compose(self.transforms[0])
        outp = single_compose(**self.batch)
        expected = MirrorTransform(dims=(0,), prob=1.)(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

    def test_compose_multiple(self):
        compose = Compose(self.transforms)
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose(self):
        compose = DropoutCompose(self.transforms[0], dropout=0.0)
        outp = compose(**self.batch)
        expected = MirrorTransform(dims=(0,), prob=1.)(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

        compose = DropoutCompose(self.transforms[0], dropout=1.0)
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose_error(self):
        with self.assertRaises(TypeError):
            compose = DropoutCompose(self.transforms, dropout=[1.0])


if __name__ == '__main__':
    unittest.main()
