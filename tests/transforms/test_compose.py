import unittest
import torch

from rising.transforms.spatial import Mirror
from rising.transforms.compose import Compose, DropoutCompose


class TestCompose(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = {"data": torch.rand(1, 1, 10, 10)}
        self.transforms = [
            Mirror(dims=(0,), prob=1.),
            Mirror(dims=(0,), prob=1.)
        ]

    def test_multiple_grad_context(self):
        aten = torch.rand(10, 10)
        with torch.enable_grad():
            with torch.no_grad():
                aten2 = aten + 2
        self.assertIsNone(aten._grad_fn)
        with self.assertRaises(RuntimeError):
            aten.mean().backward()

    def test_compose_single(self):
        single_compose = Compose(self.transforms[0])
        outp = single_compose(**self.batch)
        expected = Mirror(dims=(0,), prob=1.)(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

    def test_compose_multiple(self):
        compose = Compose(self.transforms)
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose(self):
        compose = DropoutCompose(self.transforms[0], dropout=0.0)
        outp = compose(**self.batch)
        expected = Mirror(dims=(0,), prob=1.)(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

        compose = DropoutCompose(self.transforms[0], dropout=1.0)
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose_error(self):
        with self.assertRaises(TypeError):
            compose = DropoutCompose(self.transforms, dropout=[1.0])


if __name__ == '__main__':
    unittest.main()
