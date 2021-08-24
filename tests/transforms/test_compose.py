import copy
import random
import unittest

import torch

from rising.transforms import AbstractTransform
from rising.transforms.compose import Compose, DropoutCompose, OneOf, _TransformWrapper
from rising.transforms.spatial import Mirror


class TestCompose(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = {"data": torch.rand(1, 1, 10, 10)}
        self.transforms = [Mirror(dims=(0,)), Mirror(dims=(0,))]

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
        expected = Mirror(dims=(0,))(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

    def test_compose_multiple(self):
        compose = Compose(self.transforms)
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())
        self.assertEqual(len(compose.transform_order), 2)

    def test_compose_shuffle(self):
        compose = Compose([Mirror(dims=(0,))] * 10, shuffle=True)

        random.seed(0)
        outp = compose(**self.batch)

        order = list(range(len(compose.transforms)))
        expected_order = copy.deepcopy(order)
        random.seed(0)
        random.shuffle(expected_order)

        self.assertEqual(compose.transform_order, expected_order)
        self.assertNotEqual(expected_order, order)

    def test_compose_multiple_tuple(self):
        compose = Compose(tuple(self.transforms))
        outp = compose(**self.batch)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose(self):
        compose = DropoutCompose(self.transforms[0], dropout=0.0)
        self.assertEqual(len(compose.transform_order), 1)
        outp = compose(**self.batch)
        expected = Mirror(dims=(0,))(**self.batch)
        self.assertTrue((expected["data"] == outp["data"]).all())

        compose = DropoutCompose(self.transforms, dropout=1.0)
        outp = compose(**self.batch)
        self.assertEqual(len(compose.transform_order), 2)
        self.assertTrue((self.batch["data"] == outp["data"]).all())

    def test_dropout_compose_error(self):
        with self.assertRaises(TypeError):
            compose = DropoutCompose(self.transforms, dropout=[1.0])

    def test_device_dtype_change(self):
        class DummyTrafo(AbstractTransform):
            def __init__(self, a):
                super().__init__(False)
                self.register_buffer("tmp", a)

            def __call__(self, *args, **kwargs):
                return self.tmp

        trafo_a = DummyTrafo(torch.tensor([1.0], dtype=torch.float32))
        trafo_a = trafo_a.to(torch.float32)
        trafo_b = DummyTrafo(torch.tensor([2.0], dtype=torch.float32))
        trafo_b = trafo_b.to(torch.float32)
        self.assertEqual(trafo_a.tmp.dtype, torch.float32)
        self.assertEqual(trafo_b.tmp.dtype, torch.float32)
        compose = Compose(trafo_a, trafo_b)
        compose = compose.to(torch.float64)

        self.assertEqual(compose.transforms[0].tmp.dtype, torch.float64)

    def test_wrapping_non_module_trafos(self):
        class DummyTrafo:
            def __init__(self):
                self.a = 5

            def __call__(self, *args, **kwargs):
                return 5

        dummy_trafo = DummyTrafo()

        compose = Compose([dummy_trafo])
        self.assertIsInstance(compose.transforms[0], _TransformWrapper)
        self.assertIsInstance(compose.transforms[0].trafo, DummyTrafo)

    def test_one_of_no_weight(self):
        trafo = Mirror((0,))
        comp = OneOf(trafo, trafo)
        out = comp(**self.batch)
        expected_out = trafo(**self.batch)
        self.assertTrue(out["data"].allclose(expected_out["data"]))

    def test_one_of_weight_error(self):
        with self.assertRaises(ValueError):
            trafo = Mirror((0,))
            comp = OneOf(trafo, trafo, weights=[0.33, 0.33, 0.33])

    def test_one_of_weight(self):
        for weight in [[1.0, 0.0], torch.tensor([1.0, 0.0])]:
            with self.subTest(weight=weight):
                trafo0 = Mirror((0,))
                trafo1 = Mirror((1,))
                comp = OneOf(trafo0, trafo1, weights=weight)
                out = comp(**self.batch)
                expected_out = trafo0(**self.batch)
                self.assertTrue(out["data"].allclose(expected_out["data"]))

    def test_no_trafo_error(self):
        for trafo_cls in [OneOf, Compose, DropoutCompose]:
            with self.subTest(trafo_cls=trafo_cls):
                with self.assertRaises(ValueError):
                    comp = trafo_cls([])
                with self.assertRaises(ValueError):
                    comp = trafo_cls()


if __name__ == "__main__":
    unittest.main()
