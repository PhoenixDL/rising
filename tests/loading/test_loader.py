import unittest
from typing import Mapping, Sequence
from unittest.mock import Mock, patch

import numpy as np
import torch

from rising.loading.dataset import Dataset
from rising.loading.loader import (
    BatchTransformer,
    DataLoader,
    SampleTransformer,
    _MultiProcessingDataLoaderIter,
    _seed_npy_before_worker_init,
    _SingleProcessDataLoaderIter,
    default_transform_call,
)
from rising.transforms import Mirror


class TestLoader(unittest.TestCase):
    def test_seed_npy_before_worker_init(self):
        expected_return = 100
        np.random.seed(1)
        expected = np.random.rand(1)
        worker_init = Mock(return_value=expected_return)

        output_return = _seed_npy_before_worker_init(worker_id=1, seed=0, worker_init_fn=worker_init)
        output = np.random.rand(1)
        self.assertEqual(output, expected)
        self.assertEqual(output_return, expected_return)
        worker_init.assert_called_once_with(1)

    def test_seed_npy_before_worker_init_import_error(self):
        with patch.dict("sys.modules", {"numpy": None}):
            expected_return = 100
            worker_init = Mock(return_value=expected_return)
            output_return = _seed_npy_before_worker_init(worker_id=1, seed=0, worker_init_fn=worker_init)
            self.assertEqual(output_return, expected_return)
            worker_init.assert_called_once_with(1)

    def test_dataloader_np_import_error(self):
        with patch.dict("sys.modules", {"numpy": None}):
            loader = DataLoader([0, 1, 2], num_workers=2)
            iterator = iter(loader)
            self.assertIsInstance(iterator, _MultiProcessingDataLoaderIter)

    def test_dataloader_single_process(self):
        loader = DataLoader([0, 1, 2])
        iterator = iter(loader)
        self.assertIsInstance(iterator, _SingleProcessDataLoaderIter)

    def test_dataloader_multi_process(self):
        loader = DataLoader([0, 1, 2], num_workers=2)
        iterator = iter(loader)
        self.assertIsInstance(iterator, _MultiProcessingDataLoaderIter)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda gpu available")
    def test_dataloader_gpu_transforms(self):
        device = torch.cuda.current_device()
        self.check_output_device(device=device)

    @patch("torch.cuda.is_available", side_effect=lambda: False)
    def test_dataloader_gpu_transforms_no_cuda(self, fn):
        self.check_output_device(device="cpu")

    def check_output_device(self, device):
        data = torch.rand(1, 3, 3)
        dset = [{"data": data}] * 5
        gpu_transforms = Mirror((0,))
        loader = DataLoader(dset, num_workers=0, gpu_transforms=gpu_transforms)
        iterator = iter(loader)
        outp = next(iterator)
        expected = data[None].flip([2]).to(device=device)
        self.assertTrue(outp["data"].allclose(expected))


class BatchTransformerTest(unittest.TestCase):
    def check_batch_transformer(self, collate_output):
        collate = Mock(return_value=collate_output)
        transforms = Mock(return_value=2)
        transformer = BatchTransformer(collate_fn=collate, transforms=transforms, auto_convert=False)

        output = transformer(0)

        collate.assert_called_once_with(0)
        if isinstance(collate_output, Sequence):
            transforms.assert_called_once_with(*collate_output)
        elif isinstance(collate_output, Mapping):
            transforms.assert_called_once_with(**collate_output)
        else:
            transforms.assert_called_once_with(collate_output)
        self.assertEqual(2, output)

    def test_batch_transformer(self):
        self.check_batch_transformer(0)

    def test_batch_transformer_sequence(self):
        self.check_batch_transformer((0, 1))

    def test_batch_transformer_mapping(self):
        self.check_batch_transformer({"a": 0})

    def test_batch_transformer_auto_convert(self):
        collate = Mock(return_value=0)
        transforms = Mock(return_value=np.array([0, 1]))
        transformer = BatchTransformer(collate_fn=collate, transforms=transforms, auto_convert=True)
        output = transformer(0)
        self.assertTrue((output == torch.tensor([0, 1])).all())


class SampleTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        class DummyDset(Dataset):
            def __init__(self, num_samples=10):
                super().__init__()

                self.samples = [{"data": np.random.rand(1, 28, 28)} for idx in range(num_samples)]

            def __getitem__(self, item):
                return self.samples[item]

            def __len__(self):
                return len(self.samples)

        self.dset = DummyDset(10)

    def test_no_trafo(self):
        transformer = SampleTransformer(self.dset, None)
        for i in range(len(self.dset)):
            with self.subTest(idx=i):
                self.assertTrue(np.allclose(transformer[0]["data"], self.dset[0]["data"]))

    def test_trafo_no_pseudo_batchdim(self):
        def trafo(**data):
            for k, v in data.items():
                self.assertTupleEqual(v.shape, (1, 28, 28))

            return data

        transformer = SampleTransformer(self.dset, trafo)
        for i in range(len(self.dset)):
            with self.subTest(idx=i):
                transformed = transformer[i]["data"]
                orig = self.dset[i]["data"]

                self.assertTrue(np.allclose(transformed, orig))

    def test_trafo_pseudo_batchdim(self):
        def trafo(**data):
            for k, v in data.items():
                self.assertTupleEqual(v.shape, (1, 1, 28, 28))

            return data

        transformer = SampleTransformer(self.dset, trafo, pseudo_batch_dim=True)
        for i in range(len(self.dset)):
            with self.subTest(idx=i):
                transformed = transformer[i]["data"]
                orig = self.dset[i]["data"]

                self.assertTrue(np.allclose(transformed, orig))

    def test_pseudo_batching_tensor(self):
        transformer = SampleTransformer(self.dset)

        input_tensor = torch.rand(2, 3, 4)
        batched = transformer._change_pseudo_batch_dim(input_tensor, add=True)
        self.assertTupleEqual(batched.shape, (1, 2, 3, 4))

        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)
        self.assertTupleEqual(unbatched.shape, input_tensor.shape)
        self.assertTrue(torch.allclose(unbatched, input_tensor))

    def test_pseudo_batching_array(self):
        transformer = SampleTransformer(self.dset)

        input_array = np.random.rand(2, 3, 4)
        batched = transformer._change_pseudo_batch_dim(input_array, add=True)
        self.assertTupleEqual(batched.shape, (1, 2, 3, 4))

        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)
        self.assertTupleEqual(unbatched.shape, input_array.shape)
        self.assertTrue(np.allclose(unbatched, input_array))

    def test_pseudo_batching_str(self):
        transformer = SampleTransformer(self.dset)
        input_str = "abc"

        for add in [True, False]:
            with self.subTest(add=add):
                self.assertEqual(input_str, transformer._change_pseudo_batch_dim(input_str, add=add))

    def test_pseudo_batching_int(self):
        transformer = SampleTransformer(self.dset)
        input_int = 42

        for add in [True, False]:
            with self.subTest(add=add):
                self.assertEqual(input_int, transformer._change_pseudo_batch_dim(input_int, add=add))

    def test_pseudo_batching_float(self):
        transformer = SampleTransformer(self.dset)
        input_float = 42.0

        for add in [True, False]:
            with self.subTest(add=add):
                self.assertEqual(input_float, transformer._change_pseudo_batch_dim(input_float, add=add))

    def test_pseudo_batching_mapping(self):
        transformer = SampleTransformer(self.dset)
        mapping = {"a": torch.rand(2, 3, 4), "b": torch.rand(3, 4, 5)}
        batched = transformer._change_pseudo_batch_dim(mapping, add=True)

        self.assertIsInstance(batched, type(mapping))
        self.assertEqual(len(mapping), len(batched))
        for key in batched.keys():
            with self.subTest(key=key):
                self.assertIn(key, mapping)

        for k, v in mapping.items():
            with self.subTest(k=k, v=v):
                self.assertEqual(v.ndim + 1, batched[k].ndim)

        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)
        self.assertIsInstance(unbatched, type(mapping))
        self.assertEqual(len(mapping), len(unbatched))
        for key in unbatched.keys():
            with self.subTest(key=key):
                self.assertIn(key, mapping)

        for k, v in mapping.items():
            with self.subTest(k=k, v=v):
                self.assertEqual(v.ndim, unbatched[k].ndim)

    def test_pseudo_batch_dim_named_tuple(self):
        from collections import namedtuple

        Foo = namedtuple("Foo", "bar")
        transformer = SampleTransformer(self.dset)

        foo = Foo(torch.tensor([2, 3, 4]))

        batched = transformer._change_pseudo_batch_dim(foo, add=True)
        self.assertIsInstance(batched, Foo)
        self.assertTupleEqual(batched.bar.shape, tuple([1] + list(foo.bar.shape)))

        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)
        self.assertIsInstance(unbatched, Foo)
        self.assertTrue(torch.allclose(unbatched.bar, foo.bar))

    def test_pseudo_batch_dim_sequence(self):
        transformer = SampleTransformer(self.dset)

        input_sequence = [torch.tensor([2, 3, 4]), torch.tensor([3, 4, 5])]
        batched = transformer._change_pseudo_batch_dim(input_sequence, add=True)
        self.assertEqual(len(batched), len(input_sequence))

        for idx in range(len(input_sequence)):
            with self.subTest(idx=idx):
                self.assertEqual(input_sequence[idx].ndim + 1, batched[idx].ndim)

        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)
        self.assertEqual(len(unbatched), len(input_sequence))

        for idx in range(len(input_sequence)):
            with self.subTest(idx=idx):
                self.assertTrue(torch.allclose(unbatched[idx], input_sequence[idx]))

    def test_pseudo_batch_dim_custom_obj(self):
        class Foo(object):
            self.bar = 5.0

        transformer = SampleTransformer(self.dset)
        foo = Foo()
        batched = transformer._change_pseudo_batch_dim(foo, add=True)
        unbatched = transformer._change_pseudo_batch_dim(batched, add=False)

        self.assertEqual(foo, batched)
        self.assertEqual(foo, unbatched)
        self.assertEqual(batched, unbatched)


class DefaultTransformCallTest(unittest.TestCase):
    def check_transform_call(self, inp, outp=0, fn=default_transform_call) -> unittest.mock.Mock:
        trafo = Mock(return_value=outp)
        val = fn(inp, trafo)
        self.assertEqual(outp, val)
        return trafo

    def test_default_transform_call(self):
        inp = "a"
        mock = self.check_transform_call(inp)
        mock.assert_called_once_with(inp)

    def test_default_transform_call_seq(self):
        inp = ("a", "b")
        mock = self.check_transform_call(inp)
        mock.assert_called_once_with(*inp)

    def test_default_transform_call_map(self):
        inp = {"a": 1}
        mock = self.check_transform_call(inp)
        mock.assert_called_once_with(**inp)


if __name__ == "__main__":
    unittest.main()
