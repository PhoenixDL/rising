import unittest
import torch
from typing import Sequence, Mapping
import numpy as np
from unittest.mock import Mock, patch
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter

from rising.loading.loader import _seed_npy_before_worker_init, DataLoader, \
    BatchTransformer, _MultiProcessingDataLoaderIter, default_transform_call
from rising.transforms import Mirror


class TestLoader(unittest.TestCase):
    def test_seed_npy_before_worker_init(self):
        expected_return = 100
        np.random.seed(1)
        expected = np.random.rand(1)
        worker_init = Mock(return_value=expected_return)

        output_return = _seed_npy_before_worker_init(worker_id=1, seed=0,
                                                     worker_init_fn=worker_init)
        output = np.random.rand(1)
        self.assertEqual(output, expected)
        self.assertEqual(output_return, expected_return)
        worker_init.assert_called_once_with(1)

    def test_seed_npy_before_worker_init_import_error(self):
        with patch.dict('sys.modules', {'numpy': None}):
            expected_return = 100
            worker_init = Mock(return_value=expected_return)
            output_return = _seed_npy_before_worker_init(worker_id=1, seed=0,
                                                         worker_init_fn=worker_init)
            self.assertEqual(output_return, expected_return)
            worker_init.assert_called_once_with(1)

    def test_dataloader_np_import_error(self):
        with patch.dict('sys.modules', {'numpy': None}):
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

    @patch('torch.cuda.is_available', side_effect=lambda: False)
    def test_dataloader_gpu_transforms_no_cuda(self, fn):
        self.check_output_device(device='cpu')

    def check_output_device(self, device):
        data = torch.rand(1, 3, 3)
        dset = [{"data": data}] * 5
        gpu_transforms = Mirror((0, ), prob=1)
        loader = DataLoader(dset, num_workers=0, gpu_transforms=gpu_transforms)
        iterator = iter(loader)
        outp = next(iterator)
        expected = data[None].flip([2]).to(device=device)
        self.assertTrue(outp["data"].allclose(expected))


class BatchTransformerTest(unittest.TestCase):
    def check_batch_transformer(self, collate_output):
        collate = Mock(return_value=collate_output)
        transforms = Mock(return_value=2)
        transformer = BatchTransformer(collate_fn=collate, transforms=transforms,
                                       auto_convert=False)

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
        transformer = BatchTransformer(collate_fn=collate, transforms=transforms,
                                       auto_convert=True)
        output = transformer(0)
        self.assertTrue((output == torch.tensor([0, 1])).all())


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


if __name__ == '__main__':
    unittest.main()
