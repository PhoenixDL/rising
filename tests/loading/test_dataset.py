import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import dill
import numpy as np
from torch.multiprocessing import Pool

from rising.loading.dataset import AsyncDataset, dill_helper, load_async


class LoadDummySample:
    def __call__(self, path, *args, **kwargs):
        data = {"data": np.random.rand(1, 256, 256), "label": np.random.randint(2), "id": f"sample{path}"}
        return data


def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def pickle_load(path, *args, **kwargs):
    with open(path, "rb") as f:
        pickle.load(f)


class TestBaseDatasetDir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(dir=os.path.dirname(os.path.realpath(__file__)))
        loader = LoadDummySample()
        for i in range(10):
            pickle_save(os.path.join(self.dir, f"sample{i}.pkl"), loader(i))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir)

    def test_async_dataset_dir(self):
        dataset = AsyncDataset(self.dir, pickle_load, label_load_fct=None)
        self.assertEqual(len(dataset), 10)
        for i in dataset:
            pass


class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.paths = list(range(10))

    def test_async_dataset(self):
        dataset = AsyncDataset(self.paths, LoadDummySample(), verbose=True, label_load_fct=None)
        self.assertEqual(len(dataset), 10)
        self.check_dataset_access(dataset, [0, 5, 9])
        self.check_dataset_outside_access(dataset, [10, 20])
        self.check_dataset_iter(dataset)

    def test_async_verbose_multiprocessing(self):
        # TODO: add tqdm mock to f progress bar is invoked correctly (do this when dataset tests are reworked)
        dataset = AsyncDataset(self.paths, LoadDummySample(), num_workers=4, verbose=True, label_load_fct=None)
        self.assertEqual(len(dataset), 10)
        self.check_dataset_access(dataset, [0, 5, 9])
        self.check_dataset_outside_access(dataset, [10, 20])
        self.check_dataset_iter(dataset)

    def test_async_dataset_extend(self):
        def load_mul_sample(path) -> list:
            return [LoadDummySample()(path, None)] * 4

        dataset = AsyncDataset(self.paths, load_mul_sample, num_workers=4, verbose=False, mode="extend")
        self.assertEqual(len(dataset), 40)
        self.check_dataset_access(dataset, [0, 20, 39])
        self.check_dataset_outside_access(dataset, [40, 45])
        self.check_dataset_iter(dataset)

    def test_async_dataset_mode_error(self):
        with self.assertRaises(TypeError):
            dataset = AsyncDataset(self.paths, LoadDummySample(), label_load_fct=None, mode="no_mode:P")

    def check_dataset_access(self, dataset, inside_idx):
        try:
            for _i in inside_idx:
                a = dataset[_i]
        except BaseException:
            self.assertTrue(False)

    def check_dataset_outside_access(self, dataset, outside_idx):
        for _i in outside_idx:
            with self.assertRaises(IndexError):
                a = dataset[_i]

    def check_dataset_iter(self, dataset):
        try:
            j = 0
            for i in dataset:
                self.assertIn("data", i)
                self.assertIn("label", i)
                j += 1
            assert j == len(dataset)
        except BaseException:
            raise AssertionError("Dataset iteration failed.")

    def test_subset_dataset(self):
        idx = [0, 1, 2, 5, 6]
        dataset = AsyncDataset(self.paths, LoadDummySample(), label_load_fct=None)
        subset = dataset.get_subset(idx)
        self.assertEqual(len(subset), len(idx))
        for _i, _idx in enumerate(idx):
            self.assertEqual(subset[_i]["id"], dataset[_idx]["id"])
        with self.assertRaises(IndexError):
            subset[len(idx)]


class TestHelperFunctions(unittest.TestCase):
    def test_load_async(self):
        callback = Mock()

        with Pool(processes=1) as p:
            ref = load_async(p, lambda x: x, 0, callback=callback)
            self.assertEqual(ref.get(), 0)

        callback.assert_called_once()

    def test_dill_helper(self):
        payload = dill.dumps((lambda x: x, (1,), {}))
        res = dill_helper(payload)
        self.assertEqual(res, 1)


if __name__ == "__main__":
    unittest.main()
