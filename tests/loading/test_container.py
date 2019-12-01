import unittest
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rising.loading.container import DataContainer, DataContainerID
from tests import DummyDataset, DummyDatasetID


class LoadDummySampleID:
    def __init__(self, keys=('data', 'label'), sizes=((3, 128, 128), (3,)),
                 **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.sizes = sizes

    def __call__(self, path, *args, **kwargs):
        data = {_k: np.random.rand(*_s)
                for _k, _s in zip(self.keys, self.sizes)}
        data['id'] = int(path)
        return data


class TestDataContainer(unittest.TestCase):
    def setUp(self):
        self.dset = DummyDataset(num_samples=6,
                                 load_fn=(LoadDummySampleID()),
                                 )
        self.dset_id = DummyDatasetID(num_samples=6,
                                      load_fn=LoadDummySampleID(),
                                      )
        self.split = {"train": [0, 1, 2], "val": [3, 4, 5]}
        self.kfold = [{"train": [0, 1, 2], "val": [3, 4, 5]},
                      {"val": [0, 1, 2], "train": [3, 4, 5]}]

    def test_empty_container(self):
        container = DataContainer(self.dset)
        with self.assertRaises(AttributeError):
            container.dset["train"]

        with self.assertRaises(AttributeError):
            container.fold

    def test_split_by_index(self):
        container = DataContainer(self.dset)
        container.split_by_index(self.split)
        self.check_split(container)

    def check_split(self, container):
        self._assert_split(container, [0, 1, 2], [3, 4, 5])

    def test_kfold_by_index(self):
        container = DataContainer(self.dset)
        self.check_kfold(container.kfold_by_index(self.kfold))

    def check_kfold(self, container_generator):
        for container_fold in container_generator:
            if container_fold.fold == 0:
                self._assert_split(container_fold, [0, 1, 2], [3, 4, 5])
            elif container_fold.fold == 1:
                self._assert_split(container_fold, [3, 4, 5], [0, 1, 2])
            else:
                self.assertTrue(False, "Unknown Fold")

    def test_split_by_csv(self):
        container = DataContainer(self.dset)
        p = os.path.join(os.path.dirname(__file__), '_src', 'split.csv')
        container.split_by_csv(p, 'index', sep=';')
        self.check_split_csv(container)

    def check_split_csv(self, container):
        self._assert_split(container, [0, 1], [2, 3], [4, 5])

    def test_kfold_by_csv(self):
        container = DataContainer(self.dset)
        p = os.path.join(os.path.dirname(__file__), '_src', 'kfold.csv')
        self.check_kfold_csv(container.kfold_by_csv(p, 'index', sep=';'))

    def check_kfold_csv(self, container_generator):
        for container_fold in container_generator:
            if container_fold.fold == 0:
                self._assert_split(container_fold, [0, 1], [2, 3], [4, 5])
            elif container_fold.fold == 1:
                self._assert_split(container_fold, [2, 3], [4, 5], [0, 1])
            elif container_fold.fold == 2:
                self._assert_split(container_fold, [4, 5], [0, 1], [2, 3])
            else:
                self.assertTrue(False, "Unknown Fold")

    def _assert_split(self, container, train, val=None, test=None):
        self.assertEqual([d["id"] for d in container.dset["train"]], train)
        if val is not None:
            self.assertEqual([d["id"] for d in container.dset["val"]], val)
        if test is not None:
            self.assertEqual([d["id"] for d in container.dset["test"]], test)

    def test_split_by_index_id(self):
        container = DataContainerID(self.dset_id)
        container.split_by_id(self.split)
        self.check_split(container)

    def test_kfold_by_index_id(self):
        container = DataContainerID(self.dset_id)
        self.check_kfold(container.kfold_by_id(self.kfold))

    def test_split_by_csv_id(self):
        container = DataContainerID(self.dset_id)
        p = os.path.join(os.path.dirname(__file__), '_src', 'split.csv')
        container.split_by_csv_id(p, 'index', sep=';')
        self.check_split_csv(container)

    def test_kfold_by_csv_id(self):
        container = DataContainerID(self.dset_id)
        p = os.path.join(os.path.dirname(__file__), '_src', 'kfold.csv')
        self.check_kfold_csv(container.kfold_by_csv_id(p, 'index', sep=';'))

    def test_save_split_to_csv_id(self):
        container = DataContainerID(self.dset_id)
        container.split_by_id(self.split)
        p = os.path.join(os.path.dirname(__file__),
                         "_src", "test_generated_split.csv")

        container.save_split_to_csv_id(p, "id")

        df = pd.read_csv(p)
        self.assertTrue(df["id"].to_list(), [1, 2, 3, 4, 5, 6])
        self.assertTrue(df["split"].to_list(),
                        ["train", "train", "train", "val", "val", "val"])
        os.remove(p)


if __name__ == '__main__':
    unittest.main()