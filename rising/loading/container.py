from __future__ import annotations

import pandas as pd
import typing
import pathlib
from collections import defaultdict

from rising.loading.dataset import Dataset
from rising.loading.splitter import SplitType


# TODO: Add docstrings for Datacontainer
class DataContainer:
    def __init__(self, dataset: Dataset, **kwargs):
        self._dataset = dataset
        self._dset = {}
        self._fold = None
        super().__init__(**kwargs)

    def split_by_index(self, split: SplitType):
        for key, idx in split.items():
            self._dset[key] = self._dataset.get_subset(idx)

    def kfold_by_index(self, splits: typing.Iterable[SplitType]):
        for fold, split in enumerate(splits):
            self.split_by_index(split)
            self._fold = fold
            yield self
        self._fold = None

    def split_by_csv(self, path: typing.Union[pathlib.Path, str],
                     index_column: str, **kwargs):
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(index_column)
        col = list(df.columns)
        self.split_by_index(self._read_split_from_df(df, col[0]))

    def kfold_by_csv(self, path: typing.Union[pathlib.Path, str],
                     index_column: str, **kwargs) -> DataContainer:
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(index_column)
        folds = list(df.columns)
        splits = [self._read_split_from_df(df, fold) for fold in folds]
        yield from self.kfold_by_index((splits))

    @staticmethod
    def _read_split_from_df(df: pd.DataFrame, col: str):
        split = defaultdict(list)
        for index, row in df[[col]].iterrows():
            split[str(row[col])].append(index)
        return split

    @property
    def dset(self) -> Dataset:
        if not self._dset:
            raise AttributeError("No Split found.")
        else:
            return self._dset

    @property
    def fold(self) -> int:
        if self._fold is None:
            raise AttributeError(
                "Fold not specified. Call `kfold_by_index` first.")
        else:
            return self._fold


class DataContainerID(DataContainer):
    def split_by_id(self, split: SplitType):
        split_idx = defaultdict(list)
        for key, _id in split.items():
            for _i in _id:
                split_idx[key].append(self._dataset.get_index_by_id(_i))
        super().split_by_index(split_idx)

    def kfold_by_id(
            self,
            splits: typing.Iterable[SplitType]) -> DataContainerID:
        for fold, split in enumerate(splits):
            self.split_by_id(split)
            self._fold = fold
            yield self
        self._fold = None

    def split_by_csv_id(self, path: typing.Union[pathlib.Path, str],
                        id_column: str, **kwargs):
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(id_column)
        col = list(df.columns)
        self.split_by_id(self._read_split_from_df(df, col[0]))

    def kfold_by_csv_id(self, path: typing.Union[pathlib.Path, str],
                        id_column: str, **kwargs) -> DataContainerID:
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(id_column)
        folds = list(df.columns)
        splits = [self._read_split_from_df(df, fold) for fold in folds]
        yield from self.kfold_by_id((splits))

    def save_split_to_csv_id(self,
                             path: typing.Union[pathlib.Path,
                                                str],
                             id_key: str,
                             split_column: str = 'split',
                             **kwargs):
        split_dict = {str(id_key): [], str(split_column): []}
        for key, item in self._dset.items():
            for sample in item:
                split_dict[str(id_key)].append(sample[id_key])
                split_dict[str(split_column)].append(str(key))
        pd.DataFrame(split_dict).to_csv(path, **kwargs)