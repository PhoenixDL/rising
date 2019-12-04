from __future__ import annotations

import pandas as pd
import typing
import pathlib
from collections import defaultdict

from rising.loading.dataset import Dataset
from rising.loading.splitter import SplitType


class DataContainer:
    def __init__(self, dataset: Dataset):
        """
        Handles the splitting of datasets from different sources

        Parameters
        ----------
        dataset : dataset
            the dataset to split
        kwargs
        """
        self._dataset = dataset
        self._dset = {}
        self._fold = None
        super().__init__()

    def split_by_index(self, split: SplitType) -> None:
        """
        Splits dataset by a given split-dict

        Parameters
        ----------
        split : dict
            a dictionary containing tuples of strings and lists of indices
            for each split

        """
        for key, idx in split.items():
            self._dset[key] = self._dataset.get_subset(idx)

    def kfold_by_index(self, splits: typing.Iterable[SplitType]):
        """
        Produces kfold splits based on the given indices.

        Parameters
        ----------
        splits : list
            list containing split dicts for each fold

        Yields
        ------
        DataContainer
            the data container with updated dataset splits

        """
        for fold, split in enumerate(splits):
            self.split_by_index(split)
            self._fold = fold
            yield self
        self._fold = None

    def split_by_csv(self, path: typing.Union[pathlib.Path, str],
                     index_column: str, **kwargs) -> None:
        """
        Splits a dataset by splits given in a CSV file

        Parameters
        ----------
        path : str, pathlib.Path
            the path to the csv file
        index_column : str
            the label of the index column
        **kwargs :
            additional keyword arguments (see :func:`pandas.read_csv` for
            details)

        """
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(index_column)
        col = list(df.columns)
        self.split_by_index(self._read_split_from_df(df, col[0]))

    def kfold_by_csv(self, path: typing.Union[pathlib.Path, str],
                     index_column: str, **kwargs) -> DataContainer:
        """
        Produces kfold splits based on the given csv file.

        Parameters
        ----------
        path : str, pathlib.Path
            the path to the csv file
        index_column : str
            the label of the index column
        **kwargs :
            additional keyword arguments (see :func:`pandas.read_csv` for
            details)

        Yields
        ------
        DataContainer
            the data container with updated dataset splits

        """
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(index_column)
        folds = list(df.columns)
        splits = [self._read_split_from_df(df, fold) for fold in folds]
        yield from self.kfold_by_index((splits))

    @staticmethod
    def _read_split_from_df(df: pd.DataFrame, col: str) -> SplitType:
        """
        Helper function to read a split from a given data frame

        Parameters
        ----------
        df : pandas.DataFrame
            the dataframe containing the split
        col : str
            the column inside the data frame containing the split

        Returns
        -------
        dict
            a dictionary of lists. Contains a string-list-tuple per split

        """
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
    """
    Data Container Class for datasets with an ID
    """

    def split_by_id(self, split: SplitType) -> None:
        """
        Splits the internal dataset by the given splits

        Parameters
        ----------
        split : dict
            dictionary containing a string-list tuple per split

        """
        split_idx = defaultdict(list)
        for key, _id in split.items():
            for _i in _id:
                split_idx[key].append(self._dataset.get_index_by_id(_i))
        return super().split_by_index(split_idx)

    def kfold_by_id(
            self,
            splits: typing.Iterable[SplitType]):
        """
        Produces kfold splits by an ID

        Parameters
        ----------
        splits : list
            list of dicts each containing the splits for a separate fold

        Yields
        ------
        DataContaimnerID
            the data container with updated internal datasets

        """
        for fold, split in enumerate(splits):
            self.split_by_id(split)
            self._fold = fold
            yield self
        self._fold = None

    def split_by_csv_id(self, path: typing.Union[pathlib.Path, str],
                        id_column: str, **kwargs) -> None:
        """
        Splits the internal dataset by a given id column in a given csv file

        Parameters
        ----------
        path : str or pathlib.Path
            the path to the csv file
        id_column : str
            the key of the id_column
        **kwargs :
            additionalm keyword arguments (see :func:`pandas.read_csv` for
            details)

        """
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(id_column)
        col = list(df.columns)
        return self.split_by_id(self._read_split_from_df(df, col[0]))

    def kfold_by_csv_id(self, path: typing.Union[pathlib.Path, str],
                        id_column: str, **kwargs):
        """
       Produces kfold splits by an ID column of a given csv file

       Parameters
       ----------
       path : str or pathlib.Path
            the path to the csv file
        id_column : str
            the key of the id_column
        **kwargs :
            additionalm keyword arguments (see :func:`pandas.read_csv` for
            details)

       Yields
       ------
       DataContaimnerID
           the data container with updated internal datasets

       """
        df = pd.read_csv(path, **kwargs)
        df = df.set_index(id_column)
        folds = list(df.columns)
        splits = [self._read_split_from_df(df, fold) for fold in folds]
        yield from self.kfold_by_id((splits))

    def save_split_to_csv_id(self,
                             path: typing.Union[pathlib.Path, str],
                             id_key: str,
                             split_column: str = 'split',
                             **kwargs) -> None:
        """
        Saves a split top a given csv id

        Parameters
        ----------
        path : str or pathlib.Path
            the path of the csv file
        id_key : str
            the id key inside the csv file
        split_column : str
            the name of the split_column inside the csv file
        **kwargs :
            additional keyword arguments (see :meth:`pd.DataFrame.to_csv`
            for details)

        """
        split_dict = {str(id_key): [], str(split_column): []}
        for key, item in self._dset.items():
            for sample in item:
                split_dict[str(id_key)].append(sample[id_key])
                split_dict[str(split_column)].append(str(key))
        pd.DataFrame(split_dict).to_csv(path, **kwargs)
