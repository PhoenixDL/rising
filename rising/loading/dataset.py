from __future__ import annotations

import os
import pathlib
import logging
import dill
from warnings import warn
from functools import partial
from tqdm import tqdm
from typing import Any, Sequence, Callable, Union, List, Hashable, Dict, Iterator

from torch.utils.data import Dataset as TorchDset, Subset
from multiprocessing import cpu_count, Pool as MPPool
from torch.multiprocessing import Pool
from rising.loading.debug_mode import get_debug_mode
from rising import AbstractMixin

logger = logging.getLogger(__file__)


__all__ = ["Dataset", "CacheDataset", "LazyDataset", "CacheDatasetID", "LazyDatasetID", "LazyDatasetMulReturn"]


def dill_helper(payload: Any) -> Any:
    """
    Load single sample from data serialized by dill

    Parameters
    ----------
    payload : Any
        data which is loaded with dill

    Returns
    -------
    Any
        loaded data
    """
    fn, args, kwargs = dill.loads(payload)
    return fn(*args, **kwargs)


def load_async(pool: MPPool, fn: Callable, *args, callback: Callable = None, **kwargs) -> Any:
    """
    Load data asynchronously and serialize data via dill

    Parameters
    ----------
    pool : multiprocessing.Pool
        multiprocessing pool to use for :func:`apply_async`
    fn : Callable
        function to load a single sample
    callback : Callable, optional
        optional callback, by default None

    Returns
    -------
    Any
        reference to obtain data with :func:`get`
    """
    payload = dill.dumps((fn, args, kwargs))
    return pool.apply_async(dill_helper, (payload,), callback=callback)


class Dataset(TorchDset):
    """
    Extension of PyTorch's Datasets by a ``get_subset`` method which returns a
    sub-dataset.
    """

    def __iter__(self) -> Any:
        """
        Simple iterator over dataset

        Returns
        -------
        Any
            data contained inside dataset
        """
        for i in range(len(self)):
            yield self[i]

    def get_subset(self, indices: Sequence[int]) -> Subset:
        """
        Returns a Subset of the current dataset based on given indices

        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset

        Returns
        -------
        :class:`SubsetDataset`
            the subset
        """
        subset = Subset(self, indices)
        subset.__iter__ = self.__iter__
        return subset


class CacheDataset(Dataset):
    def __init__(self,
                 data_path: Union[pathlib.Path, str, list],
                 load_fn: Callable, mode: str = "append",
                 num_workers: int = 0, verbose=False,
                 **load_kwargs):
        """
        A dataset to preload all the data and cache it for the entire
        lifetime of this class.

        Parameters
        ----------
        data_path : str, Path or list
            the path(s) containing the actual data samples
        load_fn : function
            function to load the actual data
        mode : str
            whether to append the sample to a list or to extend the list by
            it. Supported modes are: :param:`append` and :param:`extend`.
            Default: ``append``
        num_workers : int, optional
            the number of workers to use for preloading. ``0`` means, all the
            data will be loaded in the main process, while ``None`` means,
            the number of processes will default to the number of logical
            cores.
        verbose : bool
            whether to show the loading progress.
        **load_kwargs :
            additional keyword arguments. Passed directly to :param:`load_fn`

        Warnings
        --------
        if using multiprocessing to load data, there are some restrictions to which
        :func:`load_fn` are supported, please refer to the dill or pickle documentation
        """
        super().__init__()

        if get_debug_mode() and (num_workers is None or num_workers > 0):
            warn("The debug mode has been activated. "
                 "Falling back to num_workers = 0", UserWarning)
            num_workers = 0

        self._num_workers = num_workers
        self._verbosity = verbose

        self._load_fn = load_fn
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(data_path, mode)

    def _make_dataset(self, path: Union[pathlib.Path, str, list], mode: str) -> List[Any]:
        """
        Function to build the entire dataset

        Parameters
        ----------
        path : str, Path or list
            the path(s) containing the data samples
        mode : str
            whether to append or extend the dataset by the loaded sample

        Returns
        -------
        list
            the loaded data

        """
        data = []
        if not isinstance(path, list):
            assert os.path.isdir(path), '%s is not a valid directory' % path
            path = [os.path.join(path, p) for p in os.listdir(path)]

        # sort for reproducibility (this is done explicitly since the listdir
        # function does not return the paths in an ordered way on all OS)
        path = sorted(path)

        # add loading kwargs
        load_fn = partial(self._load_fn, **self._load_kwargs)

        if self._num_workers is None or self._num_workers > 0:
            _data = self.load_multi_process(load_fn, path)
        else:
            _data = self.load_single_process(load_fn, path)

        for sample in _data:
            self._add_item(data, sample, mode)
        return data

    def load_single_process(self, load_fn: Callable, path: Sequence) -> Iterator:
        """
        Helper function to load dataset with single process

        Parameters
        ----------
        load_fn : Callable
            function to load a linge sample
        path : Sequence
            a sequence of paths whih should be loaded

        Returns
        -------
        Iterator
            iterator of loaded data
        """
        if self._verbosity:
            path = tqdm(path, unit='samples', desc="Loading Samples")

        return map(load_fn, path)

    def load_multi_process(self, load_fn: Callable, path: Sequence) -> List:
        """
        Helper function to load dataset with multiple processes

        Parameters
        ----------
        load_fn : Callable
            function to load a linge sample
        path : Sequence
            a sequence of paths whih should be loaded

        Returns
        -------
        List
            list of loaded data
        """
        _processes = cpu_count() if self._num_workers is None else self._num_workers

        if self._verbosity:
            pbar = tqdm(total=len(path), unit='samples', desc="Loading Samples")

            def update(*a):
                pbar.update(1)
            callback = update
        else:
            callback = None

        with Pool(processes=_processes) as pool:
            jobs = [load_async(pool, load_fn, p, callback=callback) for p in path]
            _data = [j.get() for j in jobs]
        return _data

    @staticmethod
    def _add_item(data: list, item: Any, mode: str) -> None:
        """
        Adds items to the given data list. The actual way of adding these
        items depends on :param:`mode`

        Parameters
        ----------
        data : list
            the list containing the already loaded data
        item : Any
            the current item which will be added to the list
        mode : str
            the string specifying the mode of how the item should be added.

        """
        _mode = mode.lower()

        if _mode == 'append':
            data.append(item)
        elif _mode == 'extend':
            data.extend(item)
        else:
            raise TypeError(f"Unknown mode detected: {mode} not supported.")

    def __getitem__(self, index: int) -> Any:
        """
        Making the whole Dataset indexeable.

        Parameters
        ----------
        index : int
            the integer specifying which sample to return

        Returns
        -------
        Any, Dict
            can be any object containing a single sample, but in practice is
            often a dict

        """
        return self.data[index]

    def __len__(self) -> int:
        """
        Length of dataset

        Returns
        -------
        int
            number of elements
        """
        return len(self.data)


class LazyDataset(Dataset):
    def __init__(self, data_path: Union[str, pathlib.Path, list], load_fn: Callable, **load_kwargs):
        """
        A dataset to load all the data just in time.

        Parameters
        ----------
        data_path : str, Path or list
            the path(s) containing the actual data samples
        load_fn : function
            function to load the actual data
        load_kwargs:
            additional keyword arguments (passed to :param:`load_fn`)
        """
        super().__init__()
        self._load_fn = load_fn
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(data_path)

    def _make_dataset(self, path: Union[pathlib.Path, str, list]) -> List[str]:
        """
        Function to build the entire dataset

        Parameters
        ----------
        path : str, Path or list
            the path(s) containing the data samples

        Returns
        -------
        list
            the loaded data

        """
        if not isinstance(path, list):
            assert os.path.isdir(path), '%s is not a valid directory' % path
            path = [os.path.join(path, p) for p in os.listdir(path)]

        sorted(path)
        return path

    def __getitem__(self, index: int) -> Any:
        """
        Making the whole Dataset indexeable. Loads the necessary sample.

        Parameters
        ----------
        index : int
            the integer specifying which sample to load and return

        Returns
        -------
        Any, Dict
            can be any object containing a single sample, but in practice is
            often a dict

        """
        data_dict = self._load_fn(self.data[index], **self._load_kwargs)
        return data_dict

    def __len__(self) -> int:
        """
        Length of dataset

        Returns
        -------
        int
            number of elements
        """
        return len(self.data)


class IDManager(AbstractMixin):
    def __init__(self, id_key: str, cache_ids: bool = True, **kwargs):
        """
        Helper class which can be used as an baseclass for datasets with
        support for samples with unique ID. This class implements
        additional function which can be used to select samples
        by an ID (similar to dicts) instead of their index.
        Because get_subset is overwritten this class should be the first class
        in the mro.

        Parameters
        ----------
        id_key : str
            the id key to cache
        cache_ids : bool
            whether to cache the ids
        **kwargs :
            additional keyword arguments
        """
        super().__init__(**kwargs)
        self.id_key = id_key
        self._cached_ids = None

        if cache_ids:
            self.cache_ids()

    def cache_ids(self) -> None:
        """
        Caches the IDs

        """
        self._cached_ids = {
            sample[self.id_key]: idx for idx, sample in enumerate(self)}

    def _find_index_iterative(self, id: str) -> int:
        """
        Checks for the next index matching the given id

        Parameters
        ----------
        id : str
            the id to get the index for

        Returns
        -------
        int
            the returned index

        Raises
        ------
        KeyError
            no index matching the given id

        """
        for idx, sample in enumerate(self):
            if sample[self.id_key] == id:
                return idx
        raise KeyError(f"ID {id} not found.")

    def get_sample_by_id(self, id: str) -> dict:
        """
        Fetches the sample to a corresponding ID

        Parameters
        ----------
        id : str
            the id specifying the sample to return

        Returns
        -------
        dict
            the sample corresponding to the given ID

        """
        return self[self.get_index_by_id(id)]

    def get_index_by_id(self, id: str) -> int:
        """
        Returns the index corresponding to a given id

        Parameters
        ----------
        id : str
            the id specifying the index of which sample should be returned

        Returns
        -------
        int
            the index of the sample matching the given id

        """
        if self._cached_ids is not None:
            return self._cached_ids[id]
        else:
            return self._find_index_iterative(id)

    def get_subset(self, indices: Sequence[int]) -> Subset:
        """
        Get subset from dataset

        Parameters
        ----------
        indices: Sequence
            indices to select for subset

        Returns
        -------
        Subset
            subst of old dataset
        """
        try:
            subset = super().get_subset(indices)
            subset.cache_ids = self.cache_ids
            subset._find_index_iterative = self._find_index_iterative
            subset.get_sample_by_id = self.get_sample_by_id
            subset.get_index_by_id = self.get_index_by_id
            return subset
        except AttributeError:
            warn(f"Get subset failed, try to recover form it by manually creating a subset.", UserWarning)
            return Subset(self, indices)


class CacheDatasetID(IDManager, CacheDataset):
    def __init__(self, data_path: Union[str, pathlib.Path, list], load_fn: Callable[[Any], Dict],
                 id_key: Hashable, cache_ids: bool = True, **kwargs):
        """
        Extends CacheDataset with an option to draw samples by their ID (similar to dicts).

        Parameters
        ----------
        data_path : str, Path or list
            the path(s) containing the actual data samples
        load_fn : Callable[[Any], Dict]
            function to load the actual data
        id_key : str
            the id key inside the data dict which should be used as an identifier
        cache_ids : bool
            if `True` the ids are cached which speeds up lookups but costs more memory
        **kwargs :
            additional keyword arguments
        """
        super().__init__(data_path=data_path, load_fn=load_fn, id_key=id_key,
                         cache_ids=cache_ids, **kwargs)


class LazyDatasetID(IDManager, LazyDataset):
    def __init__(self, data_path: Union[str, pathlib.Path, list], load_fn: Callable[[Any], Dict],
                 id_key: Hashable, cache_ids: bool = True, **kwargs):
        """
        Extends LazyDataset with an option to draw samples by their ID (similar to dicts).

        Parameters
        ----------
        data_path : str, Path or list
            the path(s) containing the actual data samples
        load_fn : Callable[[Any], Dict]
            function to load the actual data
        id_key : str
            the id key inside the data dict which should be used as an identifier
        cache_ids : bool
            if `True` the ids are cached which speeds up lookups but costs more memory
        **kwargs :
            additional keyword arguments
        """
        super().__init__(data_path=data_path, load_fn=load_fn, id_key=id_key,
                         cache_ids=cache_ids, **kwargs)


class LazyDatasetMulReturn(LazyDataset):
    def __init__(self, data_path: Union[str, pathlib.Path, list], load_fn: Callable[[Any], Sequence],
                 num_samples: int = None, **load_kwargs):
        """
        Lazy dataset which accepts callable which return multiple samples at once

        Parameters
        ----------
        data_path : str, Path or list
            the path(s) containing the actual data samples
        load_fn : function
            function to load the actual data
        num_samples: int
            if num_samples is None, the dataset iterates through the data
            to determine the number of samples and create an internal mapping.
            If :param:`load_fn` always returns the same number of samples the
            iteration can be skipped and the mapping is directly created,
            assuming :param:`load_fn` always return :param:`num_samples`.
        load_kwargs:
            additional keyword arguments passed to :param:`load_fn`
        """
        super().__init__(data_path=data_path, load_fn=load_fn, **load_kwargs)
        self.mapping = None
        self.num_samples = num_samples
        self.cached_index = None
        self.cached_data = None
        self.create_mapping()

    def __getitem__(self, index: int) -> Any:
        """
        Return a single sample

        Parameters
        ----------
        index: int
            index of sample

        Returns
        -------
        Any
            data
        """
        if self.mapping[index][0] != self.cached_index:
            self.cached_data = super().__getitem__(index)
            self.cached_index = self.mapping[index][0]
        return self.cached_data[self.mapping[index][1]]

    def create_mapping(self) -> None:
        """
        Creates an internal mapping of index to individual data samples
        """
        self.mapping = []
        _data = []
        for _index, path in enumerate(self.data):
            if self.num_samples is None:
                _num_samples = len(super().__getitem__(_index))
            else:
                _num_samples = self.num_samples

            self.mapping.extend([(_index, _offset) for _offset in range(_num_samples)])
            _data.extend([path] * _num_samples)
        self.data = _data
