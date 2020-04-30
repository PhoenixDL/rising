import logging
import os
import pathlib
from functools import partial
from typing import Any, Sequence, Callable, Union, List, Iterator

import dill
from torch.multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset as TorchDset, Subset
from tqdm import tqdm

logger = logging.getLogger(__file__)

__all__ = ["Dataset", "AsyncDataset"]


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


def load_async(pool: Pool, fn: Callable, *args, callback: Callable = None, **kwargs) -> Any:
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


class AsyncDataset(Dataset):
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
