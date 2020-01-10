from __future__ import annotations

import torch
import warnings

from typing import Callable, Mapping, Sequence, Union, Any
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import DataLoader as _DataLoader, Sampler
from torch.utils.data.dataloader import \
    _SingleProcessDataLoaderIter as __SingleProcessDataLoaderIter, \
    _MultiProcessingDataLoaderIter as __MultiProcessingDataLoaderIter
from functools import partial
from threadpoolctl import threadpool_limits

from rising.loading.collate import do_nothing_collate
from rising.transforms import ToDevice, Compose
from rising.loading.debug_mode import get_debug_mode
from rising.loading.dataset import Dataset


def default_transform_call(batch: Any, transform: Callable) -> Any:
    """
    Default function to call transforms. Mapping and Sequences are
    unpacked during the transform call. Other types are passed
    as a positional argument.

    Parameters
    ----------
    batch: Any
        current batch which is passed to transforms
    transform: Callable
        transform to perform

    Returns
    -------
    Any
        transformed batch
    """
    if isinstance(batch, Mapping):
        return transform(**batch)
    elif isinstance(batch, Sequence):
        return transform(*batch)
    else:
        return transform(batch)


class DataLoader(_DataLoader):
    def __init__(self, dataset: Union[Sequence, Dataset],
                 batch_size: int = 1, shuffle: bool = False,
                 batch_transforms: Callable = None,
                 gpu_transforms: Callable = None,
                 device: Union[str, torch.device] = None,
                 sampler: Sampler = None,
                 batch_sampler: Sampler = None, num_workers: int = 0,
                 collate_fn: Callable = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: Union[int, float] = 0,
                 worker_init_fn: Callable = None,
                 multiprocessing_context=None,
                 auto_convert: bool = True,
                 transform_call: Callable[[Any, Callable], Any] = default_transform_call):
        """
        A Dataloader introducing batch-transforms, numpy seeds for worker
        processes and compatibility to the debug mode

        Note
        ----
        For Reproducibility numpy and pytorch must be seeded in the main
        process, as these frameworks will be used to generate their own seeds
        for each worker.

        Note
        ----
        ``len(dataloader)`` heuristic is based on the length of the sampler
        used. When :attr:`dataset` is an
        :class:`~torch.utils.data.IterableDataset`, an infinite sampler is
        used, whose :meth:`__len__` is not implemented, because the actual
        length depends on both the iterable as well as multi-process loading
        configurations. So one should not query this method unless they work
        with a map-style dataset. See `Dataset Types`_ for more details on
        these two types of datasets.

        Warning
        -------
        If the ``spawn`` start method is used, :attr:`worker_init_fn`
        cannot be an unpicklable object, e.g., a lambda function. See
        :ref:`multiprocessing-best-practices` on more details related
        to multiprocessing in PyTorch.

        Note
        -------
        The GPU-Transforms for a batch are always executed in the main
        process after the batch was gathered from subprocesses which apply
        the CPU-Transformations. The desired workflow is as follows:

        Disk -> CPU-Transforms -> GPU-Memory -> GPU-Transforms -> Further
        GPU Processing (e.g. training a neural network)

        Parameters
        ----------
        dataset : Dataset
            dataset from which to load the data
        batch_size : int, optional
            how many samples per batch to load (default: ``1``).
        shuffle : bool, optional
            set to ``True`` to have the data reshuffled at every epoch
            (default: ``False``)
        batch_transforms : callable, optional
            transforms which can be applied to a whole batch.
            Usually this accepts either mappings or sequences and returns the
            same type containing transformed elements
        gpu_transforms : callable, optional
            transforms which can be applied to a whole batch (on the GPU).
            Unlike :param:`batch_transforms` this is not done in multiple
            processes, but in the main process on the GPU, because GPUs are
            capable of non-blocking and asynchronous working.
            Before executing these transforms all data will be moved to
            :param:`device`. This copy is done in a non-blocking way if
            :param:`pin_memory` is set to True.
        device : str, torch.device
            the device to move the data to for gpu_transforms.
            If None: the device will be the current device.
        sampler : torch.utils.data.Sampler, optional
            defines the strategy to draw samples from
            the dataset. If specified, :attr:`shuffle` must be ``False``.
        batch_sampler : torch.utils.data.Sampler, optional
            like :attr:`sampler`, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers : int, optional
            how many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn : callable, optional
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data
            elements are a custom type, or your :attr:`collate_fn` returns a
            batch that is a custom type, see the example below.
        drop_last : bool, optional
            set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller. (default: ``False``)
        timeout : numeric, optional
            if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each
            worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and
            before data loading. (default: ``None``)
        auto_convert : bool, optional
            if set to ``True``, the batches will always be transformed to
            torch.Tensors, if possible. (default: ``True``)
        transform_call: Callable[[Any, Callable], Any], optional
            function which determines how transforms are called. By default
            Mappings and Sequences are unpacked during the transform.
        """
        super().__init__(dataset=dataset, batch_size=batch_size,
                         shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=collate_fn, pin_memory=pin_memory,
                         drop_last=drop_last, timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context)

        if gpu_transforms is not None and not torch.cuda.is_available():
            if hasattr(gpu_transforms, 'to'):
                gpu_transforms = gpu_transforms.to('cpu')
            transforms = (batch_transforms, gpu_transforms) if batch_transforms is not None else gpu_transforms
            batch_transforms = Compose(transforms)
            warnings.warn("No CUDA-capable device was found. "
                          "Applying GPU-Transforms on CPU instead.",
                          RuntimeWarning)
            gpu_transforms = None

        self.collate_fn = BatchTransformer(self.collate_fn,
                                           transforms=batch_transforms,
                                           auto_convert=auto_convert,
                                           transform_call=transform_call)
        if gpu_transforms is not None:
            if device is None:
                device = torch.cuda.current_device()

            to_gpu_trafo = ToDevice(device=device, non_blocking=pin_memory)

            gpu_transforms = Compose(to_gpu_trafo, gpu_transforms)
            gpu_transforms = gpu_transforms.to(device)

        self.gpu_transforms = BatchTransformer(do_nothing_collate,
                                               transforms=gpu_transforms,
                                               auto_convert=auto_convert,
                                               transform_call=transform_call
                                               )

    def __iter__(self) -> Union[_SingleProcessDataLoaderIter,
                                _MultiProcessingDataLoaderIter]:
        """
        Geneator iterator

        Returns
        -------
        Union[_SingleProcessDataLoaderIter,_MultiProcessingDataLoaderIter]
            iterator to load and augment data (can be either single or multiprocessing based)
        """
        if self.num_workers == 0 or get_debug_mode():
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


class BatchTransformer(object):
    """
    A callable wrapping the collate_fn to enable transformations on a
    batch-basis.
    """

    def __init__(self, collate_fn: Callable, transforms: Callable = None,
                 auto_convert: bool = True,
                 transform_call: Callable[[Any, Callable], Any] = default_transform_call):
        """
        Parameters
        ----------
        collate_fn : callable
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        transforms : callable, optional
            transforms which can be applied to a whole batch.
            Usually this accepts either mappings or sequences and returns the
            same type containing transformed elements
        auto_convert : bool, optional
            if set to ``True``, the batches will always be transformed to
            torch.Tensors, if possible. (default: ``True``)
        transform_call: Callable[[Any, Callable], Any], optional
            function which determines how transforms are called. By default
            Mappings and Sequences are unpacked during the transform.
        """

        self._collate_fn = collate_fn
        self._transforms = transforms
        self._auto_convert = auto_convert
        self._transform_call = transform_call

    def __call__(self, *args, **kwargs) -> Any:
        """
        Apply batch workflow: collate -> augmentation -> default_convert

        Returns
        -------
        Any
            batched and augmented data
        """
        batch = self._collate_fn(*args, **kwargs)

        if self._transforms is not None:
            batch = self._transform_call(batch, self._transforms)

        if self._auto_convert:
            batch = default_convert(batch)

        return batch


class _MultiProcessingDataLoaderIter(__MultiProcessingDataLoaderIter):
    # NOTE [ Numpy Seeds ]
    # This class is a subclass of
    # ``torch.utils.data.dataloader._MultiProcessingDataLoaderIter``` and only
    # adds some additional logic to provide differnt seeds for numpy in
    # each worker. These seeds are based on a base seed, which itself get's
    # generated by numpy. So to ensure reproducibility, numpy must be seeded
    # in the main process.
    def __init__(self, loader: DataLoader):
        """
        Iterator over Dataloader. Handles the complete multiprocessing

        Parameters
        ----------
        loader : DataLoader
            the dataloader instance to iterate over
        """
        try:
            import numpy as np
            # generate numpy seed. The range comes so that the seed in each
            # worker (which is this baseseed plus the worker id) is always an
            # uint32. This is because numpy only accepts uint32 as valid seeds
            npy_seed = np.random.randint(0, (2 ** 32) - (1 + loader.num_workers))
        except ImportError:
            # we don't generate a numpy seed here with torch, since we don't
            # need one; if the import fails in the main process it should
            # also fail in child processes
            npy_seed = None

        old_worker_init = loader.worker_init_fn

        if npy_seed is None:
            new_worker_init_fn = old_worker_init
        else:
            new_worker_init_fn = partial(_seed_npy_before_worker_init,
                                         seed=npy_seed,
                                         worker_init_fn=old_worker_init)
        loader.worker_init_fn = new_worker_init_fn

        with threadpool_limits(limits=1, user_api='blas'):
            super().__init__(loader)

        # reset worker_init_fn once the workers have been startet to reset
        # to original state for next epoch
        loader.worker_init_fn = old_worker_init
        self._gpu_transforms = loader.gpu_transforms

    def __next__(self) -> Any:
        """
        Get next item from iterator

        Returns
        -------
        Any
            batched and augmented data
        """
        sample = super().__next__()
        return self._gpu_transforms(sample)


class _SingleProcessDataLoaderIter(__SingleProcessDataLoaderIter):
    def __init__(self, loader: DataLoader):
        """
        Iterator over Dataloader

        Parameters
        ----------
        loader : DataLoader
            the dataloader instance to iterate over
        """
        super().__init__(loader)
        self._gpu_transforms = loader.gpu_transforms

    def __next__(self) -> Any:
        """
        Get next item from iterator

        Returns
        -------
        Any
            batched and augmented data
        """
        sample = super().__next__()
        sample = self._gpu_transforms(sample)
        return sample


def _seed_npy_before_worker_init(worker_id: int, seed: int,
                                 worker_init_fn: Callable = None):
    """
    Wrapper Function to wrap the existing worker_init_fn and seed numpy before
    calling the actual ``worker_init_fn``

    Parameters
    ----------
    worker_id : int
        the number of the worker
    seed : int32
        the base seed in a range of [0, 2**32 - (1 + ``num_workers``)].
        The range ensures, that the whole seed, which consists of the base
        seed and the ``worker_id``, can still be represented as a unit32,
        as it needs to be for numpy seeding
    worker_init_fn : callable, optional
        will be called with the ``worker_id`` after seeding numpy if it is not
        ``None``
    """
    try:
        import numpy as np
        np.random.seed(seed + worker_id)
    except ImportError:
        pass

    if worker_init_fn is not None:
        return worker_init_fn(worker_id)
