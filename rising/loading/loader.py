import collections
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Generator, Iterator, Mapping, Optional, Sequence, Union

import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as __MultiProcessingDataLoaderIter
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter as __SingleProcessDataLoaderIter

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from rising.loading.collate import do_nothing_collate
from rising.transforms import Compose, ToDevice

__all__ = ["DataLoader", "default_transform_call"]


def default_transform_call(batch: Any, transform: Callable) -> Any:
    """
    Default function to call transforms. Mapping and Sequences are
    unpacked during the transform call. Other types are passed
    as a positional argument.

    Args:
        batch: current batch which is passed to transforms
        transform: transform to perform

    Returns:
        Any: transformed batch

    """

    if isinstance(batch, Mapping):
        return transform(**batch)
    elif isinstance(batch, Sequence):
        return transform(*batch)
    else:
        return transform(batch)


class DataLoader(_DataLoader):
    """
    A DataLoader introducing batch-transforms, per-sample-transforms,
    numpy seeds for worker processes outside the dataset

    .. note::
        For Reproducibility numpy and pytorch must be seeded in the main
        process, as these frameworks will be used to generate their own
        seeds for each worker.

    .. note::
        ``len(dataloader)`` heuristic is based on the length of the sampler
        used. When :attr:`dataset` is an
        :class:`~torch.utils.data.IterableDataset`, an infinite sampler is
        used, whose :meth:`__len__` is not implemented, because the actual
        length depends on both the iterable as well as multi-process
        loading configurations. So one should not query this method unless
        they work with a map-style dataset.

    .. warning::
        If the ``spawn`` start method is used, :attr:`worker_init_fn`
        cannot be an unpicklable object, e.g., a lambda function. See
        :ref:`multiprocessing-best-practices` on more details related
        to multiprocessing in PyTorch.

    .. note::
        The GPU-Transforms for a batch are always executed in the main
        process after the batch was gathered from subprocesses which apply
        the CPU-Transformations. The desired workflow is as follows:

        Disk -> CPU-Transforms -> GPU-Memory -> GPU-Transforms -> Further
        GPU Processing (e.g. training a neural network)
    """

    def __init__(
        self,
        dataset: Union[Sequence, Dataset],
        batch_size: int = 1,
        shuffle: bool = False,
        batch_transforms: Optional[Callable] = None,
        gpu_transforms: Optional[Callable] = None,
        sample_transforms: Optional[Callable] = None,
        pseudo_batch_dim: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: Union[int, float] = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        auto_convert: bool = True,
        transform_call: Callable[[Any, Callable], Any] = default_transform_call,
        **kwargs
    ):
        """
        Args:
            dataset: dataset from which to load the data
            batch_size: how many samples per batch to load (default: ``1``).
            shuffle: set to ``True`` to have the data reshuffled at every epoch
                (default: ``False``)
            batch_transforms: transforms which can be applied to a whole
                batch. Usually this accepts either mappings or sequences and
                returns the same type containing transformed elements
            gpu_transforms: transforms which can be applied to a whole batch
                (on the GPU). Unlike :attr:`batch_transforms` this is not
                done in multiple processes, but in the main process on the
                GPU, because GPUs are capable of non-blocking and asynchronous
                working. Before executing these transforms all data will be
                moved to :attr:`device`. This copy is done in a non-blocking
                way if :attr:`pin_memory` is set to True.
            sample_transforms: transforms applied to each sample (on CPU).
                These are the first transforms applied to the data, since they
                are applied on sample retrieval from dataset before batching
                occurs.
            pseudo_batch_dim: whether the :attr:`sample_transforms` work on
                batches and thus need a pseudo batch dim of 1 to work
                correctly.
            device: the device to move the data to for gpu_transforms.
                If None: the device will be the current device.
            sampler: defines the strategy to draw samples from
                the dataset. If specified, :attr:`shuffle` must be ``False``.
            batch_sampler: like :attr:`sampler`, but returns a batch of
                indices at a time. Mutually exclusive with :attr:`batch_size`,
                :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
            num_workers: how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            collate_fn: merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            pin_memory: If ``True``, the data loader will copy Tensors
                into CUDA pinned memory before returning them.  If your data
                elements are a custom type, or your :attr:`collate_fn` returns a
                batch that is a custom type, see the example below.
            drop_last: set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size.
                If ``False`` and the size of dataset is not divisible by the batch
                size, then the last batch will be smaller. (default: ``False``)
            timeout: if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn: If not ``None``, this will be called on each
                worker subprocess with the worker id
                (an int in ``[0, num_workers - 1]``) as input, after seeding and
                before data loading. (default: ``None``)
            auto_convert: if set to ``True``, the batches will always be
                transformed to :class:`torch.Tensors`, if possible.
                (default: ``True``)
            transform_call: function which determines how transforms are
                called. By default Mappings and Sequences are unpacked during
                the transform.
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            **kwargs
        )

        if gpu_transforms is not None and not torch.cuda.is_available():
            if hasattr(gpu_transforms, "to"):
                gpu_transforms = gpu_transforms.to("cpu")
            transforms = (batch_transforms, gpu_transforms) if batch_transforms is not None else gpu_transforms
            batch_transforms = Compose(transforms)
            warnings.warn(
                "No CUDA-capable device was found. " "Applying GPU-Transforms on CPU instead.", RuntimeWarning
            )
            gpu_transforms = None

        self.batch_transforms = batch_transforms

        if gpu_transforms is not None:
            if device is None:
                device = torch.cuda.current_device()

            to_gpu_trafo = ToDevice(device=device, non_blocking=pin_memory)

            gpu_transforms = Compose(to_gpu_trafo, gpu_transforms)
            gpu_transforms = gpu_transforms.to(device)

        self.device = device
        self.sample_transforms = sample_transforms
        self.pseudo_batch_dim = pseudo_batch_dim and sample_transforms is not None
        self.gpu_transforms = gpu_transforms
        self.auto_convert = auto_convert
        self.transform_call = transform_call

    def get_batch_transformer(self) -> Callable:
        """
        A getter function for the :class:`BatchTransformer`
        Returns:
            BatchTransformer: the initialized BatchTransformer

        """
        # this is a function on purpose, since frameworks like ignite parse
        # the class dict to specify what to treat as args during reinit
        return BatchTransformer(
            self.collate_fn,
            transforms=self.batch_transforms,
            auto_convert=self.auto_convert,
            transform_call=self.transform_call,
        )

    def get_gpu_batch_transformer(self) -> Callable:
        """
        A getter function for the :class:`BatchTransformer` holding the
        GPU-Transforms

        Returns:
            BatchTransformer: the initialized BatchTransformer

        """
        # this is a function on purpose, since frameworks like ignite parse
        # the class dict to specify what to treat as args during reinit
        return BatchTransformer(
            do_nothing_collate,
            transforms=self.gpu_transforms,
            auto_convert=self.auto_convert,
            transform_call=self.transform_call,
        )

    def get_sample_transformer(self) -> Callable:
        """
        A getter function for the :class:`SampleTransformer` holding the
        Per-Sample-Transforms

        Returns:
            SampleTransformer: the initialized SampleTransformer

        """
        # this is a function on purpose, since frameworks like ignite parse
        # the class dict to specify what to treat as args during reinit
        return SampleTransformer(self.dataset, self.sample_transforms, self.pseudo_batch_dim, self.transform_call)

    def __iter__(self) -> Iterator:
        """
        Geneator iterator

        Returns:
            Iterator: iterator to load and augment data (can be either
                single or multiprocessing based)
        """
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


@contextmanager
def patch_worker_init_fn(loader: DataLoader, new_worker_init: Callable) -> Generator:
    """
    Patches the loader to temporarily have the correct worker init function.

    Args:
        loader: the loader to patch
        new_worker_init: the new worker init function

    Yields:
        the patched loader

    """

    old_init = loader.worker_init_fn
    loader.worker_init_fn = new_worker_init

    yield loader

    loader.worker_init_fn = old_init


@contextmanager
def patch_collate_fn(loader: DataLoader) -> Generator:
    """
    Patches the loader to temporarily have the correct collate function

    Args:
        loader: the loader to patch

    Yields:
        the patched loader

    """

    old_collate_fn = loader.collate_fn
    loader.collate_fn = loader.get_batch_transformer()

    yield loader

    loader.collate_fn = old_collate_fn


@contextmanager
def patch_dataset(loader: DataLoader) -> Generator:
    """
    Patches the loader to temporarily replace the dataset by a wrapped dataset
    with transforms.

    Args:
        loader: the dataloader whose dataset should be wrapped

    Yields:
        the patched loader

    """
    old_dset = loader.dataset

    loader._DataLoader__initialized = False
    loader.dataset = loader.get_sample_transformer()
    loader._DataLoader__initialized = True

    yield loader

    loader._DataLoader__initialized = False
    loader.dataset = old_dset
    loader._DataLoader__initialized = True


class BatchTransformer(object):
    """
    A callable wrapping the collate_fn to enable transformations on a
    batch-basis.
    """

    def __init__(
        self,
        collate_fn: Callable,
        transforms: Optional[Callable] = None,
        auto_convert: bool = True,
        transform_call: Callable[[Any, Callable], Any] = default_transform_call,
    ):
        """
        Args:
            collate_fn: merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            transforms: transforms which can be applied to a whole batch.
                Usually this accepts either mappings or sequences and returns the
                same type containing transformed elements
            auto_convert: if set to ``True``, the batches will always be transformed to
                torch.Tensors, if possible. (default: ``True``)
            transform_call: function which determines how transforms are called. By default
                Mappings and Sequences are unpacked during the transform.
        """

        self._collate_fn = collate_fn
        self._transforms = transforms
        self._auto_convert = auto_convert
        self._transform_call = transform_call

    def __call__(self, *args, **kwargs) -> Any:
        """
        Apply batch workflow: collate -> augmentation -> default_convert

        Args:
            *args: positional batch arguments
            **kwargs: keyword batch arguments

        Returns:
            Any: batched and augmented data
        """
        batch = self._collate_fn(*args, **kwargs)

        if self._transforms is not None:
            batch = self._transform_call(batch, self._transforms)

        if self._auto_convert:
            batch = default_convert(batch)

        return batch


class SampleTransformer(object):
    """
    A dataset wrapper applying transforms to each retrieved sample of the
    dataset
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms: Optional[Callable] = None,
        pseudo_batch_dim: bool = False,
        transform_call: Callable[[Any, Callable], Any] = default_transform_call,
    ):
        """

        Args:
            dataset: the dataset holding the samples to wrap
            transforms: the transforms to apply to the dataset
            pseudo_batch_dim: whether the transforms work on batches or samples
            transform_call: function which determines how transforms are called. By default
                Mappings and Sequences are unpacked during the transform.
        """
        self.dataset = dataset
        self.transforms = transforms
        self.pseudo_batch_dim = pseudo_batch_dim
        self.transform_call = transform_call

    def __getitem__(self, item: int) -> Any:
        """
        Returns transformed samples of the dataset

        Args:
            item: the index specifying the sample to retrieve

        Returns:
            Any: the transformed sample

        """
        sample = self.dataset[item]

        if self.pseudo_batch_dim:
            sample = self._change_pseudo_batch_dim(sample, add=True)

        if self.transforms is not None:
            sample = self.transform_call(sample, self.transforms)

        if self.pseudo_batch_dim:
            sample = self._change_pseudo_batch_dim(sample, add=False)

        return sample

    def __len__(self) -> int:
        return len(self.dataset)

    def _change_pseudo_batch_dim(self, sample: Any, add: bool) -> Any:
        """
        Adds or removes the pseudo batch size
        Args:
            sample: the sample to add the batchsize to
            add: whether to add or remove the pseudo batchsize

        Returns:
            Any: sample with added or removed pseudo batchsize

        """
        if isinstance(sample, torch.Tensor) or (NUMPY_AVAILABLE and isinstance(sample, np.ndarray)):
            if add:
                return sample[None]
            else:
                return sample[0]

        elif isinstance(sample, (float, str, int)):
            # don't add pseudo batchsize for these types since you"d have to convert.
            return sample
        elif isinstance(sample, collections.abc.Mapping):
            return {key: self._change_pseudo_batch_dim(sample[key], add=add) for key in sample}
        elif isinstance(sample, tuple) and hasattr(sample, "_fields"):  # namedtuple
            return type(sample)(*[self._change_pseudo_batch_dim(_sample, add=add) for _sample in sample])
        elif isinstance(sample, collections.abc.Sequence):
            return type(sample)([self._change_pseudo_batch_dim(elem, add=add) for elem in sample])

        return sample


class _MultiProcessingDataLoaderIter(__MultiProcessingDataLoaderIter):
    """Iterator over Dataloader. Handles the complete multiprocessing

    This class is a subclass of
    :class:`torch.utils.data.dataloader._MultiProcessingDataLoaderIter` and
    adds some additional logic for seeding numpy in each worker.
    These seeds are based on a base seed, which itselfmis generated by numpy.
    Thus numpy must be seeded in the maine process to ensure reproducibility.

    Additionally this iterator adds functionality for per-sample transforms
    outside the dataset and per-batch transforms on both, CPU and GPU.
    """

    def __init__(self, loader: DataLoader):
        """
        Args:
            loader: the dataloader instance over which to iterate
        """
        try:
            import numpy as np

            # generate numpy seed. The range comes so that the seed in each
            # worker (which is this baseseed plus the worker id) is always an
            # uint32. This is because numpy only accepts uint32 as valid seeds
            npy_seed = np.random.randint(0, (2 ** 32) - (1 + loader.num_workers), dtype=np.uint32)
        except ImportError:
            # we don't generate a numpy seed here with torch, since we don't
            # need one; if the import fails in the main process it should
            # also fail in child processes
            npy_seed = None

        old_worker_init = loader.worker_init_fn

        if npy_seed is None:
            new_worker_init_fn = old_worker_init
        else:
            new_worker_init_fn = partial(_seed_npy_before_worker_init, seed=npy_seed, worker_init_fn=old_worker_init)

        with patch_dataset(loader) as loader:
            with patch_worker_init_fn(loader, new_worker_init_fn) as loader:
                with patch_collate_fn(loader) as loader:
                    with threadpool_limits(limits=1, user_api="blas"):
                        super().__init__(loader)

        self._gpu_transforms = loader.get_gpu_batch_transformer()

    def __next__(self) -> Any:
        """
        Get next item from iterator

        Returns:
            Any: batched and augmented data
        """
        sample = super().__next__()
        return self._gpu_transforms(sample)


class _SingleProcessDataLoaderIter(__SingleProcessDataLoaderIter):
    """Iterator over Dataloader.

    This iterator adds functionality for per-sample transforms
    outside the dataset and per-batch transforms on both, CPU and GPU.

    """

    def __init__(self, loader: DataLoader):
        """
        Args:
            loader: the dataloader instance over which to iterate
        """
        with patch_dataset(loader) as loader:
            with patch_collate_fn(loader) as loader:
                super().__init__(loader)

        self._gpu_transforms = loader.get_gpu_batch_transformer()

    def __next__(self) -> Any:
        """
        Get next item from iterator

        Returns:
            Any: batched and augmented data
        """
        sample = super().__next__()
        sample = self._gpu_transforms(sample)
        return sample


def _seed_npy_before_worker_init(worker_id: int, seed: int, worker_init_fn: Optional[Callable] = None) -> Any:
    """
    Wrapper Function to wrap the existing worker_init_fn and seed numpy before
    calling the actual ``worker_init_fn``

    Args:
        worker_id: the number of the worker
        seed: the base seed in a range of [0, 2**32 - (1 + ``num_workers``)].
            The range ensures, that the whole seed, which consists of the base
            seed and the ``worker_id``, can still be represented as a unit32,
            as it needs to be for numpy seeding
        worker_init_fn: will be called with the ``worker_id`` after seeding
            numpy if it is not ``None``

    Returns:
        Any: result of :attr`worker_init_fn`

    """
    import sys

    if not sys.warnoptions and worker_id > 0:
        import warnings

        warnings.simplefilter("ignore")
    try:
        import numpy as np

        np.random.seed(seed + worker_id)
    except ImportError:
        pass

    if worker_init_fn is not None:
        return worker_init_fn(worker_id)
