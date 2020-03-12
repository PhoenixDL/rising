from rising.loading.dataset import Dataset
import os
import numpy as np
from tqdm import tqdm
from typing import Union
from pathlib import Path

from rising.loading.collate import numpy_collate


def h5ify(dset: Dataset, hdf5_path: Union[str, Path], dset_name: str, 
          overwrite: bool = False, strict: bool = True):
    """

    Parameters
    ----------
    dset : Dataset
        The dataset to write a hdf5 path from
    hdf5_path : str, Path
        the path to save the hdf5 file to
    dset_name : str
        the name of the dataset
    overwrite : bool
        whether to overwrite the file if it already exists
    strict : bool
        whether to fail if a certain value cannot be saved to numpy

    Returns
    -------
    list
        a list of strings indicating the keys, that could not be saved to the
        hdf5 file

    Raises
    ------
    FileExistsError
        if file already exists and overwrite is False

    Notes
    -----
    This function assumes that the keys and shapes are same for each sample
    in the dataset

    """
    import h5py

    if os.path.isfile(hdf5_path) and not overwrite:
        raise FileExistsError

    with h5py.File(hdf5_path, 'w') as file:
        data = file.create_group(dset_name)

        keys_to_skip = []
        dsets = {}

        for idx in tqdm(range(0, len(dset))):
            sample = numpy_collate(dset[idx])

            if isinstance(sample, Sequence) and not isinstance(sample, str):
                sample = {idx: v for idx, v in enumerate(sample)}

            if idx == 0:
                for key, val in sample.items():
                    if not isinstance(val, np.ndarray):
                        if strict:
                            raise TypeError('Invalid Type for key "%s": '
                                            'Got %s but expected %s'
                                            % (str(key), type(val).__name__,
                                               np.ndarray.__name__))
                        if key not in keys_to_skip:
                            keys_to_skip.append(key)

                        continue

                    shape = (len(dset), *val.shape)
                    dsets[key] = data.create_dataset(name=key, shape=shape)

            dsets[key][idx] = val
    return keys_to_skip


class Hdf5Dataset(Dataset):
    def __init__(self, hdf5_path: Union[str, Path], dset_name: str, 
                 keep_open: bool = False):
        """
        Dataset to read from a given hdf5 file.

        Parameters
        ----------
        hdf5_path : str, Path
            the path to the hdf5 file
        dset_name : str
            the datasets name
        keep_open : bool
            whether to keep an open file handle

        Notes
        -----
        The file handle will be opened the first time during the first
        indexing, since otherwise it will throw errors due to multiprocessing
        (h5py Files cannot be pickled).

        Warnings
        --------
        the ``keep_open`` function will cause an open file handle through the
        whole process (as long as this object exists).
        On some file systems this may slow things down,
        if there are to many open file handles.
        """
        super().__init__()

        self.hdf5_path = hdf5_path
        self.dset_name = dset_name
        self.keep_open = keep_open

    def __getitem__(self, item):
        """
        Opens the hdf5 File if necessary and extracts the wanted item from it.

        Parameters
        ----------
        item : int
            the index specifying the item to extract

        Returns
        -------
        dict
            a dictionary containing one sample per key in the dataset
        """

        import h5py
        if isinstance(self.hdf5_path, (str, Path)):
            file = h5py.File(self.hdf5_path, 'r')
        else:
            file = self.hdf5_path

        dset = file.get(self.dset_name)
        sample = {}
        for k, v in dset.items():
            sample[k] = v[item]

        if self.keep_open:
            self.hdf5_path = file
        else:
            path = file.filename
            file.close()
            self.hdf5_path = path

    def __del__(self):
        if not isinstance(self.hdf5_path, (str, Path)):
            self.hdf5_path.close()
