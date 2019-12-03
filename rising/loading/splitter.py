import copy
import typing
import logging
import warnings
from sklearn.model_selection import train_test_split, GroupShuffleSplit, \
    KFold, GroupKFold, StratifiedKFold

from rising.loading.dataset import Dataset

logger = logging.getLogger(__file__)

SplitType = typing.Dict[str, list]


class Splitter:
    def __init__(self,
                 dataset: Dataset,
                 val_size: typing.Union[int, float] = 0,
                 test_size: typing.Union[int, float] = 0):
        """
        Splits a dataset by several options

        Parameters
        ----------
        dataset : Dataset
            the dataset to split
        val_size : float, int
            the validation split;
                if float this will be interpreted as a percentage of the
                    dataset
                if int this will be interpreted as the number of samples
        test_size : float, int , optionally
            the size of the validation split; If provided it must be int or
            float.
                if float this will be interpreted as a percentage of the
                    dataset
                if int this will be interpreted as the number of samples
            if not provided or explicitly set to None, no testset will be
            created
        """
        super().__init__()
        if val_size == 0 and test_size == 0:
            warnings.warn("Can not perform splitting if val and test size is 0.")

        self._dataset = dataset
        self._total_num = len(self._dataset)
        self._idx = list(range(self._total_num))
        self._val = val_size
        self._test = test_size

        self._convert_prop_to_num()
        self._check_sizes()

    def _check_sizes(self):
        """
        Checks if the given sizes are valid for splitting

        Raises
        ------
        ValueError
            at least one of the sizes is invalid

        """
        if self._total_num <= 0:
            raise ValueError("Size must be larger than zero, not "
                             "{}".format(self._total_num))
        if self._val <= 0:
            raise ValueError("Size must be larger than zero, not "
                             "{}".format(self._val))
        if self._test < 0:
            raise ValueError("Size must be larger than zero, not "
                             "{}".format(self._test))

        if self._total_num < self._val + self._test:
            raise ValueError("Val + test size must be smaller than total, "
                             "not {}".format(self._val + self._test))

    def index_split(self, **kwargs) -> SplitType:
        """
        Splits the dataset's indices in a random way

        Parameters
        ----------
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        dict
            the dictionary containing the corresponding splits under the
            keys 'train', 'val' and (optionally) 'test'

        """
        split_dict = {}
        split_dict["train"], tmp = train_test_split(
            self._idx, test_size=self._val + self._test,
            **kwargs)

        if self._test > 0:
            # update stratified if provided,
            # necessary for index_split_stratified
            if 'stratify' in kwargs:
                kwargs['stratify'] = [kwargs['stratify'][_i] for _i in tmp]
            split_dict["val"], split_dict["test"] = train_test_split(
                tmp, test_size=self._val, **kwargs)
        else:
            split_dict["val"] = tmp
        self.log_split(split_dict, "Created Single Split with:")
        return split_dict

    def index_split_stratified(self, stratify_key: str = "label", **kwargs) -> SplitType:
        """
        Splits the dataset's indices in a stratified way

        Parameters
        ----------
        stratify_key : str
            the key specifying which value of each sample to use for
            stratification
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        dict
            the dictionary containing the corresponding splits under the
            keys 'train', 'val' and (optionally) 'test'

        """
        stratify = [d[stratify_key] for d in self._dataset]
        return self.index_split(stratify=stratify, **kwargs)

    def index_split_grouped(self, groups_key: str = "id", **kwargs) -> SplitType:
        """
        Splits the dataset's indices in a stratified way

        Parameters
        ----------
        groups_key : str
            the key specifying which value of each sample to use for
            grouping
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        dict
            the dictionary containing the corresponding splits under the
            keys 'train', 'val' and (optionally) 'test'

        Warnings
        --------
        Shuffling cannot be deactivated
        """
        split_dict = {}
        groups = [d[groups_key] for d in self._dataset]

        gsp = GroupShuffleSplit(
            n_splits=1, test_size=self._val + self._test, **kwargs)
        split_dict["train"], tmp = next(gsp.split(self._idx, groups=groups))

        if self._test > 0:
            groups_tmp = [groups[_i] for _i in tmp]
            gsp = GroupShuffleSplit(n_splits=1, test_size=self._val, **kwargs)
            split_dict["val"], split_dict["test"] = next(
                gsp.split(tmp, groups=groups_tmp))
        else:
            split_dict["val"] = tmp
        self.log_split(split_dict, "Created Single Split with:")
        return split_dict

    def index_kfold_fixed_test(self, **kwargs) -> typing.Iterable[SplitType]:
        """
        Calculates splits for a random kfold with given testset.
        If :param:`test_size` is zero, a normal kfold is generated

        Parameters
        ----------
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        list
            list containing one dict for each fold each containing the
            corresponding splits under the keys 'train', 'val' and 'test'

        """
        splits = []

        idx_dict = self.index_split(**kwargs)
        train_val_idx = idx_dict.pop("train") + idx_dict.pop("val")

        logger.info("Creating {} folds.".format(self.val_folds))
        kf = KFold(n_splits=self.val_folds, **kwargs)
        _fold = 0
        for train_idx, val_idx in kf.split(train_val_idx):
            splits.append(self._copy_and_fill_dict(
                idx_dict, train=train_idx, val=val_idx))
            self.log_split(splits[-1], f"Created Fold{_fold}.")
            _fold += 1
        return splits

    def index_kfold_fixed_test_stratified(
            self,
            stratify_key: str = "label",
            **kwargs) -> typing.Iterable[SplitType]:
        """
        Calculates splits for a stratified kfold with given testset
        If :param:`test_size` is zero, a normal kfold is generated

        Parameters
        ----------
        stratify_key : str
            the key specifying which value of each sample to use for
            stratification
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        list
            list containing one dict for each fold each containing the
            corresponding splits under the keys 'train', 'val' and 'test'

        """
        splits = []

        idx_dict = self.index_split_stratified(**kwargs)
        train_val_idx = idx_dict.pop("train") + idx_dict.pop("val")
        train_val_stratify = [
            self._dataset[_i][stratify_key] for _i in train_val_idx]

        logger.info("Creating {} folds.".format(self.val_fols))
        kf = StratifiedKFold(n_splits=self.val_folds, **kwargs)
        _fold = 0
        for train_idx, val_idx in kf.split(train_val_idx, train_val_stratify):
            splits.append(self._copy_and_fill_dict(
                idx_dict, train=train_idx, val=val_idx))
            self.log_split(splits[-1], f"Created Fold{_fold}.")
            _fold += 1
        return splits

    def index_kfold_fixed_test_grouped(self, groups_key: str = "id",
                                       **kwargs) -> typing.Iterable[SplitType]:
        """
        Calculates splits for a stratified kfold with given testset
        If :param:`test_size` is zero, a normal kfold is generated

        Parameters
        ----------
        groups_key : str
            the key specifying which value of each sample to use for
            grouping
        **kwargs :
            optional keyword arguments.
            See :func:`sklearn.model_selection.train_test_split` for details

        Returns
        -------
        list
            list containing one dict for each fold each containing the
            corresponding splits under the keys 'train', 'val' and 'test'

        """
        splits = []

        idx_dict = self.index_split_grouped(**kwargs)
        train_val_idx = idx_dict.pop("train") + idx_dict.pop("val")
        train_val_groups = [
            self._dataset[_i][groups_key] for _i in train_val_idx]

        logger.info("Creating {} folds.".format(self.val_fols))
        kf = GroupKFold(n_splits=self.val_folds, **kwargs)
        _fold = 0
        for train_idx, val_idx in kf.split(
                train_val_idx, groups=train_val_groups):
            splits.append(self._copy_and_fill_dict(
                idx_dict, train=train_idx, val=val_idx))
            self.log_split(splits[-1], f"Created Fold{_fold}.")
            _fold += 1
        return splits

    def _convert_prop_to_num(self, attributes: tuple = ("_val", "_test")
                             ) -> None:
        """
        Converts all given attributes from percentages to number of samples
        if necessary

        Parameters
        ----------
        attributes : tuple
            tuple of strings containing the attribute names

        """
        for attr in attributes:
            value = getattr(self, attr)
            if 0 < value < 1:
                setattr(self, attr, value * self._total_num)

    @staticmethod
    def log_split(dict_like: dict, desc: str = None) -> None:
        """
        Logs the new created split

        Parameters
        ----------
        dict_like : dict
            the splits (usually this dict contains the keys 'train', 'val'
            and (optionally) 'test' and a list of indices for each of them
        desc : str, optional
            the descriptor string to log before the actual splits

        """
        if desc is not None:
            logger.info(desc)
        for key, item in dict_like.items():
            logger.info(f"{str(key).upper()} contains {len(item)}  indices.")

    @staticmethod
    def _copy_and_fill_dict(dict_like: dict, **kwargs) -> dict:
        """
        copies the dict and adds the kwargs to the copy

        Parameters
        ----------
        dict_like : dict
            the dict to copy and fill
        **kwargs :
            the keyword argument added to the dict copy

        Returns
        -------
        dict
            the copied and filled dict

        """
        new_dict = copy.deepcopy(dict_like)
        new_dict.update(kwargs)
        return new_dict

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, dset: Dataset):
        self._dataset = dset
        self._total_num = len(self._dataset)
        self._idx = list(range(self._total_num))

    @property
    def val_size(self) -> int:
        return self._val

    @val_size.setter
    def val_size(self, value: typing.Union[int, float]):
        self._val = value
        self._convert_prop_to_num()
        self._check_sizes()

    @property
    def test_size(self) -> int:
        return self._test

    @test_size.setter
    def test_size(self, value: typing.Union[int, float]):
        self._test = value
        self._convert_prop_to_num()
        self._check_sizes()

    @property
    def folds(self) -> int:
        return self.val_folds * self.test_folds

    @property
    def val_folds(self) -> int:
        return int(self._total_num // self._val)

    @property
    def test_folds(self) -> int:
        return int(self._total_num // self._test)