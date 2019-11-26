import copy
import typing
import logging
import math
from sklearn.model_selection import train_test_split, GroupShuffleSplit, \
    KFold, GroupKFold, StratifiedKFold

from rising.loading.dataset import Dataset

logger = logging.getLogger(__file__)

SplitType = typing.Dict[str, list]


# TODO: Add docstrings for Splitter
class Splitter:
    def __init__(self,
                 dataset: Dataset,
                 val_size: typing.Union[int, float],
                 test_size: typing.Union[int, float] = None):
        super().__init__()
        self._dataset = dataset
        self._total_num = len(self._dataset)
        self._idx = list(range(self._total_num))
        self._val = val_size
        self._test = test_size if test_size is not None else 0

        self._check_sizes()

    def _check_sizes(self):
        if self._total_num < 0:
            raise TypeError("Size must be larger than zero, not "
                            "{}".format(self._total_num))
        if self._val < 0:
            raise TypeError("Size must be larger than zero, not "
                            "{}".format(self._val))
        if self._test < 0:
            raise TypeError("Size must be larger than zero, not "
                            "{}".format(self._test))

        self._convert_prop_to_num()
        if self._total_num < self._val + self._test:
            raise TypeError("Val + test size must be smaller than total, "
                            "not {}".format(self._val + self._test))

    def index_split(self, **kwargs) -> SplitType:
        split_dict = {}
        split_dict["train"], tmp = train_test_split(
            self._idx, test_size=self._val + self._test,
            **kwargs)

        if self._test > 0:
            split_dict["val"], split_dict["test"] = train_test_split(
                tmp, test_size=self._val, **kwargs)
        else:
            split_dict["val"] = tmp
        self.log_split(split_dict, "Created Single Split with:")
        return split_dict

    def index_split_stratified(
            self,
            stratify_key: str = "label",
            **kwargs) -> SplitType:
        split_dict = {}
        stratify = [d[stratify_key] for d in self._dataset]

        split_dict["train"], tmp = train_test_split(
            self._idx, test_size=self._val + self._test, stratify=stratify, **kwargs)

        if self._test > 0:
            stratify_tmp = [stratify[_i] for _i in tmp]
            split_dict["val"], split_dict["test"] = train_test_split(
                tmp, test_size=self._val, stratify=stratify_tmp, **kwargs)
        else:
            split_dict["val"] = tmp
        self.log_split(split_dict, "Created Single Split with:")
        return split_dict

    def index_split_grouped(
            self,
            groups_key: str = "id",
            **kwargs) -> SplitType:
        """
        ..warning:: Shuffling cannot be deactivated
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

    def _convert_prop_to_num(self, attributes: tuple = ("_val", "_test")):
        for attr in attributes:
            value = getattr(self, attr)
            if value < 1 and math.isclose(value, 0):
                setattr(self, attr, value * self._total_num)

    @staticmethod
    def log_split(dict_like: dict, desc: str = None):
        if desc is not None:
            logger.info(desc)
        for key, item in dict_like.items():
            logger.info(f"{str(key).upper()} contains {len(item)}  indices.")

    @staticmethod
    def _copy_and_fill_dict(dict_like: dict, **kwargs) -> dict:
        new_dict = copy.deepcopy(dict_like)
        for key, item in kwargs.items():
            new_dict[key] = item
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
        self._check_sizes()

    @property
    def test_size(self) -> int:
        return self._test

    @test_size.setter
    def test_size(self, value: typing.Union[int, float]):
        self._test = value
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