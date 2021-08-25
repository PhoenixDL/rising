from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

__all__ = ["box_to_seg", "seg_to_box", "instance_to_semantic", "pop_keys", "filter_keys"]


def box_to_seg(
    boxes: Sequence[Sequence[int]],
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[Union[torch.dtype, str]] = None,
    device: Optional[Union[torch.device, str]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert a sequence of bounding boxes to a segmentation

    Args:
        boxes: sequence of bounding boxes encoded as
            (dim0_min, dim1_min, dim0_max, dim1_max, [dim2_min, dim2_max]).
            Supported bounding boxes for 2D (4 entries per box)
            and 3d (6 entries per box)
        shape: if :attr:`out` is not provided, shape of output tensor must
            be specified
        dtype: if :attr:`out` is not provided,
            dtype of output tensor must be specified
        device: if :attr:`out` is not provided,
            device of output tensor must be specified
        out: if not None, the segmentation will be saved inside this tensor

    Returns:
        torch.Tensor: bounding boxes encoded as a segmentation
    """
    if out is None:
        out = torch.zeros(*shape, dtype=dtype, device=device)

    for _idx, box in enumerate(boxes, 1):
        if len(box) == 4:
            out[..., box[0] : box[2] + 1, box[1] : box[3] + 1] = _idx
        elif (len(box)) == 6:
            out[..., box[0] : box[2] + 1, box[1] : box[3] + 1, box[4] : box[5] + 1] = _idx
        else:
            raise TypeError(f"Boxes must have length 4 (2D) or 6(3D) found len {len(box)}")
    return out


def seg_to_box(seg: torch.Tensor, dim: int) -> List[torch.Tensor]:
    """
    Convert instance segmentation to bounding boxes

    Args:
        seg: segmentation of individual classes
            (index should start from one and be continuous)
        dim: number of spatial dimensions

    Returns:
        list: list of bounding boxes tuple with classes for
            bounding boxes
    """
    boxes = []
    _seg = seg.detach()
    for _idx in range(1, seg.max().detach().item() + 1):
        instance_map = (_seg == _idx).nonzero()
        _mins = instance_map.min(dim=0)[0]
        _maxs = instance_map.max(dim=0)[0]
        box = [_mins[-dim], _mins[-dim + 1], _maxs[-dim], _maxs[-dim + 1]]
        if dim > 2:
            box = box + [c for cv in zip(_mins[-dim + 2 :], _maxs[-dim + 2 :]) for c in cv]
        boxes.append(torch.tensor(box).to(dtype=torch.float, device=seg.device))
    return boxes


def instance_to_semantic(instance: torch.Tensor, cls: Sequence[int]) -> torch.Tensor:
    """
    Convert an instance segmentation to a semantic segmentation

    Args:
        instance: instance segmentation of objects
            (objects need to start from 1, 0 background)
        cls: mapping from indices from instance segmentation to real classes.

    Returns:
        torch.Tensor: semantic segmentation

    Warnings:
        :attr:`instance` needs to encode objects starting from 1 and the
        indices need to be continuous (0 is interpreted as background)
    """
    seg = torch.zeros_like(instance)
    for idx, c in enumerate(cls, 1):
        seg[instance == idx] = c
    return seg


def pop_keys(data: dict, keys: Union[Callable, Sequence], return_popped=False) -> Union[dict, Tuple[dict, dict]]:
    """
    Pops keys from a given data dict

    Args:
        data: the dictionary to pop the keys from
        keys: if callable it must return a boolean for each key indicating
            whether it should be popped from the dict.
            if sequence of strings, the strings shall be the keys to be popped
        return_popped: whether to also return the popped values
        (default: False)

    Returns:
        dict: the data without the popped values
        dict: the popped values; only if :attr`return_popped` is True

    """
    if callable(keys):
        keys = [k for k in data.keys() if keys(k)]

    popped = {}

    for k in keys:
        popped[k] = data.pop(k)

    if return_popped:
        return data, popped
    else:
        return data


def filter_keys(data: dict, keys: Union[Callable, Sequence], return_popped=False) -> Union[dict, Tuple[dict, dict]]:
    """
    Filters keys from a given data dict

    Args:
        data: the dictionary to pop the keys from
        keys: if callable it must return a boolean for each key indicating
            whether it should be retained in the dict.
            if sequence of strings, the strings shall be the keys to be
            retained
        return_popped: whether to also return the popped values
            (default: False)

    Returns:
        dict: the data without the popped values
        dict: the popped values; only if :attr:`return_popped` is True

    """
    if callable(keys):
        keys = [k for k in data.keys() if keys(k)]

    keys_to_pop = [k for k in data.keys() if k not in keys]
    return pop_keys(data=data, keys=keys_to_pop, return_popped=return_popped)
