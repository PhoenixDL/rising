import torch

from typing import Union, Sequence

from phdata.utils import check_scalar

# TODO: Gammatransform
# TODO: exponential transform ?
# TODO: FilterTransform (e.g. Gaussian Blur)


def norm_range(data: torch.Tensor, min: float, max: float,
               per_channel: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    """
    Scale range of tensor

    Parameters
    ----------
    data: torch.Tensor
        input data. Per channel option supports [C,H,W] and [C,H,W,D].
    min: float
        minimal value
    max: float
        maximal value
    per_channel: bool
        range is normalized per channel
    out: torch.Tensor
        if provided, result is saved into out

    Returns
    -------
    torch.Tensor
        normalized data
    """
    if out is None:
        out = torch.zeros_like(data)

    out = norm_min_max(data, per_channel=per_channel, out=out)
    _range = max - min
    out = (out * _range) + min
    return out


def norm_min_max(data: torch.Tensor, per_channel: bool = True,
                 out: torch.Tensor = None) -> torch.Tensor:
    """
    Scale range to [0,1]

    Parameters
    ----------
    data: torch.Tensor
        input data. Per channel option supports [C,H,W] and [C,H,W,D].
    per_channel: bool
        range is normalized per channel
    out: torch.Tensor
        if provided, result is saved into out

    Returns
    -------
    torch.Tensor
        scaled data
    """
    if out is None:
        out = torch.zeros_like(data)

    if per_channel:
        for _c in range(data.shape[0]):
            _min = data[_c].min()
            _range = data[_c].max() - _min
            out[_c] = (data[_c] - _min) / _range
    else:
        _min = data.min()
        _range = data.max() - _min
        out = (data - _min) / _range

    return out


def norm_zero_mean_unit_std(data: torch.Tensor, per_channel: bool = True,
                            out: torch.Tensor = None) -> torch.Tensor:
    """
    Normalize mean to zero and std to one

    Parameters
    ----------
    data: torch.Tensor
        input data. Per channel option supports [C,H,W] and [C,H,W,D].
    per_channel: bool
        range is normalized per channel
    out: torch.Tensor
        if provided, result is saved into out

    Returns
    -------
    torch.Tensor
        normalized data
    """
    if out is None:
        out = torch.zeros_like(data)

    if per_channel:
        for _c in range(data.shape[0]):
            out[_c] = (data[_c] - data[_c].mean()) / data[_c].std()
    else:
        out = (data - data.mean()) / data.std()

    return out


def norm_mean_std(data: torch.Tensor, mean: Union[float, Sequence], std: Union[float, Sequence],
                  per_channel: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    """
    Normalize mean and std with provided values

    Parameters
    ----------
    data: torch.Tensor
        input data. Per channel option supports [C,H,W] and [C,H,W,D].
    mean: float or Sequence
        used for mean normalization
    std: float or Sequence
        used for std normalization
    per_channel: bool
        range is normalized per channel
    out: torch.Tensor
        if provided, result is saved into out

    Returns
    -------
    torch.Tensor
        normalized data
    """
    if out is None:
        out = torch.zeros_like(data)

    if per_channel:
        if check_scalar(mean):
            mean = [mean] * data.shape[0]
        if check_scalar(std):
            std = [std] * data.shape[0]
        for _c in range(data.shape[0]):
            out[_c] = (data[_c] - mean[_c]) / std[_c]
    else:
        out = (data - mean) / std

    return out


def add_noise(data: torch.Tensor, noise_type: str, **kwargs) -> torch.Tensor:
    """
    Add noise to input

    Parameters
    ----------
    data: torch.Tensor
        input data
    noise_type: str
        supports all inplace functions of a pytorch tensor
    kwargs:
        keyword arguments passed to generating function

    Returns
    -------
    torch.Tensor
        data with added noise

    See Also
    --------
    :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
    """
    if not noise_type.endswith('_'):
        noise_type = noise_type + '_'
    noise_tensor = torch.empty_like(data, requires_grad=False)
    getattr(noise_tensor, noise_type)(**kwargs)
    return data + noise_tensor
