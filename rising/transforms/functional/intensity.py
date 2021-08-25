from typing import Optional, Sequence, Union

import torch

from rising.utils import check_scalar
from rising.utils.torchinterp1d import Interp1d

__all__ = [
    "norm_range",
    "norm_min_max",
    "norm_zero_mean_unit_std",
    "norm_mean_std",
    "add_noise",
    "add_value",
    "gamma_correction",
    "scale_by_value",
    "clamp",
    "bezier_3rd_order",
    "random_inversion",
]


def clamp(data: torch.Tensor, min: float, max: float, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Clamp tensor to minimal and maximal value

    Args:
        data: tensor to clamp
        min: lower limit
        max: upper limit
        out: output tensor

    Returns:
        Tensor: clamped tensor
    """
    return torch.clamp(data, min=float(min), max=float(max), out=out)


def norm_range(
    data: torch.Tensor, min: float, max: float, per_channel: bool = True, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scale range of tensor

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        min: minimal value
        max: maximal value
        per_channel: range is normalized per channel
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: normalized data
    """
    if out is None:
        out = torch.zeros_like(data)

    out = norm_min_max(data, per_channel=per_channel, out=out)
    _range = max - min
    out = (out * _range) + min
    return out


def norm_min_max(
    data: torch.Tensor, per_channel: bool = True, out: Optional[torch.Tensor] = None, eps: Optional[float] = 1e-8
) -> torch.Tensor:
    """
    Scale range to [0,1]

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        per_channel: range is normalized per channel
        out:  if provided, result is saved in here
        eps: small constant for numerical stability.
            If None, no factor constant will be added

    Returns:
        torch.Tensor: scaled data
    """

    def _norm(_data: torch.Tensor, _out: torch.Tensor):
        _min = _data.min()
        _range = _data.max() - _min
        if eps is not None:
            _range = _range + eps
        _out = (_data - _min) / _range
        return _out

    if out is None:
        out = torch.zeros_like(data)

    if per_channel:
        for _c in range(data.shape[0]):
            out[_c] = _norm(data[_c], out[_c])
    else:
        out = _norm(data, out)
    return out


def norm_zero_mean_unit_std(
    data: torch.Tensor, per_channel: bool = True, out: Optional[torch.Tensor] = None, eps: Optional[float] = 1e-8
) -> torch.Tensor:
    """
    Normalize mean to zero and std to one

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        per_channel: range is normalized per channel
        out: if provided, result is saved in here
        eps: small constant for numerical stability.
            If None, no factor constant will be added

    Returns:
        torch.Tensor: normalized data
    """

    def _norm(_data: torch.Tensor, _out: torch.Tensor):
        denom = _data.std()
        if eps is not None:
            denom = denom + eps
        _out = (_data - _data.mean()) / denom
        return _out

    if out is None:
        out = torch.zeros_like(data)

    if per_channel:
        for _c in range(data.shape[0]):
            out[_c] = _norm(data[_c], out[_c])
    else:
        out = _norm(data, out)
    return out


def norm_mean_std(
    data: torch.Tensor,
    mean: Union[float, Sequence],
    std: Union[float, Sequence],
    per_channel: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Normalize mean and std with provided values

    Args:
        data:input data. Per channel option supports [C,H,W] and [C,H,W,D].
        mean: used for mean normalization
        std: used for std normalization
        per_channel: range is normalized per channel
        out: if provided, result is saved into out

    Returns:
        torch.Tensor: normalized data
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


def add_noise(data: torch.Tensor, noise_type: str, out: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
    """
    Add noise to input

    Args:
        data: input data
        noise_type: supports all inplace functions of a pytorch tensor
        out: if provided, result is saved in here
        kwargs: keyword arguments passed to generating function

    Returns:
        torch.Tensor: data with added noise

    See Also:
        :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
    """
    if not noise_type.endswith("_"):
        noise_type = noise_type + "_"
    noise_tensor = torch.empty_like(data, requires_grad=False)
    getattr(noise_tensor, noise_type)(**kwargs)
    return torch.add(data, noise_tensor, out=out)


def gamma_correction(data: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Apply gamma correction to data
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        gamma: gamma for correction

    Returns:
        torch.Tensor: gamma corrected data
    """
    if torch.is_tensor(gamma):
        gamma = gamma.to(data)
    return data.pow(gamma)


def add_value(data: torch.Tensor, value: float, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Increase brightness additively by value
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        value: additive value
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: augmented data
    """
    return torch.add(data, value, out=out)


def scale_by_value(data: torch.Tensor, value: float, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Increase brightness scaled by value
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        value: scaling value
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: augmented data
    """
    return torch.mul(data, value, out=out)


def bezier_3rd_order(
    data: torch.Tensor, maxv: float = 1.0, minv: float = 0.0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    p0 = torch.zeros((1, 2))
    p1 = torch.rand((1, 2))
    p2 = torch.rand((1, 2))
    p3 = torch.ones((1, 2))

    t = torch.linspace(0.0, 1.0, 1000).unsqueeze(1)

    points = (1 - t * t * t) * p0 + 3 * (1 - t) * (1 - t) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t * p3

    # scaling according to maxv,minv
    points = points * (maxv - minv) + minv

    xvals = points[:, 0]
    yvals = points[:, 1]

    out_flat = Interp1d.apply(xvals, yvals, data.view(-1))

    return out_flat.view(data.shape)


def random_inversion(
    data: torch.Tensor,
    prob_inversion: float = 0.5,
    maxv: float = 1.0,
    minv: float = 0.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if torch.rand((1)) < prob_inversion:
        # Inversion of curve
        out = maxv + minv - data
    else:
        # do nothing
        out = data

    return out
