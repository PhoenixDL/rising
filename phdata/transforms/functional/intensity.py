from phdata.utils import check_scalar

# TODO: Gammatransform
# TODO: exponential transform ?


def norm_range(data, min, max, per_channel=True):
    data = norm_min_max(data, per_channel=per_channel)
    _range = max - min
    return (data * _range) + min


def norm_min_max(data, per_channel=True):
    if per_channel:
        for _c in range(1, data.ndim):
            _min = data[_c].min()
            _range = data[_c].max() - _min
            data[_c] = (data[_c] - _min) / _range
        return data
    else:
        _min = data.min()
        _range = data.max() - _min
        return (data - _min) / _range


def norm_zero_mean_unit_std(data, per_channel=True):
    if per_channel:
        for _c in range(1, data.ndim):
            data[_c] = (data[_c] - data[_c].min()) / data[_c].std()
        return data
    else:
        return (data - data.min()) / data.std()


def norm_mean_std(data, mean, std, per_channel=True):
    if per_channel:
        if check_scalar(mean):
            mean = [mean] * data.shape[0]
        if check_scalar(std):
            std = [std] * data.shape[0]
        for _c in range(1, data.ndim):
            data[_c] = (data[_c] - mean[_c]) / std[_c]
        return data
    else:
        return (data - data.mean()) / data.std()
