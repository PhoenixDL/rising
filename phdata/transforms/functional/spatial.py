import torch


def mirror(data, axis):
    if axis == 0:
        data = data[:, :, ::-1]
    if axis == 1:
        data = data[:, :, :, ::-1]
    if axis == 2:
        data = data[:, :, :, :, ::-1]
    return data


def rot90(data, k, dims):
    dims = [d + 2 for d in dims]
    return torch.rot90(data, k, dims)


def rot():
    pass
