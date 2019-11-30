import torch


def points_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    return torch.cat([batch,
                      batch.new_ones((*batch.size()[:-1], 1))],
                     dim=-1)


def matrix_to_homogeneous(batch: torch.Tensor) -> torch.Tensor:
    missing = torch.zeros((batch.size(0),
                           *[1 for tmp in batch.shape[1:-1]],
                           batch.size(-1)),
                          device=batch.device, dtype=batch.dtype)

    missing[..., -1] = 1

    return torch.cat([batch, missing], dim=1)


def to_cartesian(batch: torch.Tensor) -> torch.Tensor:
    return batch[:, :-1, ...]


def matrix_permute_coordinate_order(batch: torch.Tensor) -> torch.Tensor:
    return batch[:, ::-1, ::-1]
