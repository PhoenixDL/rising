import torch


def orthogonal_inverse(mat: torch.Tensor) -> torch.Tensor:
    """
    Creates the inverse matrix of an orthogonal matrix
    Its basically just the transposed matrix :D
    Args:
        mat: Orthogonal matrix
    Return:
        Inverse of the input matrix
    """
    return torch.transpose(mat, -1, -2)
