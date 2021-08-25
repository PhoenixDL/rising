import torch

__all__ = ["local_pixel_shuffle", "random_inpainting", "random_outpainting"]


def local_pixel_shuffle(
    data: torch.Tensor, n: int = -1, block_size: tuple = (0, 0, 0), rel_block_size: float = 0.1
) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size()

    if n < 0:
        n = int(1000 * channels)  # changes ~ 12.5% of voxels
    for b in range(batch_size):
        for _ in range(n):
            c = torch.randint(0, channels - 1, (1,))

            (block_size_x, block_size_y, block_size_z) = (torch.tensor([size]) for size in block_size)

            if rel_block_size > 0:
                block_size_x = torch.randint(2, int(img_rows * rel_block_size), (1,))
                block_size_y = torch.randint(2, int(img_cols * rel_block_size), (1,))
                block_size_z = torch.randint(2, int(img_deps * rel_block_size), (1,))

            x = torch.randint(0, int(img_rows - block_size_x), (1,))
            y = torch.randint(0, int(img_cols - block_size_y), (1,))
            z = torch.randint(0, int(img_deps - block_size_z), (1,))

            window = data[
                b,
                c,
                x : x + block_size_x,
                y : y + block_size_y,
                z : z + block_size_z,
            ]
            idx = torch.randperm(window.numel())
            window = window.view(-1)[idx].view(window.size())

            data[b, c, x : x + block_size_x, y : y + block_size_y, z : z + block_size_z] = window

    return data


def random_inpainting(data: torch.Tensor, n: int = 5, maxv: float = 1.0, minv: float = 0.0) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size()

    while n > 0 and torch.rand((1)) < 0.95:
        for b in range(batch_size):
            block_size_x = torch.randint(img_rows // 10, img_rows // 4, (1,))
            block_size_y = torch.randint(img_rows // 10, img_rows // 4, (1,))
            block_size_z = torch.randint(img_rows // 10, img_rows // 4, (1,))
            x = torch.randint(3, int(img_rows - block_size_x - 3), (1,))
            y = torch.randint(3, int(img_cols - block_size_y - 3), (1,))
            z = torch.randint(3, int(img_deps - block_size_z - 3), (1,))

            block = torch.rand((1, channels, block_size_x, block_size_y, block_size_z)) * (maxv - minv) + minv

            data[b, :, x : x + block_size_x, y : y + block_size_y, z : z + block_size_z] = block

            n = n - 1

    return data


def random_outpainting(data: torch.Tensor, maxv: float = 1.0, minv: float = 0.0) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size()

    out = torch.rand(data.size()) * (maxv - minv) + minv

    block_size_x = torch.randint(5 * img_rows // 7, 6 * img_rows // 7, (1,))
    block_size_y = torch.randint(5 * img_cols // 7, 6 * img_cols // 7, (1,))
    block_size_z = torch.randint(5 * img_deps // 7, 6 * img_deps // 7, (1,))
    x = torch.randint(3, int(img_rows - block_size_x - 3), (1,))
    y = torch.randint(3, int(img_cols - block_size_y - 3), (1,))
    z = torch.randint(3, int(img_deps - block_size_z - 3), (1,))

    out[:, :, x : x + block_size_x, y : y + block_size_y, z : z + block_size_z] = data[
        :, :, x : x + block_size_x, y : y + block_size_y, z : z + block_size_z
    ]

    return out
