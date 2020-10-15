import torch

__all__ = ["local_pixel_shuffle", "random_inpainting", "random_outpainting"]

# added +1 to most of the upper bounds in torch.randint as they are NOT included in the range of output
def local_pixel_shuffle(data: torch.Tensor, n: int = -1, block_size: tuple=(0,0,0), rel_block_size: float = 0.1) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size()

    if n < 0:
        n = int(1000*channels) # changes ~ 12.5% of voxels
    for b in range(batch_size):
        for _ in range(n):
            c = torch.randint(0,channels-1, (1,))
            
            (block_size_x, block_size_y, block_size_z)  = (torch.tensor([size]) for size in block_size)
            
            if rel_block_size > 0:
                block_size_x = torch.randint(2, max(2, int(img_rows * rel_block_size)) + 1, (1,))
                block_size_y = torch.randint(2, max(2, int(img_cols * rel_block_size)) + 1, (1,))
                block_size_z = torch.randint(2, max(2, int(img_deps * rel_block_size)) + 1, (1,))

            x = torch.randint(0, int(img_rows - block_size_x), (1,))
            y = torch.randint(0, int(img_cols - block_size_y), (1,))
            z = torch.randint(0, int(img_deps - block_size_z), (1,))

            window = data[b, c, x:x + block_size_x,
                            y:y + block_size_y,
                            z:z + block_size_z,
                        ]
            idx = torch.randperm(window.numel())
            window = window.view(-1)[idx].view(window.size())

            data[b, c, x:x + block_size_x,
                          y:y + block_size_y,
                          z:z + block_size_z] = window

    return data


def random_inpainting(data: torch.Tensor, n: int = 5, maxv: float = 1.0, minv: float = 0.0, max_size: tuple = (0,0,0), min_size: tuple = (0,0,0), rel_max_size: tuple = (0.25, 0.25, 0.25), rel_min_size: tuple = (0.1, 0.1, 0.1), min_border_distance: tuple = (3, 3, 3)) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size() # N,C,Z,X,Y
    
    if all((rel_max >= rel_min > 0 for rel_min, rel_max in zip(rel_min_size, rel_max_size))):
        min_x = int(rel_min_size[0] * img_rows) 
        max_x = min(img_rows - 2 * min_border_distance[0] - 1, int(rel_max_size[0] * img_rows))
        min_y = int(rel_min_size[1] * img_cols)
        max_y = min(img_cols - 2 * min_border_distance[1] - 1, int(rel_max_size[1] * img_cols))
        min_z = int(rel_min_size[2] * img_deps) 
        max_z = min(img_deps - 2 * min_border_distance[2] - 1, int(rel_max_size[2] * img_deps))
    elif all((max >= min > 0 for min, max in zip(min_size, max_size))):
        min_x, max_x = min_size[0], max_size[0]
        min_y, max_y = min_size[1], max_size[1]
        min_z, max_z = min_size[2], max_size[2]
    else:
        raise ValueError(f'random_inpainting was called with neither a valid absolut nor a valid relative min/max patch size combination. Received absolut min_size {min_size}, max_size {max_size}, and relative rel_min_size {rel_min_size}, rel_max_size {rel_max_size}')

    while n > 0 and torch.rand((1)) < 0.95:
        for b in range(batch_size):
            block_size_x = torch.randint(min_x, max_x +1, (1,)) 
            block_size_y = torch.randint(min_y, max_y +1, (1,)) 
            block_size_z = torch.randint(min_z, max_z +1, (1,)) 
            x = torch.randint(min_border_distance[0], int(img_rows - block_size_x - min_border_distance[0]), (1,))
            y = torch.randint(min_border_distance[1], int(img_cols - block_size_y - min_border_distance[1]), (1,))
            z = torch.randint(min_border_distance[2], int(img_deps - block_size_z - min_border_distance[2]), (1,))

            block = torch.rand((1, channels, block_size_x, block_size_y, block_size_z)) \
                    * (maxv-minv) + minv

            data[b, :, x:x + block_size_x,
            y:y + block_size_y,
            z:z + block_size_z] = block

            n = n - 1

    return data


def random_outpainting(data: torch.Tensor, maxv: float = 1.0, minv: float = 0.0, max_size: tuple = (0,0,0), min_size: tuple = (0,0,0), rel_max_size: tuple = (6/7, 6/7, 6/7), rel_min_size: tuple = (5/7, 5/7, 5/7), min_border_distance= (3,3,3)) -> torch.Tensor:

    batch_size, channels, img_rows, img_cols, img_deps = data.size()

    if all((rel_max >= rel_min > 0 for rel_min, rel_max in zip(rel_min_size, rel_max_size))):
        min_x = int(rel_min_size[0] * img_rows) 
        # min() is necessary to have guarantee y > x for torch.randint(x,y) calls
        # lowest possible index for block start is min_border_distance[i], highest possible is img_rows - block_size - min_border_distance[i]. -> block_size < img_rows - 2 * min_border_distance
        max_x = min(img_rows - 2 * min_border_distance[0] - 1, int(rel_max_size[0] * img_rows))
        min_y = int(rel_min_size[1] * img_cols)
        max_y = min(img_cols - 2 * min_border_distance[1] - 1, int(rel_max_size[1] * img_cols))
        min_z = int(rel_min_size[2] * img_deps) 
        max_z = min(img_deps - 2 * min_border_distance[2] - 1, int(rel_max_size[2] * img_deps))

    elif all((max >= min > 0 for min, max in zip(min_size, max_size))):
        min_x, max_x = min_size[0], max_size[0]
        min_y, max_y = min_size[1], max_size[1]
        min_z, max_z = min_size[2], max_size[2]
    else:
        raise ValueError(f'random_inpainting was called with neither a valid absolut nor a valid relative min/max patch size combination. Received absolut min_size {min_size}, max_size {max_size}, and relative rel_min_size {rel_min_size}, rel_max_size {rel_max_size}')

    out = torch.rand(data.size()) * (maxv - minv) + minv

    block_size_x = torch.randint(min_x, max_x + 1, (1,)) 
    block_size_y = torch.randint(min_y, max_y + 1, (1,)) 
    block_size_z = torch.randint(min_z, max_z + 1, (1,)) 
    x = torch.randint(min_border_distance[0], int(img_rows - block_size_x - min_border_distance[0]), (1,))
    y = torch.randint(min_border_distance[1], int(img_cols - block_size_y - min_border_distance[1]), (1,))
    z = torch.randint(min_border_distance[2], int(img_deps - block_size_z - min_border_distance[2]), (1,))

    out[:, :, x:x + block_size_x,
              y:y + block_size_y,
              z:z + block_size_z] = data[:, :, x:x + block_size_x,
                                               y:y + block_size_y,
                                               z:z + block_size_z]

    return out