"""
Provides a functional interface for transforms
(usually working on single tensors rather then collections thereof).
All transformations are implemented to work on batched tensors.
Implementations include:

* Affine Transforms
* Channel Transforms
* Cropping Transforms
* Device Transforms
* Intensity Transforms
* Spatial Transforms
* Tensor Transforms
* Utility Transforms
"""

from rising.transforms.functional.channel import one_hot_batch
from rising.transforms.functional.crop import center_crop, crop, random, random_crop
from rising.transforms.functional.intensity import (
    add_noise,
    add_value,
    bezier_3rd_order,
    clamp,
    gamma_correction,
    norm_mean_std,
    norm_min_max,
    norm_range,
    norm_zero_mean_unit_std,
    random_inversion,
    scale_by_value,
)
from rising.transforms.functional.painting import local_pixel_shuffle, random_inpainting, random_outpainting
from rising.transforms.functional.spatial import mirror, resize_native, rot90
from rising.transforms.functional.tensor import tensor_op, to_device_dtype
from rising.transforms.functional.utility import box_to_seg, filter_keys, instance_to_semantic, pop_keys, seg_to_box
