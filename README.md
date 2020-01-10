# `rising`

![Project Status](https://img.shields.io/badge/status-alpha-red)
![Python](https://img.shields.io/badge/python-3.7-blue)
![PyPI](https://img.shields.io/pypi/v/rising)
[![Actions Status](https://github.com/PhoenixDL/rising/workflows/Unittests/badge.svg)](https://github.com/PhoenixDL/rising/actions)
[![codecov](https://codecov.io/gh/PhoenixDL/rising/branch/master/graph/badge.svg)](https://codecov.io/gh/PhoenixDL/rising)
![PyPI - License](https://img.shields.io/pypi/l/rising)
[![Chat](https://img.shields.io/badge/Slack-PhoenixDL-orange)](https://join.slack.com/t/phoenixdl/shared_invite/enQtODgwODI0MTE1MjgzLTJkZDE4N2NhM2VmNzVhYTEyMzI3NzFmMDY0NjM3MzJlZWRmMTk5ZWM1YzY2YjY5ZGQ1NWI1YmJmOTdiYTdhYTE)

### :warning: Current release disclaimer :warning:
This is an alpha release which is highly experimental. All transforms should be stable and tested but there might be some bugs.

## What is `rising`?
Rising is a high-performance data loading and augmentation library for 2D *and* 3D data completely written in PyTorch.
Our goal is to provide a seamless integration into the PyTorch Ecosystem without sacrificing usability or features.

## Why another framework?
|                      | Vanilla PyTorch | Albumentations | Batchgenerators | Kornia | DALI | `rising` |
|----------------------|-----------------|----------------|-----------------|--------|------|----------|
| 3D data augmentation | ❌              | ❌              | ✅              | ❌      | ❌   | ✅       |
| gradient propagation | ❌              | ❌              | ❌              | ✅      | ❌   | ✅       |
| augmentation on GPU  | ❌              | ❌              | ❌              | ✅      | ✅   | ✅       |

### What can I do with `rising`?
Rising currently consists out of two main modules:

#### `rising.loading`
Provides classes which can be used to load your data.
We provide some baseclasses like the `Cachedataset` or `Lazydataset` which can be easily used to load data either from the RAM or hard drive.
In contrast to the native PyTorch datasets you don't need to integrate your augmentation into your dataset. Hence, the only perpose of the dataset is to provide an interface to access individual data samples.
Our `DataLoader` is a direct subclass of the PyTorch dataloader and handles the batch assembly and applies the augmentation transformations to whole batches at once.
Additionally, there is a container class which can be used to hold multiple datasets at once and thus provides a interface for the entirety of your data (e.g. train/val split or kfold).

#### `rising.transforms`
This module implements many augmentation transformations which can be used during training.
All of them are implemented directly in PyTorch such that gradients can be propagated through them or augmentations could be computed live on the GPU.
Finally, all transforms are implemented for 2D and 3D data.

In the future, support for keypoints and bounding boxes will be added (bounding boxes are suported with an inefficient workaround with segmentations).

## Installation

Pypi Installation
```bash
pip install rising
```

Editable Installation for development

```bash
git clone git@github.com:PhoenixDL/rising.git
cd rising
pip install -e .
```

Running tests inside rising directory (top directory not the package directory)
```bash
python -m unittest
```

## Project Organization
`Issues`: If you find any bugs, want some additional features or maybe just a question don't hesitate to open an issue :)

`General Project Future`: Most of the features and the milestone organisation can be found inside the `projects` tab.
Features which are planned for the next release/milestone are listed unter `TODO Next Release` while features which are not scheduled yet are under `Todo`.