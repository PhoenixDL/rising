<div align="center">
# PhoenixRising
</div>

### :warning: Current release disclaimer :warning:
This is an alpha release which is highly experimental. All transforms should be stable and tested but there might be some bugs.

## What is rising?
Rising is an augmentation library for 2D and 3D data completely written in pytorch.
Our goal is to provide a seamless integration into the PyTorch Ecosystem without sacrificing usability or features.

### Why pytorch?
PyTorch is probably the [most used deep learning framework in the research area](https://chillee.github.io/pytorch-vs-tensorflow/) und provides a great API to do all crazy stuff you can think of.
Furthermore, the core development team of `Rising` is using pytorch :sunglasses:

### What can I do with `Rising`?
Rising currently consists out of two main modules:

`Loading`: Provides a lot classes which can be used to load your data.
We provide some baseclasses like the `Cachedataset` or `Lazydataset` which can be easily used to load data into a uniform form.
In contrast to the native PyTorch datasets you don't need to integrate your augmentation into your dataset -> the only perpose of the dataset is to provide an interface to access individual data samples.
Our `DataLoader` is a direct subclass of the PyTorch dataloader and handles the batch assembly and applies the augmentation transformations to whole batches at once.
Additionally, there exists a container class which can be used to hold multiple datasets at once and thus provides a interface for the entirety of your data (e.g. train/val split or kfold).

`Transforms`: This module implements many augmentation transformations which can be used during training.
All of them are implemented directly in pytorch so gradient can be propagated through them or augmentations could be computed live on the GPU.
Finally, all transforms are implemented for 2D/3D data to fully support most the available data.

In the future support for keypoints and bounding boxes will be added (bounding boxes are suported with an inefficient workaround with segmentations).

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