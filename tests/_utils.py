import numpy as np

from rising.loading.dataset import AsyncDataset


class LoadDummySample:
    def __init__(self, keys=("data", "label"), sizes=((3, 128, 128), (3,)), **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.sizes = sizes

    def __call__(self, path, *args, **kwargs):
        data = {_k: np.random.rand(*_s) for _k, _s in zip(self.keys, self.sizes)}
        data["id"] = f"sample{path}"
        return data


class DummyDataset(AsyncDataset):
    def __init__(self, num_samples=10, load_fn=LoadDummySample(), **load_kwargs):
        super().__init__(list(range(num_samples)), load_fn, **load_kwargs)
