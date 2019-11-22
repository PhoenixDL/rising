from abc import ABC


class AbstractTransform(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        for key, item in kwargs.items():
            setattr(self, key, item)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
