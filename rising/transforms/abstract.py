from typing import Any, Callable, Sequence, Tuple, Union

import torch

from rising.random import AbstractParameter, DiscreteParameter

__all__ = ["AbstractTransform", "BaseTransform", "PerSampleTransform", "PerChannelTransform", "BaseTransformSeeded"]

augment_callable = Callable[[torch.Tensor], Any]
augment_axis_callable = Callable[[torch.Tensor, Union[float, Sequence]], Any]


class AbstractTransform(torch.nn.Module):
    """Base class for all transforms"""

    def __init__(self, grad: bool = False, **kwargs):
        """
        Args:
            grad: enable gradient computation inside transformation
        """
        super().__init__()
        self.grad = grad
        self._registered_samplers = []
        for key, item in kwargs.items():
            setattr(self, key, item)

    def register_sampler(self, name: str, sampler: Union[Sequence, AbstractParameter], *args, **kwargs):
        """
        Registers a parameter sampler to the transform.
        Internally a property is created to forward calls to the attribute to
        calls of the sampler.

        Args:
            name : the property name
            sampler : the sampler. Will be wrapped to a sampler always returning
                the same element if not already a sampler
            *args : additional positional arguments (will be forwarded to
                sampler call)
            **kwargs : additional keyword arguments (will be forwarded to
                sampler call)
        """
        self._registered_samplers.append(name)
        if hasattr(self, name):
            raise NameError("Name %s already exists" % name)
        if not isinstance(sampler, (tuple, list)):
            sampler = [sampler]

        new_sampler = []
        for _sampler in sampler:
            if not isinstance(_sampler, AbstractParameter):
                _sampler = DiscreteParameter([_sampler], replacement=True)
            new_sampler.append(_sampler)
        sampler = new_sampler

        def sample(self):
            """
            Sample random values
            """
            sample_result = tuple([_sampler(*args, **kwargs) for _sampler in sampler])

            if len(sample_result) == 1:
                return sample_result[0]
            else:
                return sample_result

        setattr(self, name, property(sample))

    def __getattribute__(self, item) -> Any:
        """
        Automatically dereference registered samplers

        Args:
            item: name of attribute

        Returns:
            Any: attribute
        """
        res = super().__getattribute__(item)
        if isinstance(res, property) and item in self._registered_samplers:
            # by first checking the type we reduce the lookup
            # time for all non property objects
            return res.__get__(self)
        else:
            return res

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call super class with correct torch context

        Args:
            *args: forwarded positional arguments
            **kwargs: forwarded keyword arguments

        Returns:
            Any: transformed data

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        with context:
            return super().__call__(*args, **kwargs)

    def forward(self, **data) -> dict:
        """
        Implement transform functionality here

        Args:
            **data: dict with data

        Returns:
            dict: dict with transformed data
        """
        raise NotImplementedError


class BaseTransform(AbstractTransform):
    """
    Transform to apply a functional interface to given keys

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per key.
    """

    def __init__(
        self,
        augment_fn: augment_callable,
        *args,
        keys: Sequence = ("data",),
        grad: bool = False,
        property_names: Sequence[str] = (),
        **kwargs
    ):
        """
        Args:
            augment_fn: function for augmentation
            *args: positional arguments passed to augment_fn
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            property_names: a tuple containing all the properties to call
                during forward pass
            **kwargs: keyword arguments passed to augment_fn
        """
        sampler_vals = [kwargs.pop(name) for name in property_names]
        super().__init__(grad=grad, **kwargs)
        self.augment_fn = augment_fn
        self.keys = keys
        self.property_names = property_names
        self.args = args
        self.kwargs = kwargs
        for name, val in zip(property_names, sampler_vals):
            self.register_sampler(name, val)

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)

        kwargs.update(self.kwargs)

        for _key in self.keys:
            data[_key] = self.augment_fn(data[_key], *self.args, **kwargs)
        return data


class BaseTransformSeeded(BaseTransform):
    """
    Transform to apply a functional interface to given keys and use the same
    pytorch(!) seed for every key.
    """

    def forward(self, **data) -> dict:
        """
        Apply transformation and use same seed for every key

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)

        kwargs.update(self.kwargs)

        seed = torch.random.get_rng_state()
        for _key in self.keys:
            torch.random.set_rng_state(seed)
            data[_key] = self.augment_fn(data[_key], *self.args, **kwargs)
        return data


class PerSampleTransform(BaseTransform):
    """
    Apply transformation to each sample in batch individually
    :attr:`augment_fn` must be callable with option :attr:`out`
    where results are saved in.

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per sample and key.
    """

    def forward(self, **data) -> dict:
        """
        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)

        kwargs.update(self.kwargs)
        for _key in self.keys:
            out = torch.empty_like(data[_key])
            for _i in range(data[_key].shape[0]):
                out[_i] = self.augment_fn(data[_key][_i], out=out[_i], **kwargs)
            data[_key] = out
        return data


class PerChannelTransform(BaseTransform):
    """
    Apply transformation per channel (but still to whole batch)

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per channel and key.
    """

    def __init__(
        self,
        augment_fn: augment_callable,
        per_channel: bool = False,
        keys: Sequence = ("data",),
        grad: bool = False,
        property_names: Tuple[str] = (),
        **kwargs
    ):
        """
        Args:
            augment_fn: function for augmentation
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=augment_fn, keys=keys, grad=grad, property_names=property_names, **kwargs)
        self.per_channel = per_channel

    def forward(self, **data) -> dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if self.per_channel:
            kwargs = {}
            for k in self.property_names:
                kwargs[k] = getattr(self, k)

            kwargs.update(self.kwargs)
            for _key in self.keys:
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    out[:, _i] = self.augment_fn(data[_key][:, _i], out=out[:, _i], **kwargs)
                data[_key] = out
            return data
        else:
            return super().forward(**data)
