import unittest
from unittest.mock import Mock, call

import torch

from rising.transforms import AbstractTransform, BaseTransform, PerChannelTransform, PerSampleTransform


class AddTransform(AbstractTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_tensor = torch.rand(1, 1, 32, 32, requires_grad=True)

    def forward(self, **data) -> dict:
        data["data"] = data["data"] + self.grad_tensor
        return data


def sum_dim(data, dims, **kwargs):
    if dims:
        dims = [d + 2 for d in dims]
        return torch.sum(data, dims, **kwargs)
    else:
        return data


class TestAbstractTransform(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch_dict = {"data": torch.rand(1, 1, 32, 32), "seg": torch.rand(1, 1, 32, 32), "label": torch.arange(3)}

    def test_abstract_transform(self):
        trafo = AbstractTransform(grad=False, internal0=True)
        self.assertTrue(trafo.internal0)
        with self.assertRaises(NotImplementedError):
            trafo(**self.batch_dict)

    def test_abstract_transform_no_grad(self):
        trafo = AddTransform(grad=False)
        output = trafo(**self.batch_dict)

        self.assertIsNone(output["data"]._grad_fn)
        with self.assertRaises(RuntimeError):
            output["data"].mean().backward()

    def test_abstract_transform_grad(self):
        trafo = AddTransform(grad=True)
        output = trafo(**self.batch_dict)
        self.assertIsNotNone(output["data"]._grad_fn)

        output["data"].mean().backward()
        self.assertIsNotNone(trafo.grad_tensor._grad)

    def test_base_transform(self):
        trafo = BaseTransform(lambda x: x + 50, keys=("data", "seg"))
        output = trafo(**self.batch_dict)

        diff_data = (output["data"] - self.batch_dict["data"]).mean().item()
        diff_seg = (output["seg"] - self.batch_dict["seg"]).mean().item()
        diff_label = (output["label"] - self.batch_dict["label"]).float().mean().item()

        self.assertEqual(diff_data, 50)
        self.assertEqual(diff_seg, 50)
        self.assertEqual(diff_label, 0)

    def test_per_sample_transform(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerSampleTransform(augment_fn, keys=("label",))
        output = trafo(**self.batch_dict)
        calls = [
            call(torch.tensor([0])),
            call(torch.tensor([1])),
            call(torch.tensor([2])),
        ]
        mock.assert_has_calls(calls)

    def test_per_channel_transform_per_channel_true(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerChannelTransform(augment_fn, per_channel=True, keys=("label",))
        self.batch_dict["label"] = self.batch_dict["label"][None]
        output = trafo(**self.batch_dict)
        calls = [
            call(torch.tensor([0])),
            call(torch.tensor([1])),
            call(torch.tensor([2])),
        ]
        mock.assert_has_calls(calls)

    def test_per_channel_transform_per_channel_false(self):
        mock = Mock(return_value=0)

        def augment_fn(inp, *args, **kwargs):
            return mock(inp)

        trafo = PerChannelTransform(augment_fn, per_channel=False, keys=("label",))
        self.batch_dict["label"] = self.batch_dict["label"][None]
        output = trafo(**self.batch_dict)
        mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
