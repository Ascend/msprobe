import pytest
from unittest.mock import patch, MagicMock
from mindspore import nn, context
from mindspore.common.initializer import Normal
import mindspore as ms

from msprobe.mindspore.monitor.common_func import (
    is_valid_instance,
    get_submodules,
    get_parameters,
    get_rank,
    comm_is_initialized,
)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestModelUtils:
    @classmethod
    def setup_class(cls):
        """Setup once for all tests in this class"""
        cls.ms_model = MSModel()
        if TORCH_AVAILABLE:
            cls.torch_model = TorchModel()

    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests in this class"""
        pass


    def test_is_valid_instance_if_model_is_cell_or_module_then_return_true(self):
        with patch('msprobe.mindspore.monitor.common_func.is_mindtorch') as mock_is_mindtorch, \
             patch('msprobe.mindspore.monitor.common_func.torch', torch, create=True):
            if TORCH_AVAILABLE:
                mock_is_mindtorch.return_value = True
                assert is_valid_instance(self.torch_model)
            mock_is_mindtorch.return_value = False
            assert is_valid_instance(self.ms_model)

    def test_is_valid_instance_if_input_is_string_then_return_false(self):
        assert not is_valid_instance("not a model")

    def test_is_valid_instance_if_input_is_number_then_return_false(self):
        assert not is_valid_instance(123)

    def test_get_submodules_if_model_is_valid_then_return_non_empty_dict(self):
        with patch('msprobe.mindspore.monitor.common_func.is_mindtorch') as mock_is_mindtorch, \
             patch('msprobe.mindspore.monitor.common_func.torch', torch, create=True):
            mock_is_mindtorch.return_value = True
            if TORCH_AVAILABLE:
                submodules = dict(get_submodules(self.torch_model))
                assert len(submodules) > 0
                assert any(name == 'conv1' for name in submodules)

            mock_is_mindtorch.return_value = False
            submodules = dict(get_submodules(self.ms_model))
            assert len(submodules) > 0
            assert any(name.endswith('conv1') for name in submodules)


    def test_get_submodules_if_model_is_invalid_then_return_empty_dict(self):
        assert get_submodules("invalid") == {}

    def test_get_parameters_if_model_is_valid_then_return_non_empty_dict(self):
        with patch('msprobe.mindspore.monitor.common_func.is_mindtorch') as mock_is_mindtorch, \
             patch('msprobe.mindspore.monitor.common_func.torch', torch, create=True):
            mock_is_mindtorch.return_value = True
            if TORCH_AVAILABLE:
                params = dict(get_parameters(self.torch_model))
                assert any(name == 'conv1.weight' for name in params)
            mock_is_mindtorch.return_value = False
            params = dict(get_parameters(self.ms_model))
            assert any('conv1.weight' in name for name in params)


    def test_get_parameters_if_model_is_invalid_then_return_empty_dict(self):
        assert get_parameters(123) == {}

    def test_get_rank_if_comm_initialized_then_return_integer(self):
        rank = get_rank()
        assert isinstance(rank, int)
        assert rank >= 0

    def test_comm_is_initialized_when_called_then_return_boolean(self):
        assert isinstance(comm_is_initialized(), bool)


# Test models
class MSModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, has_bias=True, weight_init=Normal(0.02))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

if TORCH_AVAILABLE:
    class TorchModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch_nn.Conv2d(3, 64, 3)
            self.bn1 = torch_nn.BatchNorm2d(64)
            self.relu = torch_nn.ReLU()
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x