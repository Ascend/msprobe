import os
import random
import shutil
import unittest
import torch
import json
import numpy as np
from packaging.version import parse as version_parse
import torch.nn as nn
import mindspore as ms
import mindspore.nn as ms_nn
from mindspore import Tensor
from msprobe.core.config_check.config_checker import ConfigChecker
from msprobe.core.config_check.checkers.pip_checker import PipPackageChecker
from msprobe.core.config_check.checkers.random_checker import RandomChecker
from msprobe.core.config_check.checkers.dataset_checker import DatasetChecker
from msprobe.core.config_check.checkers.weights_checker import WeightsChecker
from msprobe.core.common.file_utils import read_xlsx
from msprobe.core.common.framework_adapter import FmkAdp


testdir = os.path.dirname(__file__)
config_checking_dir = os.path.dirname(testdir)
temp_dir = os.path.join(config_checking_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)
ms.set_context(device_target="CPU")


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    ms.set_seed(seed)


class MockPyTorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


class MockMindSporeModule(ms_nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear = ms_nn.Dense(10, 5)
        self.relu = ms_nn.ReLU()

    def construct(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


def get_test_dataset():
    inputs = [torch.rand(10, 10) for _ in range(10)]
    labels = [torch.randint(0, 5, (10,)) for _ in range(10)]
    ms_inputs = [Tensor(input.numpy()) for input in inputs]
    ms_labels = [Tensor(label.numpy()) for label in labels]
    return zip(inputs, labels), zip(ms_inputs, ms_labels)


def get_test_model(use_pytorch=True):
    if use_pytorch:
        test_module = MockPyTorchModule()
        nn.init.constant_(test_module.linear.weight, 1.0)
        nn.init.constant_(test_module.linear.bias, 1.0)
        return test_module
    else:
        test_module = MockMindSporeModule()
        for param in test_module.get_parameters():
            param.set_data(ms.Tensor(np.ones(param.data.shape), dtype=param.data.dtype))
        return test_module


@unittest.mock.patch("msprobe.core.config_check.checkers.pip_checker.collect_pip_data")
@unittest.mock.patch("msprobe.core.config_check.checkers.env_args_checker.collect_env_data")
def train_test(seed, output_zip_path, shell_path, mock_env, mock_pip):
    if seed == 1234:
        mock_pip.return_value = "transformers=0.0.1"
        mock_env.return_value = {"NCCL_DETERMINISTIC": True}
    else:
        mock_pip.return_value = "transformers=0.0.2"
        mock_env.return_value = {"HCCL_DETERMINISTIC": False, "ASCEND_LAUNCH_BLOCKING": 1}
    seed_all(seed)

    use_pytorch = seed == 1234
    test_dataset, ms_test_dataset = get_test_dataset()
    test_module = get_test_model(use_pytorch)

    if use_pytorch:
        loss_fun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(test_module.parameters(), lr=1e-2, foreach=False)
        ConfigChecker(test_module, shell_path, output_zip_path)

        for input_data, label in test_dataset:
            output = test_module(input_data, y=input_data)
            loss = loss_fun(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        loss_fun = ms_nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = ms_nn.SGD(test_module.trainable_params(), learning_rate=1e-2)
        train_network = ms_nn.TrainOneStepCell(ms_nn.WithLossCell(test_module, loss_fun), optimizer)
        ConfigChecker(test_module, shell_path, output_zip_path, fmk="mindspore")
        
        for input_data, label in ms_test_dataset:
            loss = train_network(input_data, label)



class TestConfigChecker(unittest.TestCase):
    def tearDown(self):
        FmkAdp.set_fmk("pytorch")
        shutil.rmtree(temp_dir)

    @unittest.skipIf(version_parse(np.__version__) >= version_parse("2.0.0"),
                     "mindspore is not compatible with numpy >= 2.0")
    def test_all(self):
        train_test(1234, os.path.join(temp_dir, "config_check_pack1.zip"), [os.path.join(testdir, "cmp.sh")])

        ConfigChecker.pre_forward_fun_list = []
        ConfigChecker.step = 0
        RandomChecker.write_once = False
        ConfigChecker.apply_patches("pytorch")
        ConfigChecker.apply_patches("mindspore")

        train_test(1233, os.path.join(temp_dir, "config_check_pack2.zip"), [os.path.join(testdir, "bench.sh")])

        ConfigChecker.compare(os.path.join(temp_dir, "config_check_pack1.zip"),
                              os.path.join(temp_dir, "config_check_pack2.zip"),
                              os.path.join(temp_dir, "compare_output"))

        compare_output_dir = os.path.join(temp_dir, "compare_output")

        total_check_result = read_xlsx(os.path.join(compare_output_dir, ConfigChecker.result_filename))
        self.assertEqual(total_check_result.columns.tolist(), ConfigChecker.result_header)
        target_total_check_result = [
            ['env', "error"],
            ['pip', "error"],
            ['dataset', "error"],
            ['weights', "error"],
            ['hyperparameters', "error"],
            ['random', "error"]
        ]
        self.assertEqual(total_check_result.values.tolist(), target_total_check_result)

        pip_data_check_result = read_xlsx(os.path.join(compare_output_dir, ConfigChecker.result_filename),
                                          sheet_name=PipPackageChecker.target_name_in_zip)
        self.assertEqual(pip_data_check_result.columns.tolist(), PipPackageChecker.result_header)
        self.assertEqual(pip_data_check_result.iloc[0].tolist(), ['transformers', '0.0.1', '0.0.2', 'error'])

        random_check_result = read_xlsx(os.path.join(compare_output_dir, ConfigChecker.result_filename),
                                        sheet_name=RandomChecker.target_name_in_zip)
        self.assertEqual(random_check_result.columns.tolist(), RandomChecker.result_header)
        self.assertEqual(len(random_check_result), 7)

        dataset_check_result = read_xlsx(os.path.join(compare_output_dir, ConfigChecker.result_filename),
                                         sheet_name=DatasetChecker.target_name_in_zip)
        self.assertEqual(dataset_check_result.columns.tolist(), DatasetChecker.result_header)
        self.assertEqual(len(dataset_check_result), 20)

        weight_check_result = read_xlsx(os.path.join(compare_output_dir, ConfigChecker.result_filename),
                                        sheet_name=WeightsChecker.target_name_in_zip)
        self.assertEqual(weight_check_result.columns.tolist(), WeightsChecker.result_header)
        self.assertEqual(len(weight_check_result), 20)
