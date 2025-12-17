# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import sys
import unittest
from typing import Set, List

import libcst

sys.path.append(os.path.abspath("../../../../"))
sys.path.append(os.path.abspath("../../../../src/ms_fmk_transplt"))

from analysis.unsupported_api_analysis.unsupported_api_visitor import ApiInstance
from analysis.precision_performance_advice_analysis.precision_performance_advice_visitor import \
    (analyse_precision_performance_advice_api, generate_perf_suggest)
from analysis.precision_performance_advice_analysis.prec_perf_utils import PerfApiSuggest
from analysis.precision_performance_advice_analysis.prec_perf_utils import AdviceInfo
from analysis.unsupported_api_analysis.unsupported_api_visitor import ApiInstance
from utils import trans_utils as utils


class TestGeneratePerfSuggest(unittest.TestCase):
    def setUp(self):
        perf_suggest = {
            "mock_api": {
                "dependency": ["mock_api_dept1", "mock_api_dept2"],
                "msg": "mock_api_dept is used, it is recommended to use mock_api()"
            }
        }
        self.perf_inst = PerfApiSuggest(perf_suggest)

    def test_generate_perf_suggest_give_suggest(self):
        self._call_mock_api("mock_api_dept1")
        self._call_mock_api("mock_api_dept2")
        suggest_list = generate_perf_suggest(self.perf_inst)
        assert len(suggest_list) > 0

    def test_generate_perf_suggest_no_suggest1(self):
        self._call_mock_api("mock_api_dept1")
        suggest_list = generate_perf_suggest(self.perf_inst)
        assert len(suggest_list) == 0

    def test_generate_perf_suggest_no_suggest2(self):
        self._call_mock_api("mock_api_dept1")
        self._call_mock_api("mock_api_dept2")
        self._call_mock_api("mock_api")
        suggest_list = generate_perf_suggest(self.perf_inst)
        assert len(suggest_list) == 0
    
    def _call_mock_api(self, full_name: str):
        if full_name in self.perf_inst.dependency:
            self.perf_inst.dependency[full_name] = True
        if full_name in self.perf_inst.suggest_apis:
            self.perf_inst.suggest_apis[full_name] = True


class TestPrecisionPerformanceAdviceVisitor(unittest.TestCase):

    def setUp(self):
        prec_perf_advice_dict = utils.parse_precision_performance_advice_file()
        api_prec_dict = prec_perf_advice_dict.get("api_precision_dict")
        api_perf_dict = prec_perf_advice_dict.get("api_performance_dict")
        api_params_perf_dict = prec_perf_advice_dict.get("api_parameters_performance_dict")
        perf_api_suggest_dict = prec_perf_advice_dict.get("performance_api_suggest_use")
        perf_api_suggest = PerfApiSuggest(perf_api_suggest_dict)
        self.perf_config_dict = prec_perf_advice_dict.get("performance_configuration_dict")
        self.advice_info = AdviceInfo(api_prec_dict, api_perf_dict, api_params_perf_dict, perf_api_suggest)

    def test_precision(self):
        py_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

# 1. baddbmm has precision problem
M = torch.randn(10, 3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
torch.baddbmm(M, batch1, batch2).size()
# 2. reshpae has precision problem
torch.reshape(batch1, (2, 5, 3, 4))
# 3. nn.Embedding has precision problem
n, d, m = 3, 5, 7
embedding = nn.Embedding(n, d, max_norm=True)
# 4. nn.function.embedding has precision problem
input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedding_matrix = torch.rand(10, 3)
F.embedding(input, embedding_matrix)
# 5. torch.matmul has precision problem
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (precision_advice_list, _), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = {
            "torch.baddbmm", "torch.reshape", "torch.nn.Embedding",
            "torch.nn.functional.embedding", "torch.matmul"
        }
        result = self._get_result_name_set(precision_advice_list)
        assert result == expect

    def test_perf_api(self):
        py_code = """
import torch
import torch_npu
import torch.optim as optim
import torch.nn as nn

# 1. overflow checking
torch.autograd.set_detect_anomaly(True)
torch.autograd.detect_anomaly()
torch.autograd.gradcheck()
# 2. optim
optimzier1 = optim.Lamb()
optimizer2 = optim.RMSpropTF()
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = {
            "torch.autograd.set_detect_anomaly",
            "torch.autograd.detect_anomaly",
            "torch.autograd.gradcheck",
            "torch.optim.Lamb",
            "torch.optim.RMSpropTF"
        }
        result = self._get_result_name_set(performance_advice_list)
        assert result == expect

    def test_api_params_perf_1(self):
        py_code = """
import torch
import torch_npu
from torch.utils.data import DataLoader
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

# 1. torch.utils.data.DataLoader
dataset = mockset()
loader = DataLoader(dataset)

model = Model()
# 2. torch.nn.parallel.DistributedDataParallel
model = DDP(model)
# 3. to (use index)
model = model.to("npu:0", True)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = {
            "When using the Ascend AI processor for training, it is recommended to set pin_memory to True.\n"
            "For more information, please refer to: https://www.hiascend.com/document/detail/zh/"
            "canncommercial/700/modeldevpt/ptmigr/AImpug_000056.html",
            "To speed up data loading, it is recommended to set num_workers to a value greater than 0.",
            "To optimize communication between devices, it is recommended to set bucket_cap_mb to 500.",
        }
        result = self._get_result_suggest_set(performance_advice_list)
        assert result == expect

    def test_api_params_perf_2(self):
        py_code = """
import torch
import torch_npu
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

model = Model()
# 1. to (use key-val pair)
model = model.to("npu:0", non_blocking=False)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = {
            "To optimize communication between devices, it is recommended to set non_blocking to True.\n"
            "For more information, please refer to: https://www.hiascend.com/document/detail/zh/"
            "canncommercial/700/modeldevpt/ptmigr/AImpug_000055.html",
        }
        result = self._get_result_suggest_set(performance_advice_list)
        assert result == expect

    def test_api_params_perf_3(self):
        py_code = """
import torch
import torch_npu
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

model = Model()
# 1. to (use key-val pair)
model = model.to("npu:0", False)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = {
            "To optimize communication between devices, it is recommended to set non_blocking to True.\n"
            "For more information, please refer to: https://www.hiascend.com/document/detail/zh/"
            "canncommercial/700/modeldevpt/ptmigr/AImpug_000055.html",
        }
        result = self._get_result_suggest_set(performance_advice_list)
        assert result == expect

    def test_api_params_perf_4(self):
        py_code = """
import torch
import torch_npu
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

model = Model()
# 1. to (use key-val pair)
model = model.to("npu:0", torch.float, True)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, performance_advice_list), _, _ = \
            analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        expect = set()
        result = self._get_result_suggest_set(performance_advice_list)
        assert result == expect

    def test_perf_api_suggest_use_1(self):
        py_code = """
import torch
import torch_npu
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

model = Model()
# 1. torch.nn.parallel.DistributedDataParallel
model = DDP(model)
# 3. to (use index)
model = model.to("npu:0", True)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, _), _, _ = analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        perf_suggest = generate_perf_suggest(self.advice_info.perf_api_suggest)
        expect = {
            "DistributedDataParallel is used in the codes, it is recommended to use no_sync().",
        }
        result = self._get_result_suggest_set(perf_suggest)
        assert result == expect

    def test_perf_api_suggest_use_2(self):
        py_code = """
import torch
import torch_npu
import torch.nn.parallel.DistributedDataParallel as DDP
from utils import mockset, Model

model = Model()
# 1. torch.nn.parallel.DistributedDataParallel
model = DDP(model)
# 3. to (use index)
model = model.to("npu:0", True)
with model.no_sync():
    pass
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, _), _, _ = analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        perf_suggest = generate_perf_suggest(self.advice_info.perf_api_suggest)
        expect = set()
        result = self._get_result_suggest_set(perf_suggest)
        assert result == expect

    def test_perf_api_suggest_use_3(self):
        py_code = """
import torch
import torch_npu
from utils import mockset, Model

model = Model()
model = model.to("npu:0", True)
        """
        wrapper = libcst.metadata.MetadataWrapper(libcst.parse_module(py_code))
        (_, _), _, _ = analyse_precision_performance_advice_api(wrapper, self.advice_info, None)
        perf_suggest = generate_perf_suggest(self.advice_info.perf_api_suggest)
        expect = set()
        result = self._get_result_suggest_set(perf_suggest)
        assert result == expect

    def _get_result_name_set(self, result: List[ApiInstance]):
        result_set = set()
        for api in result:
            result_set.add(api.name)
        return result_set
    
    def _get_result_suggest_set(self, result: List[ApiInstance]):
        result_set = set()
        for api in result:
            result_set.add(api.info)
        return result_set
