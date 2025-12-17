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

sys.path.append(os.path.abspath("../../../../"))
sys.path.append(os.path.abspath("../../../../src/ms_fmk_transplt"))

from analysis.precision_performance_advice_analysis.prec_perf_utils import PerfApiSuggest


class TestPrecPerfUtils(unittest.TestCase):

    def setUp(self):
        self.perf_suggest = {
            "no_sync": {
                "dependency": ["torch.nn.parallel.DistributedDataParallel"],
                "msg": "DistributedDataParallel is used in the codes, it is recommended to use no_sync()."
            },
            "mock_api": {
                "dependency": ["mock_api_dept1", "mock_api_dept2"],
                "msg": "mock_api_dept is used, it is recommended to use mock_api()"
            }
        }
        self.expect_dept_val = {
            "torch.nn.parallel.DistributedDataParallel": False,
            "mock_api_dept1": False,
            "mock_api_dept2": False
        }

    def test_set_dependence(self):
        perf_api_inst = PerfApiSuggest(self.perf_suggest)
        assert perf_api_inst.dependency == self.expect_dept_val
