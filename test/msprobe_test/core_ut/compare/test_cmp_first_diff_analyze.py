# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import unittest
from unittest.mock import patch

import pandas as pd

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.diff_analyze.first_diff_analyze import FirstDiffAnalyze
from msprobe.core.compare.config import ModeConfig


class TestFirstDiffAnalyze(unittest.TestCase):
    def setUp(self):
        self.header = ['NPU name', 'L2norm diff',
                       'MaxRelativeErr', 'MinRelativeErr', 'MeanRelativeErr', 'NormRelativeErr',
                       'state', 'api_origin_name']
        self.data = [
            ['Functional.conv2d.0.forward.input.0', 1, '0.0%', '0.0%', '0.0%', '0.0%', 'input', 'Functional.conv2d.0.forward'],
            ['Functional.conv2d.0.forward.output.0', 1, '99.0%', '99.0%', '99.0%', '99.0%', 'output', 'Functional.conv2d.0.forward']
        ]
        self.result_df = pd.DataFrame(self.data, columns=self.header)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5]})
    def test_single_metric_diff_check_true(self):
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, '')
        result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '60.0%')
        self.assertTrue(result)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5]})
    def test_single_metric_diff_check_false(self):
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, '')
        result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertFalse(result)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'NormRelativeErr': [0.5]})
    def test_single_metric_diff_check_miss_threshold(self):
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, '')
        with self.assertRaises(CompareException) as context:
            result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertEqual(context.exception.code, CompareException.MISSING_THRESHOLD_ERROR)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5, 1.0]})
    def test_single_metric_diff_check_wrong_threshold(self):
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, '')
        with self.assertRaises(CompareException) as context:
            result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertEqual(context.exception.code, CompareException.WRONG_THRESHOLD_ERROR)

    def test_single_api_check_within_threshold(self):
        result_slice = [
            ['Functional.conv2d.0.forward.input.0', 1, '0.0%', '0.0%', '0.0%', '0.0%', 'input', 'Functional.conv2d.0.forward'],
            ['Functional.conv2d.0.forward.output.0', 1, '0.1%', '0.1%', '0.1%', '0.1%', 'output', 'Functional.conv2d.0.forward']
        ]
        expected_result = {
            'is_same': True,
            'op_items': [
                {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '0.0%', 'MinRelativeErr': '0.0%',
                 'MeanRelativeErr': '0.0%', 'NormRelativeErr': '0.0%',
                 'state': 'input', 'api_origin_name': 'Functional.conv2d.0.forward'},
                {'NPU name': 'Functional.conv2d.0.forward.output.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '0.1%', 'MinRelativeErr': '0.1%',
                 'MeanRelativeErr': '0.1%', 'NormRelativeErr': '0.1%',
                 'state': 'output', 'api_origin_name': 'Functional.conv2d.0.forward'}
            ]
        }
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, 'rank0')
        result = first_diff_analyze.single_api_check(result_slice, self.header)
        self.assertEqual(result, expected_result)

    def test_single_api_check_exceed_threshold(self):
        result_slice = [
            ['Functional.conv2d.0.forward.input.0', 1, '88.0%', '88.0%', '88.0%', '88.0%', 'input', 'Functional.conv2d.0.forward'],
            ['Functional.conv2d.0.forward.output.0', 1, '99.0%', '99.0%', '99.0%', '99.0%', 'output', 'Functional.conv2d.0.forward']
        ]
        expected_result = {
            'is_same': False,
            'op_items': [
                {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '88.0%', 'MinRelativeErr': '88.0%',
                 'MeanRelativeErr': '88.0%', 'NormRelativeErr': '88.0%',
                 'state': 'input', 'api_origin_name': 'Functional.conv2d.0.forward'},
                {'NPU name': 'Functional.conv2d.0.forward.output.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '99.0%', 'MinRelativeErr': '99.0%',
                 'MeanRelativeErr': '99.0%', 'NormRelativeErr': '99.0%',
                 'state': 'output', 'api_origin_name': 'Functional.conv2d.0.forward'},
            ]
        }
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, 'rank0')
        result = first_diff_analyze.single_api_check(result_slice, self.header)
        self.assertEqual(result, expected_result)

    def test_single_api_check_md5_same_true(self):
        md5_header = CompareConst.MD5_COMPARE_RESULT_HEADER + [CompareConst.STACK, Const.STATE, Const.API_ORIGIN_NAME]
        result_slice = [
            ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.int32', 'torch.int32',
             '[]', '[]', 'True', 'True', '2144df1c', '2144df1c', True, 'pass', '[]',
             '', 'input', 'Functional.conv2d.0.forward']
        ]
        expected_result = {
            'is_same': True,
            'op_items': [
                {CompareConst.NPU_NAME: 'Functional.conv2d.0.forward.input.0',
                 CompareConst.BENCH_NAME: 'Functional.conv2d.0.forward.input.0',
                 CompareConst.NPU_DTYPE: 'torch.int32', CompareConst.BENCH_DTYPE: 'torch.int32',
                 CompareConst.ERROR_MESSAGE: '[]',
                 CompareConst.NPU_SHAPE: '[]', CompareConst.BENCH_SHAPE: '[]',
                 CompareConst.NPU_REQ_GRAD: 'True', CompareConst.BENCH_REQ_GRAD: 'True',
                 CompareConst.NPU_MD5: '2144df1c', CompareConst.BENCH_MD5: '2144df1c',
                 CompareConst.REQ_GRAD_CONSIST: True,
                 CompareConst.RESULT: 'pass', CompareConst.STACK: '',
                 Const.STATE: 'input', Const.API_ORIGIN_NAME: 'Functional.conv2d.0.forward'
                 }
            ]
        }
        mode_config = ModeConfig(dump_mode=Const.MD5, first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, 'rank0')
        result = first_diff_analyze.single_api_check(result_slice, md5_header)
        self.assertEqual(result, expected_result)

    def test_single_api_check_md5_same_false(self):
        md5_header = CompareConst.MD5_COMPARE_RESULT_HEADER + [CompareConst.STACK, Const.STATE, Const.API_ORIGIN_NAME]
        result_slice = [
            ['Functional.conv2d.0.forward.output.0', 'Functional.conv2d.0.forward.output.0', 'torch.int32', 'torch.int32',
             '[]', '[]', 'True', 'True', '2144df1c', '2100df1c', True, 'Different', '[]',
             '', 'output', 'Functional.conv2d.0.forward']
        ]
        expected_result = {
            'is_same': False,
            'op_items': [
                {CompareConst.NPU_NAME: 'Functional.conv2d.0.forward.output.0',
                 CompareConst.BENCH_NAME: 'Functional.conv2d.0.forward.output.0',
                 CompareConst.NPU_DTYPE: 'torch.int32', CompareConst.BENCH_DTYPE: 'torch.int32',
                 CompareConst.ERROR_MESSAGE: '[]',
                 CompareConst.NPU_SHAPE: '[]', CompareConst.BENCH_SHAPE: '[]',
                 CompareConst.NPU_REQ_GRAD: 'True', CompareConst.BENCH_REQ_GRAD: 'True',
                 CompareConst.NPU_MD5: '2144df1c', CompareConst.BENCH_MD5: '2100df1c',
                 CompareConst.REQ_GRAD_CONSIST: True,
                 CompareConst.RESULT: 'Different', CompareConst.STACK: '',
                 Const.STATE: 'output', Const.API_ORIGIN_NAME: 'Functional.conv2d.0.forward'
                 }
            ]
        }
        mode_config = ModeConfig(dump_mode=Const.MD5, first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, 'rank0')
        result = first_diff_analyze.single_api_check(result_slice, md5_header)
        self.assertEqual(result, expected_result)

    def test_check_summary(self):
        expected_result = {
            'Functional.conv2d.0.forward': {
                'is_same': False,
                'op_items': [
                    {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                     'MaxRelativeErr': '0.0%', 'MinRelativeErr': '0.0%',
                     'MeanRelativeErr': '0.0%', 'NormRelativeErr': '0.0%',
                     'state': 'input', 'api_origin_name': 'Functional.conv2d.0.forward'},
                    {'NPU name': 'Functional.conv2d.0.forward.output.0', 'L2norm diff': 1,
                     'MaxRelativeErr': '99.0%', 'MinRelativeErr': '99.0%',
                     'MeanRelativeErr': '99.0%', 'NormRelativeErr': '99.0%',
                     'state': 'output', 'api_origin_name': 'Functional.conv2d.0.forward'},
                ]
            }
        }
        mode_config = ModeConfig(first_diff_analyze=True)
        first_diff_analyze = FirstDiffAnalyze(mode_config, 'rank1')
        result = first_diff_analyze.check(self.result_df)
        self.assertEqual(result, expected_result)
