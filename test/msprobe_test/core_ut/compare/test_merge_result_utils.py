# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

import unittest
from unittest.mock import patch

from msprobe.core.common.const import CompareConst
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.merge_result.utils import replace_compare_index_dict, check_config


class TestReplaceCompareIndexDict(unittest.TestCase):

    def setUp(self):
        # 初始化测试数据
        self.compare_index_dict = {
            'Max diff': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            'L2norm diff': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            'MeanRelativeErr': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            CompareConst.NPU_MAX: {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            },
            CompareConst.BENCH_MAX: {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            }
        }
        self.compare_index_list = ['Max diff', 'L2norm diff', 'MeanRelativeErr', 'NPU max', 'Bench max']
        self.rank_num = 0

    def test_process_compare_index_dict_na(self):
        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 检查是否替换了 N/A 值
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

    def test_no_na_values(self):
        # 修改测试数据，确保没有 N/A 值
        for index in self.compare_index_list[:-2]:  # 排除 'NPU max' 和 'Bench max'
            self.compare_index_dict[index] = {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            }

        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 验证返回值没有变化
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

    def test_non_string_npu_bench(self):
        # 修改 NPU 和 Bench 统计量为非字符串类型
        self.compare_index_dict[CompareConst.NPU_MAX] = {
            'op_name_1': {0: 123},
            'op_name_2': {0: 123}
        }
        self.compare_index_dict[CompareConst.BENCH_MAX] = {
            'op_name_1': {0: 123},
            'op_name_2': {0: 123}
        }

        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        expected_value = 'NPU:123  Bench:123'
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], expected_value)

    def test_missing_npu_bench_max(self):
        # 移除 NPU_MAX 和 BENCH_MAX 键
        del self.compare_index_dict[CompareConst.NPU_MAX]
        del self.compare_index_dict[CompareConst.BENCH_MAX]

        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 验证原始数据未改变
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'N/A')

    def test_unsupported_values(self):
        # 'unsupported'
        self.compare_index_dict['Max diff'] = {
            'op_name_1': {0: 'unsupported'},
            'op_name_2': {0: 'unsupported'}
        }
        self.compare_index_dict['L2norm diff'] = {
            'op_name_1': {0: 'unsupported'},
            'op_name_2': {0: 'unsupported'}
        }
        self.compare_index_dict['MeanRelativeErr'] = {
            'op_name_1': {0: 'unsupported'},
            'op_name_2': {0: 'unsupported'}
        }

        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 检查是否替换了'unsupported'
        expected_value = 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3'

        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], expected_value)

    def test_nan_values(self):
        # 'Nan'
        self.compare_index_dict['Max diff'] = {
            'op_name_1': {0: 'Nan'},
            'op_name_2': {0: 'Nan'}
        }
        self.compare_index_dict['L2norm diff'] = {
            'op_name_1': {0: 'Nan'},
            'op_name_2': {0: 'Nan'}
        }
        self.compare_index_dict['MeanRelativeErr'] = {
            'op_name_1': {0: 'Nan'},
            'op_name_2': {0: 'Nan'}
        }

        result = replace_compare_index_dict(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 检查是否替换了'Nan'
        expected_value = 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3'

        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], expected_value)

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], expected_value)
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], expected_value)

    def test_empty_dict(self):
        # 测试空字典的处理
        empty_dict = {}
        result = replace_compare_index_dict(empty_dict, [], self.rank_num)
        self.assertEqual(result, {})

    def test_empty_compare_index_list(self):
        # 测试空 compare_index_list 的情况
        result = replace_compare_index_dict(self.compare_index_dict, [], self.rank_num)
        self.assertEqual(result, self.compare_index_dict)


class TestCheckConfig(unittest.TestCase):

    @patch('msprobe.core.common.file_utils.logger.error')
    def test_check_config_empty(self, mock_logger_error):
        config = None

        with self.assertRaises(CompareException):
            check_config(config)

        mock_logger_error.assert_called_once_with('config.yaml is empty, please check.')

    @patch('msprobe.core.common.file_utils.logger.error')
    def test_check_config_missing_api(self, mock_logger_error):
        config = {
            'compare_index': ['index1', 'index2']
        }

        with self.assertRaises(CompareException):
            check_config(config)

        mock_logger_error.assert_called_once_with('The APIs required to merge data were not found.')

    @patch('msprobe.core.common.file_utils.logger.error')
    def test_check_config_api_is_not_list(self, mock_logger_error):
        config = {
            'api': 'api1',
            'compare_index': ['index1', 'index2']
        }

        with self.assertRaises(CompareException):
            check_config(config)

        mock_logger_error.assert_called_once_with("The config format of 'api' is incorrect, please check.")

    @patch('msprobe.core.common.file_utils.logger.error')
    def test_check_config_compare_index_is_not_list(self, mock_logger_error):
        config = {
            'api': ['api1', 'api2'],
            'compare_index': 'index1'
        }

        with self.assertRaises(CompareException):
            check_config(config)

        mock_logger_error.assert_called_once_with("The config format of 'compare_index' is incorrect, please check.")

    def test_check_config_compare_index_is_none(self):
        config = {
            'api': ['api1', 'api2'],
            'compare_index': None
        }
        result_target = {
            'api': ['api1', 'api2'],
            'compare_index': []
        }
        result = check_config(config)

        self.assertEqual(result, result_target)

    @patch('msprobe.core.common.file_utils.logger.error')
    def test_check_config_success(self, mock_logger_error):
        config = {
            'api': ['api1', 'api2'],
            'compare_index': ['index1', 'index2']
        }

        result = check_config(config)

        self.assertEqual(result, config)
        mock_logger_error.assert_not_called()
