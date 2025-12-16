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

import unittest
import struct
import numpy as np
from unittest import mock

from dump_parse import dump, mapping
from vector_cmp.fusion_manager import fusion_op
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.fusion_manager.compare_npu_vs_npu import NpuVsNpuComparison
from algorithm_manager.algorithm_manager import AlgorithmManager
from dump_parse import dump, dump_utils, mapping
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from cmp_utils.constant.const_manager import DD

class TestUtilsMethods(unittest.TestCase):

    def test_compare_npu_vs_npu1(self):
        compare_data = dump.CompareData('/home/left', '/home/right', 1, False, "")
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_list = [fusion_op.FusionOp(4, 'aaa', [], 'Left', ['/home/left/aaa.aaa.21.333333'], attr)]
        ret, match, result = NpuVsNpuComparison(compare_data, fusion_op_list,
                                                AlgorithmManager('', 'all', '')).compare()
        result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(match, False)
        self.assertEqual(len(result_list), 1)
        actual = ['4', 'aaa', '*', 'NaN', '[aaa] There is no right dump file for the op "aaa".']
        self.assertEqual(result_list[0][0], actual[0])

    def test_compare_npu_vs_npu2(self):
        compare_data = dump.CompareData('/home/left', '/home/right', 1, False, "")
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_list = [fusion_op.FusionOp(6, 'xxx', [], 'Right', ['/home/right/aaa.aaa.21.333333'], attr)]
        ret, match, result = NpuVsNpuComparison(compare_data, fusion_op_list,
                                                AlgorithmManager('', 'all', '')).compare()
        result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(match, False)
        self.assertEqual(len(result_list), 1)
        actual = ['6', '*', 'xxx', 'NaN', '[xxx] There is no left dump file for the op "xxx".']
        self.assertEqual(result_list[0][0], actual[0])

    def test_compare_npu_vs_npu3(self):
        compare_data = dump.CompareData('/home/left', '/home/right', 1, False, "")
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_list = [
            fusion_op.FusionOp(6, 'xxx', [], 'Left',
                               ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr),
            fusion_op.FusionOp(6, 'xxx', [], 'Right', ['/home/right/aaa.aaa.21.333333'], attr)]
        left_dump_data = DumpData()
        left_dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        left_dump_data = dump_utils.convert_dump_data(left_dump_data)
        right_dump_data = DumpData()
        right_dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        right_dump_data = dump_utils.convert_dump_data(right_dump_data)
        with mock.patch('dump_parse.dump_utils.parse_dump_file',
                        side_effect=[left_dump_data, right_dump_data, left_dump_data, right_dump_data]):
            ret, match, result = NpuVsNpuComparison(compare_data,
                                                    fusion_op_list, AlgorithmManager('', 'all', '')).compare()
            result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
        self.assertEqual(match, True)
        self.assertEqual(len(result_list), 1)
        actual = ['6', 'xxx', 'xxx', 'NaN',
                  '[xxx] The number of output does not match (0 vs 1).,[xxx] The number of input does not match (1 vs 0).']
        self.assertEqual(result_list[0][0], actual[0])

    def test_compare_npu_vs_npu4(self):
        compare_data = dump.CompareData('/home/left', '/home/right', 1, False, "")
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_list = [
            fusion_op.FusionOp(6, 'aaa', [], 'Left', ['/home/left/aaa.aaa.21.999999'], attr),
            fusion_op.FusionOp(6, 'aaa', [], 'Right', ['/home/right/aaa.aaa.21.999999'], attr)]
        left_dump_data = DumpData()
        left_dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        left_dump_data = dump_utils.convert_dump_data(left_dump_data)
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with mock.patch('dump_parse.dump_utils.parse_dump_file',
                        side_effect=[left_dump_data, left_dump_data]):
            with mock.patch('sys.argv', args):
                manager = AlgorithmManager('', 'all', '')
                ret, match, result = NpuVsNpuComparison(compare_data, fusion_op_list, manager).compare()
                result_list = self.get_result_list(result)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(match, True)
        self.assertEqual(len(result_list), 1)
        actual = ['6', 'aaa', 'aaa', 'float32', 'NaN', 'aaa', 'float32', 'NaN', 'aaa:input:0', '[1,3,4,4]',
            '1.000000', '0.000000', '0.000000', '0.000000', '0.000000', '(23.500;13.853),(23.500;13.853)',
            '0.000000', '0.000000', '0.000000', '0.000000', '']

        self.assertEqual(result_list[0], actual)

    def test_compare_npu_vs_npu5(self):
        compare_data = dump.CompareData('/home/left', '/home/right', 1, False, "")
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_list = [
            fusion_op.FusionOp(6, 'aaa', [], 'Left', ['/home/left/aaa.aaa.21.333333'], attr),
            fusion_op.FusionOp(6, 'aaa', [], 'Right', ['/home/right/aaa.aaa.21.333333'], attr)]
        left_dump_data = DumpData()
        left_dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        left_dump_data = dump_utils.convert_dump_data(left_dump_data)
        right_dump_data = DumpData()
        right_dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 2, 4, 4]))
        right_dump_data = dump_utils.convert_dump_data(right_dump_data)
        with mock.patch('dump_parse.dump_utils.parse_dump_file',
                        side_effect=[right_dump_data, left_dump_data]):
            ret, match, result = NpuVsNpuComparison(compare_data,
                                                    fusion_op_list, AlgorithmManager('', 'all', '')).compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
        self.assertEqual(match, True)

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = OpOutput()
        op_output.data_type = DD.DT_FLOAT
        op_output.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_output.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('f' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _make_op_input(dd_format, shape):
        op_input = OpInput()
        op_input.data_type = DD.DT_FLOAT
        op_input.format = dd_format
        length = 1
        if shape is None:
            length = 20
        else:
            for dim in shape:
                op_input.shape.dim.append(dim)
                length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_input.data = struct.pack('f' * length, *origin_numpy)
        return op_input

    @staticmethod
    def get_result_list(result):
        result_list = []
        for single_res in result:
            for item in single_res.result_list:
                result_list.append(item)
        return result_list

if __name__ == '__main__':
    unittest.main()
