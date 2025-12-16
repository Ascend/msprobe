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
from unittest import mock
import pytest
import numpy as np
import struct
from dump_parse.proto_dump_data import DumpData, OpInput, OpOutput
from vector_cmp.compare_detail import compare_detail
from vector_cmp.compare_detail import detail
from cmp_utils import utils, utils_type, path_check
from vector_cmp.fusion_manager import fusion_op
from format_manager.format_manager import FormatManager
from cmp_utils.constant.const_manager import DD
from cmp_utils.constant.compare_error import CompareError
from dump_parse import dump, dump_utils, mapping


class TestUtilsMethods(unittest.TestCase):

    def test_get_my_output_tensor_list1(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('prob', 'output', '3')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(6, 'xxx', [], 'Left',
                                            ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=True)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_UNSUPPORTED_COMPARE_ERROR)

    def test_get_my_output_tensor_list2(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('prob', 'output', '3')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(6, 'xxx', [], 'Left',
                                            ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        fusion_op_comparison.compare_data.get_left_dump_data = \
            mock.Mock(side_effect=CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR))
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                            return_value=utils_type.FusionRelation.L1Fusion):
                ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_get_my_output_tensor_list3(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('prob', 'output', '3')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(6, 'xxx', [], 'Left',
                                            ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        dump_data = DumpData()
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        fusion_op_comparison.compare_data.get_left_dump_data = mock.Mock(return_value=("/home/bin/", dump_data))
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                            return_value=utils_type.FusionRelation.L1Fusion):
                ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_get_my_output_tensor_list4(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('prob', 'input', '3')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(6, 'xxx', [], 'Left',
                                            ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        dump_data = DumpData()
        dump_data.input.append(self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        fusion_op_comparison.compare_data.get_left_dump_data = mock.Mock(return_value=("/home/bin", dump_data))
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                            return_value=utils_type.FusionRelation.L1Fusion):
                ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare1(self):
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId('prob', 'output', '0')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(
            6, 'prob', [], 'Left', ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        dump_data = DumpData()
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        fusion_op_comparison.compare_data.get_left_dump_data = mock.Mock(return_value=("/home/bin", dump_data))
        fusion_op_comparison.get_right_dump_data = mock.Mock(
            side_effect=CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR))
        fusion_op_comparison.sort_l1_fusion_dump_file = mock.Mock(return_value=[[10, fusion_op.FusionOp(
            6, 'gggg', [], 'Left', ['/home/left/aaa.aaa.21.333333'], attr), 'xx,bb']])
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                            return_value=utils_type.FusionRelation.L1Fusion):
                with mock.patch('os.path.exists', return_value=False):
                    ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare2(self):
        detail_info = mock.Mock()
        detail_info.top_n = 1
        detail_info.tensor_id = detail.TensorId('prob', 'output', '0')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(
            6, 'prob', [], 'Left', ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        fusion_op_comparison.format_manager = FormatManager("")
        fusion_op_comparison.format_manager.check_arguments_valid()
        fusion_op_comparison.compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=True)
        dump_data = DumpData()
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        tensor = fusion_op.Tensor('prob', 0, "NCHW", [1, 3, 4, 4])
        tensor.set_path('/home/bin')
        tensor.set_data(dump_data.output_data[0])
        fusion_op_comparison.compare_data.get_left_dump_data = mock.Mock(return_value=("/home/bin", dump_data))
        fusion_op_comparison.get_right_dump_data = mock.Mock(return_value=tensor)
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with pytest.raises(CompareError) as err:
            with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                            return_value=utils_type.FusionRelation.OneToOne):
                with mock.patch('os.path.exists', return_value=False):
                    with mock.patch('numpy.save'):
                        ret = detail_comparison.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_compare3(self):
        detail_info = mock.Mock()
        detail_info.top_n = 10
        detail_info.ignore_result = True
        detail_info.tensor_id = detail.TensorId('prob', 'output', '0')
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(
            6, 'prob', [], 'Left', ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        detail_info.make_detail_header = mock.Mock(
            return_value='Index,N C H W,LeftOp,RightOp,AbsoluteError,RelativeError')
        fusion_op_comparison = mock.Mock()
        fusion_op_comparison.compare_rule = mock.Mock()
        fusion_op_comparison.compare_rule.fusion_info = mock.Mock()
        fusion_op_comparison.compare_data = mock.Mock()
        fusion_op_comparison.format_manager = FormatManager("")
        fusion_op_comparison.format_manager.check_arguments_valid()
        fusion_op_comparison.compare_data.is_standard_quant_vs_origin = mock.Mock(return_value=False)
        dump_data = DumpData()
        dump_data.output.append(self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data = dump_utils.convert_dump_data(dump_data)
        fusion_op_comparison.compare_data.get_left_dump_data = mock.Mock(return_value=("/home/bin", dump_data))
        tensor = fusion_op.Tensor('prob', 0, "NCHW", [1, 3, 4, 4])
        tensor.set_path('/home/bin')
        tensor.set_data(dump_data.output_data[0])
        fusion_op_comparison.get_right_dump_data = mock.Mock(return_value=tensor)
        detail_comparison = compare_detail.DetailComparison(detail_info, fusion_op_comparison, "/home/demo")
        with mock.patch('vector_cmp.fusion_manager.fusion_rule_parser.get_relation_for_fusion', 
                        return_value=utils_type.FusionRelation.OneToOne):
            with mock.patch('os.path.exists', return_value=False):
                with mock.patch('numpy.save'):
                    with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                        open_file.write = None
                        ret = detail_comparison.compare()
        self.assertEqual(ret, 0)

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
