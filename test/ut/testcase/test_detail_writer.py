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

import queue

import unittest
from unittest import mock
import numpy as np

from vector_cmp.compare_detail.detail_writer import MinMaxValue
from vector_cmp.compare_detail.detail_writer import TopN
from vector_cmp.compare_detail.detail_writer import DetailWriter
from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.compare_detail import detail
from vector_cmp.fusion_manager import fusion_op


class TestUtilsMethods(unittest.TestCase):
    def test_calculate(self):
        min_max_value = MinMaxValue()

        min_max_value.set_min_absolute_error(0.010678)
        min_max_value.set_max_absolute_error(0.031430)
        min_max_value.set_min_relative_error(0.000000)
        min_max_value.set_max_relative_error(0.057544)
        self.assertEqual(
            "MinAbsoluteError:0.010678\nMaxAbsoluteError:0.031430\nMinRelativeError:0.000000\nMaxRelativeError:0.057544",
            min_max_value.get_min_and_max_value())

        top_n = TopN()
        error_type = "absolute"
        top_n.set_absolute_top_n_list([0, 1, 3, 6],
                                      [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 6]],
                                      [4.853518, 0.577631, 1.002630, 1.086603],
                                      [4.864196, 0.546201, 0.990363, 1.071308],
                                      [0.010678, 0.031430, 0.012268, 0.015295],
                                      [0.002195, 0.057544, 0.012387, np.nan], error_type)
        error_type = "relative"
        top_n.set_relative_top_n_list([0, 1, 3],
                                      [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 3]],
                                      [4.853518, 0.577631, 1.002630],
                                      [4.864196, 0.546201, 0.990363],
                                      [0.010678, 0.031430, 0.012268],
                                      [0.002195, 0.057544, 0.012387], error_type)

        self.assertEqual(top_n.get_absolute_error_top_n_list()[0],
                         '1,0 0 0 1,0.577631\t,0.546201\t,0.031430\t,0.057544\t\n')
        self.assertEqual(top_n.get_relative_error_top_n_list()[0],
                         '1,0 0 0 1,0.577631\t,0.546201\t,0.031430\t,0.057544\t\n')

    def test_padding_shape_to_4d(self):
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        dims = detail_writer._padding_shape_to_4d((1, 4, 6, 10))
        self.assertEqual(dims, [1, 4, 6, 10])

    def test_write_detail_result_multi_proc(self):
        generator = ((x, [i for i in range(x * 10, (x + 1) * 10)], [i for i in range(x * 10, (x + 1) * 10)],
                      [i for i in range(x * 10, (x + 1) * 10)]) for x in range(2000))
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        with mock.patch('multiprocessing.cpu_count', return_value=8):
            detail_writer._write_detail_result_multi_proc(generator)

    def test_separate_group_generator(self):
        detail_info = mock.Mock()
        detail_info.max_line = 20
        detail_writer = DetailWriter('/home/demo/', detail_info)
        dim_list = [i for i in range(50)]
        my_output_data = np.array([np.random.random() for _ in range(50)])
        ground_truth_data = np.array([np.random.random() for _ in range(50)])
        abs_error = np.array([np.random.random() for _ in range(50)])
        relative_error = np.array([np.random.random() for _ in range(50)])
        generator = detail_writer._separate_group_generator(dim_list, (my_output_data, ground_truth_data,
                                                                       abs_error, relative_error))
        group = next(generator)
        self.assertEqual(len(group), 7)
        self.assertEqual(len(group[1]), 20)
        next(generator)
        group = next(generator)
        self.assertEqual(len(group[1]), 10)

    def test_multi_process_group_write(self):
        task_group = [(1, [i for i in range(20)], np.array([np.random.random() for _ in range(20)]),
                       np.array([np.random.random() for _ in range(20)]),
                       np.array([np.random.random() for _ in range(20)]),
                       np.array([np.random.random() for _ in range(20)]), False)]
        detail_info = mock.Mock()
        detail_info.ignore_result = False
        detail_info.max_line = 20
        detail_writer = DetailWriter('/home/demo/', detail_info)
        param_queue = queue.Queue()
        with mock.patch.object(detail_writer, '_make_detail_output_file'), \
                mock.patch.object(detail_writer, '_write_one_detail'), \
                mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            open_file.write = None
            open_file.close = None
            detail_writer._multi_process_group_write(task_group, param_queue)
            self.assertEqual(param_queue.get(), 20)

    def test_transform_dim_list(self):
        detail_info = mock.Mock()
        detail_info.detail_format = ''
        detail_writer = DetailWriter('/home/demo/', detail_info)
        dim = (1, 2, 1, 4)
        dims = detail_writer._transform_dim_list(dim)
        self.assertEqual(dims[0], (0, 0, 0, 0))
        self.assertEqual(dims[5], (0, 1, 0, 1))

    def test_array_calculate(self):
        detail_info = mock.Mock()
        detail_info.top_n = 2
        detail_writer = DetailWriter('/home/demo/', detail_info)
        my_output_data = np.array([4.853518, 0.577631, 1.002630, 1.086603, 0.000003])
        ground_truth_data = np.array([4.864196, 0.546201, 0.990363, 1.086603, 0.0])
        dim_list = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 6], [0, 0, 1, 1]]
        detail_writer._array_calculate(my_output_data, ground_truth_data, dim_list)
        self.assertEqual(detail_writer._total_num, 5)
        self.assertEqual(detail_writer.min_max_value._min_absolute_error, 0.0)
        self.assertEqual(detail_writer.min_max_value._max_absolute_error, 0.03142999999999996)
        self.assertEqual(detail_writer.min_max_value._min_relative_error, 0.0)
        self.assertEqual(detail_writer.min_max_value._max_relative_error, 0.057542919181766336)
        abs_top_n_list = detail_writer.top_n.get_absolute_error_top_n_list()
        self.assertEqual(len(abs_top_n_list), 2)
        self.assertEqual(abs_top_n_list[0],
                         '1,0 0 0 1,0.577631\t,0.546201\t,0.031430\t,0.057543\t\n')
        relative_top_n_list = detail_writer.top_n.get_relative_error_top_n_list()
        self.assertEqual(len(relative_top_n_list), 2)
        self.assertEqual(relative_top_n_list[0], '1,0 0 0 1,0.577631\t,0.546201\t,0.031430\t,0.057543\t\n')

    def test_write1(self):
        op = "gradients_logistic_loss_8_Log1p_grad_addgradients_logistic_loss_8_Log1p_grad_Reciprocalgradients_" \
             "logistic_loss_8_Log1p_grad_mulgradients_logistic_loss_8_Exp_grad_mulgradients_logistic_loss_8_Select_1_" \
             "grad_Selectgradients_logistic_loss_8_Select_1_grad_Select_1gradients_logistic_loss_8_Neg_grad_Neggradients_AddN_3"
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId(op, 'output', '3')
        detail_info.top_n = 3
        detail_info.ignore_result = True
        detail_info.my_output_ops = op
        detail_info.ground_truth_ops = op
        detail_info.detail_format = 'N C H W'
        attr = fusion_op.OpAttr([], '', False, 12)
        fusion_op_info = fusion_op.FusionOp(
            6, op, [], 'Left', ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        detail_info.make_detail_header = mock.Mock(
            return_value='Index,N C H W,LeftOp,RightOp,AbsoluteError,RelativeError')
        detail_writer = DetailWriter("/home/ddd", detail_info)
        numpy_data = np.arange(12)
        with mock.patch('numpy.save'):
            with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                open_file.write = None
                detail_writer.write([1, 3, 2, 2], numpy_data, numpy_data)

    def test_write2(self):
        op = 'prob'
        detail_info = mock.Mock()
        detail_info.tensor_id = detail.TensorId(op, 'output', '3')
        detail_info.top_n = 3
        detail_info.ignore_result = True
        detail_info.my_output_ops = op
        detail_info.ground_truth_ops = op
        detail_info.detail_format = 'N C H W'
        attr = fusion_op.OpAttr([], '', False, 6)
        fusion_op_info = fusion_op.FusionOp(
            6, op, [], 'Left', ['/home/left/aaa.aaa.21.333333', '/home/left/aaa.aaa.21.433333'], attr)
        fusion_op_info.is_inner_node = mock.Mock(return_value=False)
        detail_info.get_detail_op = mock.Mock(return_value=(fusion_op_info, [fusion_op_info]))
        detail_info.make_detail_header = mock.Mock(
            return_value='Index,N C H W,LeftOp,RightOp,AbsoluteError,RelativeError')
        detail_writer = DetailWriter("/home/ddd", detail_info)
        numpy_data = np.arange(12)
        with mock.patch('numpy.save'):
            with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                open_file.write = None
                detail_writer.write([1, 3, 2, 2], numpy_data, numpy_data)

    def test_delete_old_detail_result_files1(self):
        tensor_id = detail.TensorId('conv1conv1_relu', 'output', '0')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_writer = DetailWriter("/home/demo", detail_info)
        with mock.patch("os.path.exists", return_value=False):
            detail_writer.delete_old_detail_result_files()

    def test_delete_old_detail_result_files2(self):
        tensor_id = detail.TensorId('conv1conv1_relu', 'output', '0')
        detail_info = detail.DetailInfo(tensor_id, 10, True, ConstManager.MAX_DETAIL_INFO_LINE_COUNT)
        detail_writer = DetailWriter("/home/demo", detail_info)
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch('dump_parse.mapping.read_mapping_file',
                            return_value={'1234566.csv': 'conv1conv1_relu_output_0_0.csv',
                                          '999999.csv': 'conv1conv1_relu_output_0_10000.csv',
                                          '6376427478658.csv': 'conv2_relu_output_0_10000.csv'}):
                with mock.patch("os.listdir", return_value=["conv1conv1_relu_output_0_summary.txt",
                                                            "'1234566.csv'", '999999.csv', "simple_op_mapping.csv"]):
                    with mock.patch("os.remove"):
                        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                            open_file.write = None
                            detail_writer.delete_old_detail_result_files()

    def test_new_top_n_obj1(self):
        top_n = TopN(False)
        self.assertEqual(top_n.is_bool, False)

    def test_new_top_n_obj2(self):
        top_n = TopN(True)
        self.assertEqual(top_n.is_bool, True)

    def test_bool_write1(self):
        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            DetailWriter._write_one_detail(0, open_file, ['1', '2'], 1, 2, 1, 2, True)

    def test_bool_write2(self):
        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
            DetailWriter._write_one_detail(0, open_file, ['1', '2', '2'], 1, 2, 1, 2, False)

    def test_get_top_n_res1(self):
        top_n = TopN(True)
        top_n._get_top_n_result([[0, 'name', True, False, 1, 2]], 0, 1)

    def test_get_top_n_res2(self):
        top_n = TopN(False)
        top_n._get_top_n_result([[0, 'name', 1, 2, 1, 2]], 0, 1)

    def test_cal_err1(self):
        my_output_data = np.array([1.1, 2, 3])
        ground_truth_data = np.array([11, 22, 33])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, False)
        self.assertEqual(9.9, absolute_error[0])
        self.assertEqual(20, absolute_error[1])
        self.assertEqual(30, absolute_error[2])
        self.assertEqual(0.9, relative_error[0])
        self.assertEqual(round(0.90909091, 3), round(relative_error[1], 3))
        self.assertEqual(round(0.90909091, 3), round(relative_error[2], 3))

    def test_cal_err2(self):
        my_output_data = np.array([True, False])
        ground_truth_data = np.array([False, True])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        detail_writer._total_num = 2
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, True)
        self.assertTrue(np.isnan(absolute_error[0]))
        self.assertTrue(np.isnan(absolute_error[1]))
        self.assertTrue(np.isnan(relative_error[0]))
        self.assertTrue(np.isnan(relative_error[1]))

    def test_get_err_top_n_index1(self):
        my_output_data = np.array([1.1, 2, 3])
        ground_truth_data = np.array([11, 22, 33])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, False)
        top_n = 2
        absolute_top_n_index, relative_top_n_index = detail_writer._get_error_top_n_index(absolute_error,
                                                                                          relative_error, top_n,
                                                                                          False)
        self.assertEqual([1, 2], absolute_top_n_index)
        self.assertEqual([1, 2], relative_top_n_index)

    def test_get_err_top_n_index2(self):
        my_output_data = np.array([True, False, True])
        ground_truth_data = np.array([False, True, True])
        detail_info = mock.Mock()
        detail_writer = DetailWriter('/home/demo/', detail_info)
        detail_writer._total_num = 2
        absolute_error, relative_error = detail_writer._cal_err(my_output_data, ground_truth_data, True)
        top_n = 2
        absolute_top_n_index, relative_top_n_index = detail_writer._get_error_top_n_index(absolute_error,
                                                                                          relative_error, top_n,
                                                                                          True)
        self.assertEqual([0, 1], absolute_top_n_index)
        self.assertEqual([0, 1], relative_top_n_index)
