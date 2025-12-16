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

import csv
import sys
import time
import unittest
import multiprocessing
import h5py
import pytest
import numpy as np
from unittest import mock
import argparse

from pytorch_cmp import compare_pytorch
from vector_cmp.fusion_manager import compare_result
from pytorch_cmp.compare_pytorch import PytorchComparison
from cmp_utils.constant.compare_error import CompareError


class TestPytorchComparison(PytorchComparison):

    def do_compare_and_get_result(self: any, dataset_list: list,
                                  fusion_op_result: compare_result.FusionOpComResult,
                                  op_info: compare_result.PytorchOpInfo) -> list:
        return self._do_compare_and_get_result(dataset_list, fusion_op_result, op_info)


class TestUtilsMethods(unittest.TestCase):

    @staticmethod
    def _construct_args():
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(help='commands')
        compare_parser = subparsers.add_parser(
            'compare', help='Compare network or single op.')
        group = compare_parser.add_mutually_exclusive_group()
        compare_parser.add_argument(
            '-m', '--my_dump_path', dest='my_dump_path', default='',
            help='<Required> my dump path, the data compared with golden data',
            required=True)
        compare_parser.add_argument(
            '-g', '--golden_dump_path', dest='golden_dump_path', default='',
            help='<Required> the golden dump path', required=True)
        compare_parser.add_argument(
            '-f', '--fusion_rule_file', dest='fusion_rule_file', default='',
            help='<Optional> the fusion rule file path')
        compare_parser.add_argument(
            '-q', '--quant_fusion_rule_file', dest='quant_fusion_rule_file',
            default='', help='<Optional> the quant fusion rule file path')
        compare_parser.add_argument('-out', '--output', dest='output_path',
                                    default='', help='<Optional> the output path')
        compare_parser.add_argument('-op', '--op_name', dest='op_name',
                                    default=None,
                                    help='<Optional> operator name')
        group.add_argument(
            '-o', '--output_tensor', dest='output', default=None,
            help='<Optional> the index of output, takes effect only when'
                 ' the "-op" exists')
        group.add_argument(
            '-i', '--input_tensor', dest='input', default=None,
            help='<Optional> the index for input, takes effect only when'
                 ' the "-op" exists')
        compare_parser.add_argument(
            '-c', '--custom_script_path', dest='custom_script_path', default='',
            help='<Optional> the user-defined script path, '
                 'including format conversion')
        compare_parser.add_argument('-alg', '--algorithm', dest='algorithm', type=str, default="all",
                                    help='<Optional> comparison dimension selection')
        compare_parser.add_argument(
            '-a', '--algorithm_options', dest='algorithm_options', default='',
            help='<Optional> the arguments for each algorithm.')
        compare_parser.add_argument('-map', '--mapping', dest="mapping", action="store_true",
                                    help="<Optional> create mappings between my output operators"
                                         "and ground truth operators.",
                                    default=False, required=False)
        compare_parser.add_argument(
            '-v', '--version', dest='dump_version', choices=[1, 2], type=int,
            default=2,
            help='<Optional> the version of the dump file, '
                 '1 means the protobuf dump file, 2 means the binary dump file, '
                 'the default value is 2.')
        compare_parser.add_argument(
            '-p', '--post_process', dest='post_process', choices=[0, 1], type=int, default=0,
            help='<Optional> whether to extract the compare result, only pytorch is supported.')
        compare_parser.add_argument('-advisor', dest="advisor", action="store_true",
                                    help="<optional> Enable advisor after compare.", required=False)
        return parser

    def test_check_arguments_valid1(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0', '-op', 'Addmtest']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('cmp_utils.path_check.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('cmp_utils.path_check.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch("os.path.isfile", return_value=True):
                            with mock.patch("pytorch_cmp.hdf5_parser.Hdf5Parser.open_file",
                                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                                with mock.patch('os.open',
                                                side_effect=OSError) as open_file, \
                                        mock.patch('os.fdopen'):
                                    open_file.write = None
                                    args = parser.parse_args(sys.argv[1:])
                                    compare = compare_pytorch.PytorchComparison(args)
                                    compare.check_arguments_valid(args)
        self.assertEqual(err.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_arguments_valid2(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p','1', '-out', './']
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("pytorch_cmp.hdf5_parser.Hdf5Parser.open_file",
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('os.open',
                                        side_effect=OSError) as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            args = parser.parse_args(sys.argv[1:])
                            compare = compare_pytorch.PytorchComparison(args)
                            compare.check_arguments_valid(args)

    def test_compare_by_multi_process(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '1', '-out', './']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args):
            with mock.patch('cmp_utils.path_check.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("pytorch_cmp.hdf5_parser.Hdf5Parser.open_file",
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch('os.open',
                                        side_effect=OSError) as open_file, \
                                mock.patch('os.fdopen'):
                            args = parser.parse_args(sys.argv[1:])
                            compare = compare_pytorch.PytorchComparison(args)
                            compare.check_arguments_valid(args)
                            compare._compare_by_multi_process()

    def test_compare_in_one_process(self):

        def _create_group(file_handle, dataset_path):
            return file_handle.create_group(dataset_path)

        def _create_dataset(group, dataset_name, device_type):
            data_value = np.ones([2, 2], dtype='f')
            dataset = group.create_dataset(dataset_name, shape=(2, 2), dtype='f', data=data_value)
            dataset.attrs.create("DataType", 1)
            dataset.attrs.create("DeviceType", device_type)
            dataset.attrs.create("FormatType", 3)
            dataset.attrs.create("Type", 0)
            dataset.attrs.create("Stride", (2, 1))

        def stub_open_file(file_path, modle='r'):
            if file_path == '/home/left.h5':
                tf = './mydump_%s.h5' \
                    % time.strftime("%Y%m%d%H%M%S",
                                    time.localtime(time.time()))
                mydump_file_handle = h5py.File(tf, driver='core', mode='a', backing_store=False)
                # op AbsBackward
                group = _create_group(mydump_file_handle, "/AbsBackward/10/input/grads/")
                _create_dataset(group, "grad_0", 10)
                group = _create_group(mydump_file_handle, "/AbsBackward/11/input/grads/")
                _create_dataset(group, "grad_1", 10)
                group = _create_group(mydump_file_handle, "/AbsBackward/10/output/grads/")
                _create_dataset(group, "result_0", 10)
                _create_dataset(group, "result_1", 10)

                # op NpuConv2D
                group = _create_group(mydump_file_handle, "/NpuConv2D/12/input/grads/")
                _create_dataset(group, "grad_0", 10)
                group = _create_group(mydump_file_handle, "/NpuConv2D/13/input/grads/")
                _create_dataset(group, "grad_1", 10)
                group = _create_group(mydump_file_handle, "/NpuConv2D/12/output/grads/")
                _create_dataset(group, "result_0", 10)
                _create_dataset(group, "result_1", 10)

                # op Madd
                group = _create_group(mydump_file_handle, "/Madd/14/input/grads/")
                _create_dataset(group, "grad_0", 10)
                _create_dataset(group, "grad_1", 10)
                group = _create_group(mydump_file_handle, "/Madd/15/output/grads/")
                _create_dataset(group, "result_0", 10)
                _create_dataset(group, "result_1", 10)
                return mydump_file_handle
            if file_path == '/home/right.h5':
                tf = './golden_%s.h5' \
                    % time.strftime("%Y%m%d%H%M%S",
                                    time.localtime(time.time()))
                golden_file_handle = \
                    h5py.File(tf,  driver='core', mode='a', backing_store=False)
                # op AbsBackward
                group = _create_group(golden_file_handle, "/AbsBackward/20/input/grads/")
                _create_dataset(group, "grad_0", 1)
                group = _create_group(golden_file_handle, "/AbsBackward/21/input/grads/")
                _create_dataset(group, "grad_1", 1)
                group = _create_group(golden_file_handle, "/AbsBackward/20/output/grads/")
                _create_dataset(group, "result_0", 1)
                _create_dataset(group, "result_2", 1)

                # op CudnnConv2D
                group = _create_group(golden_file_handle, "/CudnnConv2D/22/input/grads/")
                _create_dataset(group, "grad_0", 1)
                group = _create_group(golden_file_handle, "/CudnnConv2D/23/input/grads/")
                _create_dataset(group, "grad_1", 1)
                group = _create_group(golden_file_handle, "/CudnnConv2D/22/output/grads/")
                _create_dataset(group, "result_0", 1)
                _create_dataset(group, "result_1", 1)

                # op Badd
                group = _create_group(golden_file_handle, "/Badd/24/input/grads/")
                _create_dataset(group, "grad_0", 1)
                _create_dataset(group, "grad_1", 1)
                group = _create_group(golden_file_handle, "/Badd/25/output/grads/")
                _create_dataset(group, "result_0", 1)
                _create_dataset(group, "result_1", 1)
                return golden_file_handle

        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5','-p', '1', '-out', '/home/demo']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            args = parser.parse_args(sys.argv[1:])
            pytorch_compare = compare_pytorch.PytorchComparison(args)

        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('cmp_utils.path_check.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR),\
                    mock.patch('os.path.exists', return_value=True):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("pytorch_cmp.hdf5_parser._open_h5py_file",
                                    side_effect=stub_open_file):
                        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                            with mock.patch('cmp_utils.utils.sort_result_file_by_index', return_value=None):
                                open_file.write = None
                                ret = pytorch_compare.compare()
                                all_orders = pytorch_compare.compare_data.get_all_orders()
                                pytorch_compare.compare_data.my_dump.need_compare_input = True
                                pytorch_compare._compare_in_one_process(all_orders, None)

        order = pytorch_compare.compare_data.my_dump.get_order_by_ext_opname("test_opt_name")
        self.assertEqual(order, 16)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

        with mock.patch('cmp_utils.path_check.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('cmp_utils.path_check.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.exists', return_value=True):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("pytorch_cmp.hdf5_parser._open_h5py_file",
                                    side_effect=stub_open_file):
                        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                            with mock.patch('pytorch_cmp.compare_pytorch.PytorchComparison.'
                                            '_get_compare_dump_data',
                                            side_effect=CompareError):
                                with mock.patch('os.path.getsize', return_value=None):
                                    open_file.write = None
                                    ret = pytorch_compare.compare()
                                    all_orders = pytorch_compare.compare_data.get_all_orders()
                                    pytorch_compare.compare_data.my_dump.need_compare_input = True
                                    pytorch_compare._compare_in_one_process(all_orders, None)

    def test_compare_one_op_exception1(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0', '-out', '/home/out']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            args = parser.parse_args(sys.argv[1:])
            pytorch_compare = compare_pytorch.PytorchComparison(args)
            with mock.patch('pytorch_cmp.pytorch_dump_data.CompareData.get_my_dump_datasets',
                            return_value=[]):
                pytorch_compare._compare_one_op(1, "Admm:2", mock.Mock)

    def test_get_item_location(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0', '-out', '/home/demo']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        row = ["CosineSimilarity", "MyDumpDataPath", "GoldenDumpDataPath"]
        with mock.patch('sys.argv', args), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            args = parser.parse_args(sys.argv[1:])
            pytorch_compare = compare_pytorch.PytorchComparison(args)
            index_result = pytorch_compare._get_item_location(row)

        self.assertEqual(index_result, [0, 1, 2])

    def test_filter_one_line(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0', '-out', '/home/demo']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        row = [0.93, "MyDumpDataPath", "GoldenDumpDataPath"]
        result_path = "/home/test"
        position = [0, 1, 2]
        with mock.patch('sys.argv', args), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch("csv.writer", return_value=None) as writer:
                with mock.patch('pytorch_cmp.hdf5_parser.Hdf5Parser.get_dump_data',
                                side_effect=[np.array(np.arange(9)).reshape(3, 3),
                                             np.array(np.arange(9)).reshape(3, 3).T]):
                    with mock.patch('pytorch_cmp.compare_pytorch.PytorchComparison._save_numpy_data',
                                    return_value=None):
                        args = parser.parse_args(sys.argv[1:])
                        pytorch_compare = compare_pytorch.PytorchComparison(args)
                        pytorch_compare._filter_one_line(result_path, row, writer, position)

    def test_filter_result_process(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0', '-out', '/home/demo']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        row = [0.93, "MyDumpDataPath", "GoldenDumpDataPath"]
        result_path = "/home/test"
        position = [0, 1, 2]
        with mock.patch('sys.argv', args), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                with mock.patch("csv.reader", return_value=[["CosineSimilarity", "MyDumpDataPath","GoldenDumpDataPath"],
                                                            [0.93, "/home/test/my1", "/home/test/gold1"]]):
                    with mock.patch('pytorch_cmp.compare_pytorch.PytorchComparison._filter_one_line',
                                    return_value=None):
                        args = parser.parse_args(sys.argv[1:])
                        pytorch_compare = compare_pytorch.PytorchComparison(args)
                        pytorch_compare._filter_result_process("/home/test", open_file, "/home/filter")

    def test_do_compare_and_get_result(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g', '/home/right.h5', '-out', '/home/out']
        my_dump_data = np.asarray([2.0213, -0.9118, -0.7277, -1.2509, -0.3222, -0.1876, 0.0432, 1.0614])
        golden_dump_data = np.asarray([2.0213, -0.9118, -0.7277, -1.2509, -0.3222, -0.1876, 0.0432, 1.0614])
        shape = my_dump_data.shape
        dataset_list = ['/addmm/3/input/mat1', '/addmm/10/input/mat1']
        op_info = compare_result.PytorchOpInfo(1, 'addmm', '/addmm/3/input/mat1', '/addmm/10/input/mat1')
        actual = ['1', 'addmm', '/addmm/3/input/mat1', '/addmm/10/input/mat1', 'float64', '[8]',
                  '1.000000', '0.000000', '0.000000', '0.000000', '0.000000', '(-0.034;1.017),(-0.034;1.017)',
                  '0.000000', '0.000000', '0.000000', '0.000000', '']
        with mock.patch('sys.argv', args[1:]), \
            mock.patch('cmp_utils.path_check.check_output_path_valid',
                return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('pytorch_cmp.compare_pytorch.PytorchComparison._get_compare_dump_data',
                            return_value=[my_dump_data, golden_dump_data, shape]):
                args = parser.parse_args(sys.argv)
                pytorch_compare = TestPytorchComparison(args)
                fusion_op_result = compare_result.FusionOpComResult(pytorch_compare.algorithm_manager)
                result_list = pytorch_compare.do_compare_and_get_result(dataset_list, fusion_op_result, op_info)
                self.assertEqual(result_list[0], actual)
