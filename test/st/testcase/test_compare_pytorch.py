import multiprocessing
import time
import unittest

import numpy as np
import pytest
import sys
import utils
import compare_pytorch
import hdf5_parser
import h5py
from pytorch_dump_data import DataType
from unittest import mock
from compare_error import CompareError
import argparse


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

    def test_compare_pytorch_param1(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-op', 'Addmtest', '-p', 0]
        with pytest.raises(utils.CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch("os.path.isfile", return_value=True):
                            with mock.patch("hdf5_parser.Hdf5Parser.open_file",
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

    def test_compare_pytorch_param2(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-f', '/home/fusion_rule', '-p', '1']
        with pytest.raises(utils.CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('utils.check_output_path_valid',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        with mock.patch("os.path.isfile", return_value=True):
                            with mock.patch("hdf5_parser.Hdf5Parser.open_file",
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

    def test_compare_pytorch(self):

        def _create_group(file_handle, dataset_path):
            return file_handle.create_group(dataset_path)

        def _create_dataset(group, dataset_name, device_type, data_type=1):
            data_value = np.ones([2, 2], dtype='f')
            dataset = group.create_dataset(dataset_name, shape=(2, 2), dtype='f', data=data_value)
            dataset.attrs.create("DataType", data_type)
            dataset.attrs.create("DeviceType", device_type)
            dataset.attrs.create("FormatType", 3)
            dataset.attrs.create("Type", 0)
            dataset.attrs.create("Stride", (2, 1))

        def _create_dataset_abnormal(group, dataset_name, device_type):
            data_value = np.ones([2, 2], dtype='f')
            dataset = group.create_dataset(dataset_name, shape=(2, 2), dtype='f', data=data_value)
            dataset.attrs.create("DataType", 1)
            dataset.attrs.create("DeviceType", device_type)
            dataset.attrs.create("FormatType", 3)
            dataset.attrs.create("Type", 0)
            dataset.attrs.create("Stride", (2, 2))

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
                _create_dataset(group, "grad_0", 10, 6)
                group = _create_group(mydump_file_handle, "/NpuConv2D/13/input/grads/")
                _create_dataset(group, "grad_1", 10, 7)
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
                tf = './golden1_%s.h5' % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
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
                _create_dataset_abnormal(group, "grad_0", 1)
                group = _create_group(golden_file_handle, "/CudnnConv2D/23/input/grads/")
                _create_dataset_abnormal(group, "grad_1", 1)
                group = _create_group(golden_file_handle, "/CudnnConv2D/22/output/grads/")
                _create_dataset_abnormal(group, "result_0", 1)
                _create_dataset_abnormal(group, "result_1", 1)

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
                '/home/right.h5', '-p', '0']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args):
            args = parser.parse_args(sys.argv[1:])
            pytorch_compare = compare_pytorch.PytorchComparison(args)

        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('utils.check_output_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR),\
                    mock.patch('os.path.exists',return_value=True):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("hdf5_parser._open_h5py_file",
                                    side_effect=stub_open_file):
                        with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                            with mock.patch('utils.sort_result_file_by_index', return_value=None):
                                open_file.write = None
                                ret = pytorch_compare.compare()
                                all_orders = pytorch_compare.compare_data.get_all_orders()
                                pytorch_compare.compare_data.my_dump.need_compare_input = True
                                pytorch_compare._compare_in_one_process(all_orders, None)

        order = pytorch_compare.compare_data.my_dump.get_order_by_ext_opname("test_opt_name")
        self.assertEqual(order, 16)
        self.assertEqual(ret, 0)

    def test_hdf5_parser_open(self):
        h5py.File = mock.Mock
        hdf5_parser._open_h5py_file('./testfile')

    def test_hdf5_parser_get_dump_data_attr(self):
        parser = hdf5_parser.Hdf5Parser("/home/test1.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, [])
        with pytest.raises(utils.CompareError) as error:
            with mock.patch("hdf5_parser._open_h5py_file",
                            return_value=True):
                parser.file_handle = True
                has_attr, attr_value = parser.get_dump_data_attr("/test/path", "DataType")
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)

    def test_check_value(self):
        parser = hdf5_parser.Hdf5Parser("/home/test.h5", hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE, [])
        with pytest.raises(utils.CompareError) as error:
            tmp = [None] * 1000001
            parser._check_value(tmp)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_get_value(self):
        self.assertEqual(1, DataType.get_value('Float'))

    def test_get_item_location(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        row = ["CosineSimilarity", "MyDumpDataPath", "GoldenDumpDataPath"]
        with mock.patch('sys.argv', args):
            args = parser.parse_args(sys.argv[1:])
            pytorch_compare = compare_pytorch.PytorchComparison(args)
            index_result = pytorch_compare._get_item_location(row)

        self.assertEqual(index_result, [0, 1, 2])

    def test_filter_one_line(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        row = [0.93, "MyDumpDataPath", "GoldenDumpDataPath"]
        result_path = "/home/test"
        position = [0, 1, 2]
        with mock.patch('sys.argv', args):
            with mock.patch("csv.writer", return_value=None) as writer:
                with mock.patch('hdf5_parser.Hdf5Parser.get_dump_data',
                                side_effect=[np.array(np.arange(9)).reshape(3, 3),
                                             np.array(np.arange(9)).reshape(3, 3).T]):
                    with mock.patch('compare_pytorch.PytorchComparison._save_numpy_data', return_value=None):
                        args = parser.parse_args(sys.argv[1:])
                        pytorch_compare = compare_pytorch.PytorchComparison(args)
                        pytorch_compare._filter_one_line(result_path, row, writer, position)

    def test_filter_result_process(self):
        parser = self._construct_args()
        args = ['aaa.py', 'compare', '-m', '/home/left.h5', '-g',
                '/home/right.h5', '-p', '0']
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args):
            with mock.patch('os.open') as open_file, mock.patch('os.fdopen'):
                with mock.patch("csv.reader", return_value=[["CosineSimilarity", "MyDumpDataPath","GoldenDumpDataPath"],
                                                            [0.93, "/home/test/my1", "/home/test/gold1"]]):
                    with mock.patch('compare_pytorch.PytorchComparison._filter_one_line', return_value=None):
                        args = parser.parse_args(sys.argv[1:])
                        pytorch_compare = compare_pytorch.PytorchComparison(args)
                        pytorch_compare._filter_result_process("/home/test", open_file, "/home/filter")

