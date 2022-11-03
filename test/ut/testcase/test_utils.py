import unittest

import struct

import csv
import pytest
import numpy as np
import utils
import log
import fusion_op
import dump_data_pb2 as DD
from compare_error import CompareError
from progress import Progress
from unittest import mock
from google.protobuf.message import DecodeError


class TestUtilsMethods(unittest.TestCase):

    def test_print_info_log(self):
        log.print_info_log('test info log')

    def test_print_error_log(self):
        log.print_error_log('test error log')

    def test_print_warn_log(self):
        log.print_warn_log('test warn log')

    @mock.patch("common.get_dtype_by_data_type")
    def test_deserialize_dump_data_to_array(self, mock_common):
        mock_common.return_value = np.uint8
        op_output = mock.Mock()
        op_output.data = b'\x01\x02'
        op_output.shape.dim = [1]
        numpy_data = utils.deserialize_dump_data_to_array(op_output)
        self.assertEqual(len(numpy_data), 2)

    def test_space_to_comma(self):
        value = 'a,b,c d,e 1.0 0.1 (3.0,4.0);(5.3,6.5)'
        new_value = utils.space_to_comma(value)
        self.assertEqual(new_value, 'a b c,d e,1.0,0.1,(3.0 4.0);(5.3 6.5)')

    def test_check_name_valid1(self):
        ret = utils.check_name_valid('')
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_name_valid2(self):
        ret = utils.check_name_valid('xxx$%^&**&^&')
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_name_valid3(self):
        ret = utils.check_name_valid('prob')
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_read_numpy_file1(self):
        dump_data = np.arange(2)
        with mock.patch('numpy.loadtxt', return_value=dump_data):
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                ret = utils.read_numpy_file('/home/a.txt')
        self.assertEqual(len(dump_data), len(ret))
        self.assertEqual(dump_data[0], ret[0])
        self.assertEqual(dump_data[1], ret[1])

    def test_read_numpy_file2(self):
        with pytest.raises(utils.CompareError) as error:
            utils.read_numpy_file('')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_read_numpy_file3(self):
        dump_data = np.arange(2)
        with mock.patch('numpy.load', return_value=dump_data):
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                ret = utils.read_numpy_file('/home/a.bin')
        self.assertEqual(len(dump_data), len(ret))
        self.assertEqual(dump_data[0], ret[0])
        self.assertEqual(dump_data[1], ret[1])

    def test_read_numpy_file4(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.load', side_effect=ValueError):
                    utils.read_numpy_file('a.bin')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file5(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.load', side_effect=UnicodeDecodeError):
                    utils.read_numpy_file('a.bin')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file6(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                    mock.patch('os.path.getsize', return_value=12):
                with mock.patch('numpy.loadtxt', side_effect=UnicodeDecodeError):
                    utils.read_numpy_file('a.txt')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_read_numpy_file7(self):
        dump_data = np.arange(2)
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('numpy.loadtxt', return_value=dump_data):
                with mock.patch('utils.check_path_valid',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                        mock.patch('os.path.getsize', return_value=0):
                    ret = utils.read_numpy_file('/home/a.txt')
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_convert_shape_to_string(self):
        shape_str = utils.convert_shape_to_string([1, 3, 224, 224])
        self.assertEqual(shape_str, '(1, 3, 224, 224)')

    def test_get_string_from_list1(self):
        shape_str = utils.get_string_from_list([1, 3, 224, 224], 'x')
        self.assertEqual(shape_str, '1x3x224x224')

    def test_get_string_from_list2(self):
        shape_str = utils.get_string_from_list(["1", "3", "224", "224"], 'x')
        self.assertEqual(shape_str, '1x3x224x224')

    def test_format_value(self):
        value = 2.3658412
        new_value = utils.format_value(value)
        self.assertEqual(new_value, '2.365841')

    def test_check_path_valid1(self):
        ret = utils.check_path_valid('', True)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_path_valid2(self):
        ret = utils.check_path_valid('/home/7%##3', True)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_path_valid3(self):
        with mock.patch('os.path.exists', return_value=False):
            ret = utils.check_path_valid('/home/result.txt', False)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid4(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=False):
                ret = utils.check_path_valid('/home/result.txt', False)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid5(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                ret = utils.check_path_valid('/home/result', True, True)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_check_path_valid6(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.path.isfile', return_value=False):
                    ret = utils.check_path_valid(
                        '/home/result.txt', True, False, utils.PathType.File)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid7(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', return_value=True):
                with mock.patch('os.path.isdir', return_value=False):
                    ret = utils.check_path_valid(
                        '/home/result.txt', True, False,
                        utils.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_path_valid8(self):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.access', side_effect=[True, False]):
                ret = utils.check_path_valid(
                    '/home/result', True, True, utils.PathType.Directory)
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_parse_dump_file1(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_INVALID_PATH_ERROR):
                utils.parse_dump_file('/home', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_parse_dump_file2(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('builtins.open', side_effect=IOError):
                    utils.parse_dump_file('/home', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file3(self):
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize', return_value=0):
                    utils.parse_dump_file('/home/a.dump', 2)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file4(self):
        dump_data = DD.DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with mock.patch('utils.check_path_valid',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.path.getsize', return_value=len(dump_data_ser)):
                with mock.patch('builtins.open',
                                mock.mock_open(read_data=dump_data_ser)):
                    dump_data = utils.parse_dump_file('/home/a.dump', 1)
        self.assertEqual(dump_data.output_data[0].data_type, DD.DT_FLOAT16)
        data_byte = dump_data.output_data[0].data.tobytes()
        self.assertEqual(len( data_byte), 48)

    def test_parse_dump_file5(self):
        dump_data = DD.DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.path.getsize', return_value=len(dump_data_ser)):
                with mock.patch('builtins.open', mock.mock_open(read_data=dump_data_ser)):
                    dump_data = utils.parse_dump_file('/home/a.dump', 1)
        self.assertEqual(dump_data.output_data[0].size, 48)

    def test_parse_dump_file6(self):
        dump_data = DD.DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize', return_value=len(dump_data_ser)):
                    with mock.patch('builtins.open', mock.mock_open(read_data=dump_data_ser)):
                        with mock.patch('dump_data_pb2.DumpData.ParseFromString', return_value=1000):
                            utils.parse_dump_file('/home/a.dump', 0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file7(self):
        dump_data = DD.DumpData()
        output = dump_data.output.add()
        self._set_op_output(output, DD.FORMAT_NC1HWC0, [1, 3, 2, 2, 2])
        dump_data_ser = dump_data.SerializeToString()
        with pytest.raises(utils.CompareError) as error:
            with mock.patch('utils.check_path_valid',
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                with mock.patch('os.path.getsize',
                                return_value=len(dump_data_ser)):
                    with mock.patch('builtins.open',
                                    mock.mock_open(read_data=dump_data_ser)):
                        with mock.patch(
                                'dump_data_pb2.DumpData.ParseFromString',
                                side_effect=DecodeError):
                            utils.parse_dump_file('/home/a.dump', 0)
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_parse_dump_file8(self):
        with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch('os.path.getsize', return_value=1210):
                with mock.patch('numpy.load', return_value=np.ones([1, 3, 2, 2])):
                    dump_data = utils.parse_dump_file('/home/a.npy', 0)
        self.assertEqual(dump_data.output_data[0].shape[1], 3)

    def test_print_progress1(self):
        with mock.patch('time.time', return_value=3):
            progress = Progress(10)
            progress.update_progress()
            progress.print_progress(52.25)

    def test_print_progress2(self):
        with mock.patch('time.time', return_value=3):
            progress = Progress(10)
            progress.update_progress()
            progress.print_progress()

    def test_print_deprecated_warning(self):
        log.print_deprecated_warning("/home/a.py")

    def test_read_mapping_file1(self):
        mapping_file_path = "/home/demo/1.csv"
        csv_object = [[13243254435, "/home/demo/Add.0.1223242.npy"]]
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch("csv.reader", return_value=csv_object):
                    hash_map = utils.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {13243254435: "/home/demo/Add.0.1223242.npy"})

    def test_read_mapping_file2(self):
        mapping_file_path = "/home/demo/1.csv"
        csv_object = [[13243254435]]
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch("csv.reader", return_value=csv_object):
                    hash_map = utils.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {})

    def test_read_mapping_file3(self):
        mapping_file_path = "/home/demo/1.csv"
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data="demo")):
                with mock.patch("csv.reader", side_effect=IOError):
                    hash_map = utils.read_mapping_file(mapping_file_path)
        self.assertEqual(hash_map, {})

    def test_sort_result_file_by_index1(self):
        result_reader = [['0', 'input_ids', 'input_ids', 'input_ids:output:0', '1.000000', '0.000000', '0.000000',
                          '0.000000', '0.000000', '(1171.945;2293.594),(1171.945;2293.594)', '']]
        result_file = "/home/demo/1.csv"
        result_file_content = 'Index,LeftOp,RightOp,TensorIndex,MaxAbsoluteError,MaxAbsoluteError,CompareFailReason'
        with mock.patch('builtins.open', mock.mock_open(read_data=result_file_content)):
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                csv.reader = mock.Mock(return_value=result_reader)
                utils.sort_result_file_by_index(result_file)

    def test_sort_result_file_by_index2(self):
        result_file = "/home/demo/2.csv"
        result_file_content = 'Index LeftOp RightOp TensorIndex MaxAbsoluteError MaxAbsoluteError CompareFailReason\n'
        result = '0 input_ids input_ids input_ids:output:0 1.000000 0.000000 0.000000 0.000000 0.000000' \
                 ' \'(1171.945;2293.594) (1171.945;2293.594)\''
        with mock.patch('builtins.open', mock.mock_open(read_data=result_file_content + result)):
            with mock.patch('os.open'), mock.patch("os.fdopen"):
                utils.sort_result_file_by_index(result_file, False)

    def test_sort_result_file_by_index3(self):
        result_file = "/home/demo/3.csv"
        with mock.patch('builtins.open', side_effect=IOError):
            utils.sort_result_file_by_index(result_file)

    def test_get_shape_type1(self):
        shape_dim_array = [0]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils.ShapeType.Tensor)

    def test_get_shape_type2(self):
        shape_dim_array = [1, 1, 1]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils.ShapeType.Scalar)

    def test_get_shape_type3(self):
        shape_dim_array = [1, 3, 2]
        result = utils.get_shape_type(shape_dim_array)
        self.assertEqual(result, utils.ShapeType.Tensor)

    def test_get_path_list_for_str1(self):
        with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            path_list = utils.get_path_list_for_str('/home/a.bin')
        self.assertEqual(1, len(path_list))

    def test_get_path_list_for_str2(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
                utils.get_path_list_for_str('/home/a.bin')
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_UNKNOWN_ERROR)

    def test_get_path_list_for_str3(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
                utils.get_path_list_for_str('/home/a.bin,/home/b.bin')
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_get_path_list_for_str4(self):
        with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
            path_list = utils.get_path_list_for_str('/home/a.bin,/home/b.bin')
        self.assertEqual(2, len(path_list))

    def test_make_msnpy_file_name1(self):
        file_path = '/home/aa.bin'
        file_name = utils.make_msnpy_file_name(file_path, 'xx', 'input', 0, DD.FORMAT_NC1HWC0)
        self.assertEqual('aa.bin.input.0.NC1HWC0.npy', file_name)

    def test_make_msnpy_file_name2(self):
        file_path = '/home/axxx/FusionOp_Conv2DBackpropInput_ReluGradV2.Gradients_Default_network-TrainOneStepCell' \
                    '_network-WithLossCell__backbone-ResNet_gradReLU_FusionOp_Conv2DBackpropInput_' \
                    'ReluGradV2-op2739.31.9.1630556346146690'
        file_name = utils.make_msnpy_file_name(file_path,
                                               'ReluGradV2-op2739',
                                               'input', 0, DD.FORMAT_NC1HWC0)
        self.assertEqual(
            'FusionOp_Conv2DBackpropInput_ReluGradV2.ReluGradV2-op2739.31.9.1630556346146690.input.0.NC1HWC0.npy',
            file_name)

    @staticmethod
    def _make_fusion_op():
        attr = fusion_op.OpAttr(['conv1', 'conv1_relu'], '', False, 12)
        output_desc_list = []
        output_desc = fusion_op.OutputDesc('conv1_relu', 0, 'NCHW',
                                           [1, 3, 224, 224])
        output_desc_list.append(output_desc)
        return fusion_op.FusionOp(12, 'conv1conv1_relu', ['a:0,b:0'], 'Relu', output_desc_list, attr)

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = DD.OpOutput()
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('e' * length, *origin_numpy)
        return op_output

    @staticmethod
    def _set_op_output(op_output, dd_format, shape):
        op_output.data_type = DD.DT_FLOAT16
        op_output.format = dd_format
        length = 1
        for dim in shape:
            op_output.shape.dim.append(dim)
            length *= dim
        data_list = np.arange(length)
        origin_numpy = np.array(data_list, np.float16)
        op_output.data = struct.pack('e' * length, *origin_numpy)
        op_output.size = len(op_output.data)
        return op_output


if __name__ == '__main__':
    unittest.main()
