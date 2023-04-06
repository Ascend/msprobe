import unittest

import struct

import multiprocessing

import pytest
import numpy as np
import compare_vector
import json
from src.compare.dump_parse import dump
from src.compare.cmp_utils import utils
from src.compare.vector_cmp.fusion_manager import fusion_rule_parser
import dump_data_pb2 as DD
from src.compare.vector_cmp.fusion_manager import fusion_op
from unittest import mock
from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.vector_cmp.fusion_manager.fusion_op import OutputDesc, FusionOp, OpAttr
from src.compare.vector_cmp.range_manager.range_manager import RangeManager
from src.compare.vector_cmp.range_manager.range_mode import RangeMode
from src.compare.vector_cmp.range_manager.select_mode import SelectMode

class TestUtilsMethods(unittest.TestCase):
    def test_compare1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                main = compare_vector.VectorComparison()
                main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.access', return_value=False):
                            main = compare_vector.VectorComparison()
                            main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result.txt']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.access', return_value=False):
                            main = compare_vector.VectorComparison()
                            main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob', '-i', 'xxx']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result', '-d', 'prob%$']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare9(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.path.isdir', return_value=False), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare10(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare11(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir', side_effect=[['alg_CosineSimilarity.py'], ['xx.g.gg.xx']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare12(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'], ['aaa.aaa.0.1111111111111111',
                                                                              'aaa.0.1111111111111111.quant']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare13(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'], ['aaa.0.1111111111111111.pb']]), \
                        mock.patch('os.path.isdir', side_effect=[True, False]), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_compare14(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.pb'],
                                                ['aaa.0.1111111111111111.pb']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def test_compare15(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.aaa.0.1111111111111111',
                                                 'ccc.aaa.1.1111111111111111'],
                                                ['ccc.aaa.0.1111111111111111',
                                                 'convert_failed_file_list.txt']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare16(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.dump'],
                                                ['aaa.0.1111111111111111.dump']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare17(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-q', '/honm/b.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.quant'],
                                                ['aaa.0.1111111111111111.dump']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare18(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-q', '/honm/b.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.dump'],
                                                ['aaa.0.1111111111111111.quant']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare19(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.quant'],
                                                ['aaa.0.1111111111111111.dump']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare20(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.dump'],
                                                ['aaa.0.1111111111111111.pb']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare21(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.quant'],
                                                ['aaa.0.1111111111111111.dump']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare22(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-q',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.quant'],
                                                ['aaa.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_compare23(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.aaa.0.1111111111111111'],
                                                ['aaa.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'],
                                                ]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(read_data=b'01x03')):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def test_compare24(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result']
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['aaa.0.1111111111111111.dump'],
                                                ['aaa.0.1111111111111111.dump']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    main = compare_vector.VectorComparison()
                    main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare25(self):
        args = ['aaa.py', '-l', '/home/left', '-s', '/home/right', '-o',
                '/home/result.txt', '-d', 'prob', '-t', 'xxx']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                main = compare_vector.VectorComparison()
                main.compare()
        self.assertEqual(error.value.code,
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_vector1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['ReduceMeanD.conv1conv1_relu.6.4.1613727240764749'],
                                            ['input.4.1613727240736566.pb',
                                             'trans_Cast_1167.4.1613727241293941.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch('builtins.open', mock.mock_open(
                                read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    with mock.patch("os.path.realpath", return_value="/home/demo/a.json"):
                                        open_file.write = None
                                        main = compare_vector.VectorComparison(arguments)
                                        main.compare()
                                        key = \
                                            main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                                'conv1conv1_relu']
                                        ret, dump_match, result = main._compare_by_fusion_op(
                                            key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)

    def test_compare_vector2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                            ['data.0.1111111111111111.pb',
                                             'conv1_relu.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json_object()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                with mock.patch('os.open') as open_file, \
                                        mock.patch('os.fdopen'):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ccccc.0.1111111111111111'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ddddd.0.1111111111111111'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'conv1conv1_relu')
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ccccc.0.1111111111111111'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ddddd.0.1111111111111111'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'ccccc')
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, False)
        self.assertEqual(len(result), 1)

    def test_compare_vector5(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ccccc.0.1111111111111111'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ddddd.0.1111111111111111'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'ddddd')
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, False)
        self.assertEqual(len(result), 1)

    def test_compare_vector6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                            ['data.0.1111111111111111.pb',
                                             'conv1_relu.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 1, 1, 1]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                            ['data.0.1111111111111111.pb',
                                             'conv1_relu.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [0, 1, 1]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111'],
                                            ['data.0.1111111111111111.pb',
                                             'conv1_relu.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_json()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'conv1conv1_relu']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
        self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector_10(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        with mock.patch("compare_vector.VectorComparison._compare_by_multi_process", return_value=(0, False)):
            with mock.patch("src.compare.cmp_utils.utils.sort_result_file_by_index"):
                compare_vector_instance = compare_vector.VectorComparison(arguments)
                compare_vector_instance._compare_vector()
                compare_vector_instance.args["range"] = '[1,-1,2]'
                compare_vector_instance._compare_vector()

    def test_compare_vector12(self):
        args = ['aaa.py', '-l', '/home/left', '-s', '/home/right', '-o',
                '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = ""
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ccccc.0.1111111111111111'],
                                            ['xxxx.conv1conv1_relu.0.1111111111111111',
                                             'xxxx.ddddd.0.1111111111111111'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("json.load", return_value=self._make_json_object()):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_csv_content())):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                open_file.write = None
                                main = compare_vector.VectorComparison(arguments)
                                main.compare()
                                ret, dump_match, result = main._compare_by_fusion_op(
                                    'conv1conv1_relu')
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 2)

    def test_compare_vector13(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        with mock.patch("compare_vector.VectorComparison._compare_by_multi_process", return_value=(0, False)):
            with mock.patch("src.compare.cmp_utils.utils.sort_result_file_by_index"):
                compare_vector_instance = compare_vector.VectorComparison(arguments)
                compare_vector_instance._compare_vector()
                compare_vector_instance.args["select"] = '1,2,4'
                compare_vector_instance._compare_vector()

    def test_compare_vector_l1_fusion1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result.txt']
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/a.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = 'all'
        arguments.algorithm_options = ''
        arguments.range = None
        arguments.select = None
        multiprocessing.Manager = mock.Mock
        multiprocessing.Manager.RLock = mock.Mock
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        result = [[0, True, "data&message"]]
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.A1.0.1111111111111111',
                                             'xxxx.A2.0.1111111111111121',
                                             'xxxx.C1.0.1111111111111131',
                                             'xxxx.C2.0.1111111111111141'],
                                            ['A_relu.0.1111111111111111.pb',
                                             'data.0.1111111111111111.pb',
                                             'C.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch("json.load", return_value=self._make_L1_fusion_json_object()):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_csv_content())):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                with mock.patch("multiprocessing.pool.ApplyResult.get", return_value=result):
                                    open_file.write = None
                                    main = compare_vector.VectorComparison(arguments)
                                    main.compare()
                                    key = \
                                        main.compare_rule.fusion_info.op_name_to_fusion_op_name_map[
                                            'A1']
                                    ret, dump_match, result = main._compare_by_fusion_op(
                                        key)
        self.assertEqual(ret, 0)
        self.assertEqual(dump_match, True)
        self.assertEqual(len(result), 4)

    def test_has_range1(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_has_range2(self):
        args = ['--select']
        with mock.patch('sys.argv', args):
            self.assertTrue(RangeManager._has_cmd())

    def test_adjust_header(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            header = ['h1', 'h2', 'h3']
            RangeManager.adjust_header(header)
        self.assertEqual(['h1', 'OpSequence', 'h2', 'h3'], header)

    def test_adjust_data(self):
        args = ['-s']
        with mock.patch('sys.argv', args):
            data = ['h1', 'h2', 'h3']
            RangeManager.adjust_data(data, 1)
        self.assertEqual(['h1', '1', 'h2', 'h3'], data)

    def test_get_range_ops(self):
        compare_rule = mock.Mock
        compare_rule.fusion_info = mock.Mock
        op1 = fusion_op.FusionOp(0, 'a', [], 'data', [], fusion_op.OpAttr([], '', False, 1))
        op2 = fusion_op.FusionOp(0, 'b', [], 'data', [], fusion_op.OpAttr([], '', False, 2))
        op3 = fusion_op.FusionOp(0, 'c', [], 'data', [], fusion_op.OpAttr([], '', False, 3))
        op4 = fusion_op.FusionOp(0, 'd', [], 'data', [], fusion_op.OpAttr([], '', False, 4))
        op5 = fusion_op.FusionOp(0, 'e', [], 'data', [], fusion_op.OpAttr([], '', False, 5))
        op6 = fusion_op.FusionOp(0, 'f', [], 'data', [], fusion_op.OpAttr([], '', False, 6))
        op7 = fusion_op.FusionOp(0, 'g', [], 'data', [], fusion_op.OpAttr([], '', False, 7))
        op8 = fusion_op.FusionOp(0, 'h', [], 'data', [], fusion_op.OpAttr([], '', False, 8))
        op9 = fusion_op.FusionOp(0, 'i', [], 'data', [], fusion_op.OpAttr([], '', False, 9))
        op10 = fusion_op.FusionOp(0, 'j', [], 'data', [], fusion_op.OpAttr([], '', False, 10))
        fusion_list = [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10]
        compare_rule.fusion_info.op_list = fusion_list
        compare_rule.fusion_info.op_name_to_fusion_op_name_map = {"a": 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e',
                                                                  'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j'}
        range_manager = SelectMode("3,5,2")
        range_manager.check_input_valid(100)
        op_name_list = range_manager.get_all_ops(compare_rule)
        self.assertEqual(3, len(op_name_list))
        self.assertEqual(['b', 'c', 'e'], op_name_list)

    def test_parse_selected_op1(self):
        with pytest.raises(CompareError) as error:
            SelectMode("xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op2(self):
        with pytest.raises(CompareError) as error:
            SelectMode(",-1,xx")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op3(self):
        with pytest.raises(CompareError) as error:
            SelectMode("xx,,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op4(self):
        with pytest.raises(CompareError) as error:
            SelectMode(",xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_selected_op5(self):
        range_manager = SelectMode("1,2,3")
        self.assertEqual(range_manager.selected_op[0], 1)
        self.assertEqual(range_manager.selected_op[1], 2)
        self.assertEqual(range_manager.selected_op[2], 3)

    def test_parse_selected_op6(self):
        range_manager = SelectMode("10,5,7")
        self.assertEqual(range_manager.selected_op[0], 5)
        self.assertEqual(range_manager.selected_op[1], 7)
        self.assertEqual(range_manager.selected_op[2], 10)

        with pytest.raises(CompareError) as error:
            RangeMode("xx,,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range2(self):
        with pytest.raises(CompareError) as error:
            RangeMode(",xx,")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_parse_range3(self):
        range_manager = RangeMode(",,")
        self.assertEqual(range_manager.step, 1)
        self.assertEqual(range_manager.end, -1)
        self.assertEqual(range_manager.start, 1)

    def test_parse_range4(self):
        range_manager = RangeMode("5,10,2")
        self.assertEqual(range_manager.start, 5)
        self.assertEqual(range_manager.step, 2)
        self.assertEqual(range_manager.end, 10)

    def test_check_selected_valid1(self):
        with pytest.raises(CompareError) as error:
            range_manager = SelectMode("99")
            range_manager.check_input_valid(10)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare_detail1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'aaa']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['ccc.aaa.0.1111111111111111'],
                                            ['aaa.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        main = compare_vector.VectorComparison()
                        ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare_detail2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.path.getsize', return_value=100), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        main = compare_vector.VectorComparison()
                        ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_compare_detail3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'dynamic_const_471']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'],
                                                ['']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu',
                '-i', '10']
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail5(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-o',
                '/home/result', '-d', 'data']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['ccc.data.0.1111111111111111'],
                                            ['ccc.data.0.1111111111111111'],
                                            ['convert_NC1HWC0_to_NCHW.py']]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True):
                with mock.patch('builtins.open',
                                mock.mock_open(
                                    read_data=self._make_input_json().encode(
                                        'utf-8'))):
                    main = compare_vector.VectorComparison()
                    ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare_detail6(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data',
                '-i', '10']
        dump_data = DD.DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail7(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data']
        dump_data = DD.DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_WRITE_FILE_ERROR)

    def test_compare_detail8(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'data', '-t',
                'input']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['ccc.data.0.1111111111111111'],
                                                ['data.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_input_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail9(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu',
                '-t', 'input']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['ccc.conv1conv1_relu.0.1111111111111111'],
                                            ['data.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py'], []]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail10(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'conv1conv1_relu']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['conv1_relu.0.1111111111111111.dump'],
                                                ['data.0.1111111111111111.pb',
                                                 'conv1_relu.0.1111111111111111.pb'],
                                                [], ['convert_NC1HWC0_to_NCHW.py']]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_input_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_compare_detail_l1_fusion1(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'A1']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['xxxx.A1.0.1111111111111111',
                                                 'xxxx.A2.0.1111111111111121',
                                                 'xxxx.C1.0.1111111111111131',
                                                 'xxxx.C2.0.1111111111111141'],
                                                ['A_relu.0.1111111111111111.pb',
                                                 'data.0.1111111111111111.pb',
                                                 'C.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_L1_fusion_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_detail_l1_fusion2(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'A1', '-t', 'input']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.A1.0.1111111111111111',
                                             'xxxx.A2.0.1111111111111121',
                                             'xxxx.C1.0.1111111111111131',
                                             'xxxx.C2.0.1111111111111141'],
                                            ['A_relu.0.1111111111111111.pb',
                                             'data.0.1111111111111111.pb',
                                             'C.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py'], []]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_l1_fusion3(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'B1']
        dump_data = DD.DumpData()
        dump_data.input.append(
            self._make_op_input(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with pytest.raises(CompareError) as err:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.exists', return_value=True), \
                        mock.patch('os.access', return_value=True), \
                        mock.patch('os.remove'), \
                        mock.patch('os.listdir',
                                   side_effect=[['alg_CosineSimilarity.py'],
                                                ['xxxx.A1.0.1111111111111111',
                                                 'xxxx.A2.0.1111111111111121',
                                                 'xxxx.C1.0.1111111111111131',
                                                 'xxxx.C2.0.1111111111111141'],
                                                ['A_relu.0.1111111111111111.pb',
                                                 'data.0.1111111111111111.pb',
                                                 'C.0.1111111111111111.pb'],
                                                ['convert_NC1HWC0_to_NCHW.py'], []]), \
                        mock.patch('os.path.isdir', return_value=True), \
                        mock.patch('os.path.isfile', return_value=True), \
                        mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                    with mock.patch("os.path.getsize", return_value=100):
                        with mock.patch('builtins.open',
                                        mock.mock_open(
                                            read_data=self._make_L1_fusion_json().encode(
                                                'utf-8'))):
                            with mock.patch('os.open') as open_file, \
                                    mock.patch('os.fdopen'):
                                open_file.write = None
                                main = compare_vector.VectorComparison()
                                ret = main.compare()
        self.assertEqual(err.value.args[0], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_detail_l1_fusion4(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C1']
        dump_data = DD.DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.A1.0.1111111111111111',
                                             'xxxx.A2.0.1111111111111121',
                                             'xxxx.C1.0.1111111111111131',
                                             'xxxx.C2.0.1111111111111141'],
                                            ['A_relu.0.1111111111111111.pb',
                                             'data.0.1111111111111111.pb',
                                             'C.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py'], []]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_l1_fusion5(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right', '-f',
                '/home/a.json', '-o', '/home/result', '-d', 'C2']
        dump_data = DD.DumpData()
        dump_data.output.append(
            self._make_op_output(DD.FORMAT_NCHW, [1, 3, 4, 4]))
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True), \
                    mock.patch('os.access', return_value=True), \
                    mock.patch('os.remove'), \
                    mock.patch('os.listdir',
                               side_effect=[['alg_CosineSimilarity.py'],
                                            ['xxxx.A1.0.1111111111111111',
                                             'xxxx.A2.0.1111111111111121',
                                             'xxxx.C1.0.1111111111111131',
                                             'xxxx.C2.0.1111111111111141'],
                                            ['A_relu.0.1111111111111111.pb',
                                             'data.0.1111111111111111.pb',
                                             'C.0.1111111111111111.pb'],
                                            ['convert_NC1HWC0_to_NCHW.py'],
                                            []]), \
                    mock.patch('os.path.isdir', return_value=True), \
                    mock.patch('os.path.isfile', return_value=True), \
                    mock.patch('src.compare.cmp_utils.utils.parse_dump_file', return_value=dump_data):
                with mock.patch("os.path.getsize", return_value=100):
                    with mock.patch('builtins.open',
                                    mock.mock_open(
                                        read_data=self._make_L1_fusion_json().encode(
                                            'utf-8'))):
                        with mock.patch('os.open') as open_file, \
                                mock.patch('os.fdopen'):
                            open_file.write = None
                            main = compare_vector.VectorComparison()
                            ret = main.compare()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_make_mapping_table_by_op_name(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = "all"
        arguments.algorithm_options = ""
        arguments.range = None
        arguments.select = None
        compare = compare_vector.VectorComparison(arguments)
        compare.compare_rule = mock.Mock
        compare.compare_rule.fusion_info = mock.Mock
        compare.compare_data = mock.Mock
        compare.compare_data.left_dump_info = mock.Mock
        compare.compare_data.left_dump_info.type = None
        compare.compare_data.left_dump_info.get_op_dump_file = mock.Mock(side_effect=ValueError)
        op_name = "output_ids"
        op_id = 0
        original_op_names = ["input_ids"]
        input_list = []
        op_type = "Data"
        l1_fusion_no = ""
        is_multi_op = False
        origin_name = "input_ids"
        origin_format = "NHWC"
        origin_shape = [1, 128]
        origin_output_index = 0
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 1)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare.compare_rule.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare._make_mapping_table_by_op_name(["demo"])

    def test_merge_close_and_open_fusion_rule(self):
        close_fusion_rule = mock.Mock()
        close_op = FusionOp(0, 'CIn', ['a:0'], 'Relu', [OutputDesc('C', 1, 'NCHW', [1, 3, 4, 4])],
                            OpAttr(['C'], '1', True, 0))
        close_fusion_rule.fusion_op_name_to_op_map = {'CIn': [close_op]}
        close_fusion_rule.op_name_to_fusion_op_name_map = {'CIn': 'CIn'}
        close_fusion_rule.get_origin_name_to_op_name_map = mock.Mock(return_value={'C': 'CIn'})
        open_fusion_rule = mock.Mock()
        open_op = FusionOp(0, 'BCIn', ['a:0'], 'Relu', [OutputDesc('C', 1, 'NCHW', [1, 3, 4, 4])],
                           OpAttr(['B', 'C'], '1', True, 0))
        open_fusion_rule.fusion_op_name_to_op_map = {'CIn': [open_op]}
        open_fusion_rule.op_name_to_fusion_op_name_map = {'CIn': 'CIn'}
        fusion_rule_parser.merge_close_and_open_fusion_rule(open_fusion_rule, close_fusion_rule)
        self.assertEqual(['CIn'], open_op.attr.original_op_names)
        self.assertEqual('CIn', open_op.output_desc[0].origin_name)
        self.assertEqual(0, open_op.output_desc[0].origin_output_index)

    def test_make_mapping_table_by_op_name1(self):
        arguments = mock.Mock()
        arguments.fusion_rule_file = "/home/b.json"
        arguments.quant_fusion_rule_file = ""
        arguments.close_fusion_rule_file = ""
        arguments.my_dump_path = "/home/demo"
        arguments.golden_dump_path = "/home/dt"
        arguments.dump_version = 1
        arguments.op_name = ""
        arguments.output_path = "/home/de"
        arguments.custom_script_path = ""
        arguments.algorithm = "all"
        arguments.algorithm_options = ""
        arguments.range = None
        arguments.select = None
        compare = compare_vector.VectorComparison(arguments)
        compare.compare_rule = mock.Mock
        compare.compare_rule.fusion_info = mock.Mock
        op_name = "input_ids"
        op_id = 0
        original_op_names = ["input_ids"]
        input_list = []
        op_type = "Data"
        l1_fusion_no = ""
        is_multi_op = False
        origin_name = "input_ids"
        origin_format = "NHWC"
        origin_shape = [1, 128]
        origin_output_index = 0
        output_desc = OutputDesc(origin_name, origin_output_index, origin_format, origin_shape)
        attr = OpAttr(original_op_names, l1_fusion_no, is_multi_op, 1)
        fusion_list = [FusionOp(op_id, op_name, input_list, op_type, output_desc, attr)]
        compare.compare_rule.fusion_info.fusion_op_name_to_op_map = {"demo": fusion_list}
        compare._make_mapping_table_by_op_name(["demo"])

    @staticmethod
    def _make_op_output(dd_format, shape):
        op_output = DD.OpOutput()
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
        op_input = DD.OpInput()
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
    def _make_offline_dump_info():
        dump_info = dump.DumpInfo('/home')
        dump_info.type = dump.DumpType.Offline
        dump_info.op_name_to_file_map = {
            'aabb': ['aaa.aabb.0.1111111111111111',
                     'aaa.aabb.10.1111111111111111'],
            'data': ['Input.data.0.1111111111111111'],
            'conv1conv1_relu': ['aaa.conv1conv1_relu.0.1111111111111111'],
            'conv1conv1_relu2': ['aaa.conv1conv1_relu2.0.1111111111111111'],
            'res2s_branch1': ['aaa.res2s_branch1.0.1111111111111111'],
            'conv1_quant': ['quant.conv1_quant.0.1111111111111111'],
            'pool5': ['nnnnn.pool5.0.1111111111111111'],
            'cc_dd': ['aaa.cc_dd.0.1111111111111111']}
        return dump_info

    @staticmethod
    def _make_sim_dump_info():
        dump_info = dump.DumpInfo('/home')
        dump_info.type = dump.DumpType.Simulation
        dump_info.op_name_to_file_map = {
            'data.0': ['data.0.1111111111111111.dump'],
            'a.0': ['a.0.1111111111111111.dump'],
            'conv1_relu.0': ['conv1_relu.0.1111111111111111.dump'],
            'conv1_relu.1': ['conv1_relu.1.1111111111111111.dump'],
            'res2s_branch1_relu.0': [
                'res2s_branch1_relu.0.1111111111111111.dump'],
            'prob.2': ['prob.2.1111111111111111.dump'],
            'prob.1': ['prob.1.1111111111111111.dump']}
        return dump_info

    @staticmethod
    def _make_quant_dump_info():
        dump_info = dump.DumpInfo('/home')
        dump_info.type = dump.DumpType.Quant
        dump_info.op_name_to_file_map = {
            'data.0': ['data.0.1111111111111111.quant'],
            'a.0': ['a.0.1111111111111111.quant'],
            'prob.2': ['prob.2.1111111111111111.quant'],
            'prob.0': ['prob.0.1111111111111111.quant'],
            'res2s_branch1_relu.0': [
                'res2s_branch1_relu.0.1111111111111111.quant'],
            'prob.1': ['prob.1.1111111111111111.quant']}
        return dump_info

    @staticmethod
    def _make_standard_dump_info():
        dump_info = dump.DumpInfo('/home')
        dump_info.type = dump.DumpType.Standard
        dump_info.op_name_to_file_map = {
            'data.0': ['data.0.1111111111111111.pb'],
            'a.0': ['a.0.1111111111111111.pb'],
            'prob.2': ['prob.2.1111111111111111.pb'],
            'conv1_relu.0': ['conv1_relu.0.1111111111111111.pb'],
            'conv1_relu.1': ['conv1_relu.1.1111111111111111.pb'],
            'res2s_branch1_relu.0': [
                'res2s_branch1_relu.0.1111111111111111.pb'],
            'prob.1': ['prob.1.1111111111111111.pb']}
        return dump_info

    @staticmethod
    def _make_json():
        return {'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'conv1conv1_relu2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 1}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'res2s_branch1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["res2s_branch1",
                                                "res2s_branch1_relu"]}}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'res2s_branch1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_432', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'prob', 'type': 'PORR'},
                 {'name': 'conv1_quant', 'type': 'AscendQuant'},
                 {'name': 'pool5', 'type': 'pool'},
                 ],
             },
            {
                'name': 'merge2', 'op': []
            }
        ]}

    @staticmethod
    def _make_input_json():
        return json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data',
                  'type': 'Input',
                  "attr": [
                      {"key": "xxx",
                       "value": 'xxx'},
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': 'xxx',
                           'value': 'xxx'},
                          {'key': 'origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_471', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'dynamic_const_387', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },

                 ],
             },
        ]})

    @staticmethod
    def _make_csv_content():
        content = 'Index,LeftOp,RightOp,TensorIndex,MaxAbsoluteError,MaxAbsoluteError,CompareFailReason'
        return content

    @staticmethod
    def _make_json_object():
        json_object = {'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data',
                  'type': 'Input',
                  "attr": [
                      {"key": "xxx",
                       "value": 'xxx'},
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': 'xxx',
                           'value': 'xxx'},
                          {'key': 'origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },
                 {'name': 'dynamic_const_471', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'dynamic_const_387', 'type': 'Const', "attr": [
                     {"key": "_datadump_original_op_names",
                      "value": {"list": {"val_type": 1}}}]},
                 {'name': 'conv1conv1_relu',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["scale_conv1", "conv1",
                                                "bn_conv1", "conv1_relu"]}}}
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'conv1_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                          {'key': 'origin_shape',
                           'value': {"list": {"val_type": 1,
                                              "i": [1, 3, 4, 4]}}},
                      ]},
                  ]
                  },

                 ],
             },
        ]}
        return json_object

    @staticmethod
    def _make_L1_fusion_json():
        return json.dumps({'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data', 'type': 'input'},
                 {'name': 'A1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'A2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },

                 {'name': 'B1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'B2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 ],
             },

        ]})

    @staticmethod
    def _make_L1_fusion_json_object():
        return {'name': 'resnet50', 'graph': [
            {'name': 'merge1', 'op':
                [{'name': 'data', 'type': 'input'},
                 {'name': 'A1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'A2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["A_conv1", "A_relu"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "data:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'A_relu'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },

                 {'name': 'B1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'B2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["B"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "A2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'B'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C1',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B1:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 {'name': 'C2',
                  'type': 'Relu',
                  "attr": [
                      {"key": "_datadump_original_op_names",
                       "value": {"list": {"val_type": 1,
                                          "s": ["C"]}}},
                      {"key": "_L1_fusion_sub_graph_no",
                       "value": {'s': '1'}},
                  ],
                  "input": [
                      "B2:0",
                      "dynamic_const_471:0",
                      "dynamic_const_387:0"
                  ],
                  'output_desc': [
                      {'attr': [
                          {'key': '_datadump_origin_name',
                           'value': {'s': 'C'}},
                          {'key': '_datadump_origin_output_index',
                           'value': {'i': 0}},
                          {'key': '_datadump_origin_format',
                           'value': {'s': 'NCHW'}},
                      ]},
                  ]
                  },
                 ],
             },

        ]}


if __name__ == '__main__':
    unittest.main()
