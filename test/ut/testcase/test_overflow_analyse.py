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

import argparse
import unittest
from unittest import mock

import numpy as np
import pytest

from cmp_utils import file_utils
from overflow import overflow_analyse
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_check_argument(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        with mock.patch("cmp_utils.path_check.check_path_valid",
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch("cmp_utils.path_check.check_output_path_valid",
                            return_value=CompareError.MSACCUCMP_NONE_ERROR):
                ret = decode.check_argument(args)
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

        with mock.patch("cmp_utils.path_check.check_path_valid",
                        return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
            ret = decode.check_argument(args)
        self.assertEqual(ret, CompareError.MSACCUCMP_UNKNOWN_ERROR)

        with mock.patch("cmp_utils.path_check.check_path_valid",
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            with mock.patch("cmp_utils.path_check.check_output_path_valid",
                            return_value=CompareError.MSACCUCMP_UNKNOWN_ERROR):
                ret = decode.check_argument(args)
        self.assertEqual(ret, CompareError.MSACCUCMP_UNKNOWN_ERROR)

    def test_output_path_argument_when_is_softlink_then_raise_error(self):
        args = mock.Mock()
        args.dump_path = ''
        args.output_path = '/home/result'
        with self.assertRaises(CompareError) as error:
            with mock.patch("os.path.islink", return_value=True):
                overflow_tester = overflow_analyse.OverflowAnalyse(args)
        self.assertEqual(str(error.exception), "3")

    def test_find_all_debug_files(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        file_desc = {
            "file_path": "/test/Opdebug",
            "timestamp": int("161333160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }

        dump_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, {})
        with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_dump_files',
                        return_value=[dump_file_desc]):
            ret = decode._find_all_debug_files()
        self.assertEqual(ret, True)

        with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_dump_files', return_value=[]):
            ret = decode._find_all_debug_files()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_analyse(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        file_desc = {
            "file_path": "/test/Opdebug",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }

        decode.debug_files = ["/home/testfile1", "home/testfile2", "home/testfile3"]

        dump_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, {})
        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._find_all_debug_files',
                        return_value=True):
            with mock.patch('overflow.overflow_analyse.OverflowAnalyse._get_parsed_debug_file',
                            return_value=dump_file_desc):
                with mock.patch('cmp_utils.file_utils.FileUtils.load_json_file',
                                return_value=''):
                    with mock.patch('cmp_utils.file_utils.FileUtils.load_json_file',
                                    return_value=''):
                        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._json_summary',
                                        return_value='result_'):
                            with mock.patch('cmp_utils.file_utils.FileUtils.save_file',
                                            return_value=True):
                                ret = decode.analyse()
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_gen_overflow_info(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        over_type = 'AIV'
        json_txt = {
                "model_id": 365,
                "stream_id": 396,
                "task_id": 65,
                "task_type": 66,
                "context_id": 65535,
                "thread_id": 65535,
                "pc_start": "0x124040184000",
                "para_base": "0x1240401ff5f0",
                "core_id": 0,
                "block_id": 15,
                "status": 1
        }
        res = []
        ret = Analyse(args).get_overflow_info()(res, over_type, json_txt)
        self.assertEqual(res, [' [AIV][TaskId:65][StreamId:396][Status:1]'])
        self.assertEqual(ret, (65, 396, 65535, 65535))

    def test_insert_delimiter(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        ret = decode._insert_delimiter([], 1)
        self.assertEqual(ret, ['=================================================[%d]'
                               '==================================================' % 1])

    def test_npy_data_summary_case1(self):
        source_data = np.array([[1, 2, 3, 4],
                               [2, 3, 4, 5],
                               [3, 4, 5, 6]])
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        ret = decode.npy_data_summary(source_data)
        self.assertEqual(ret, '[Shape: (3, 4)] [Dtype: int64] [Max: 6] [Min: 1] [Mean: 3.5]')

    def test_npy_data_summary_case2(self):
        mock_data = np.array([[1, 2, 3, 4],
                               [2, 3, 4, 5],
                               [3, 4, 5, 6]])
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        source_data = "/home/test.npy"
        decode = overflow_analyse.OverflowAnalyse(args)
        with mock.patch('numpy.load', return_value=mock_data):
            ret = decode.npy_data_summary(source_data)
        self.assertEqual(ret, '[Shape: (3, 4)] [Dtype: int64] [Max: 6] [Min: 1] [Mean: 3.5]')

    def test_npy_data_summary_case3(self):
        source_data = "/home/test.bin"
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        with pytest.raises(CompareError) as err:
            decode.npy_data_summary(source_data)
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_npy_data_summary_case4(self):
        source_data = np.array([])
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        with pytest.raises(CompareError) as err:
            decode.npy_data_summary(source_data)
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def test_json_summary_case1(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        over_type = 'AI Core'
        json_txt = {
            'AI Core':
                {
                    'task_id': 1,
                    'stream_id': 25,
                    'status': 253
                },
            'L2 Atomic Add':
                {
                    'task_id': 2,
                    'stream_id': 35,
                    'status': 0
                }
        }
        overflow_index = 1
        debug_file = 'Op_debug_file'

        file_desc = {
            "file_path": "/test/Opdebug",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }
        debug_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._find_dump_files_by_task_id',
                        side_effect=CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)):
            ret = decode._json_summary(overflow_index, json_txt, debug_file_desc)
        print(ret)
        expect_result = '=================================================[1]' \
                        '==================================================\n' \
                        ' [AI Core][TaskId:1][StreamId:25][Status:253]\n [timestamp:161233160]'
        self.assertEqual(ret, expect_result)

    def test_json_summary_new_format(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        json_txt = {
            "magic": "0x5a5a5a5a",
            "version": 0,
            "acc_list": {
                "valid": 1,
                "acc_type": "AIV",
                "rsv": 0,
                "data_len": 88,
                "data": {
                    "model_id": 365,
                    "stream_id": 396,
                    "task_id": 65,
                    "task_type": 66,
                    "context_id": 65535,
                    "thread_id": 65535,
                    "pc_start": "0x124040184000",
                    "para_base": "0x1240401ff5f0",
                    "core_id": 0,
                    "block_id": 15,
                    "status": 1
                }
            }
        }
        overflow_index = 1

        file_desc = {
            "file_path": "/test1/Opdebug",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('12'),
            "stream_id": '12'
        }

        debug_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._find_dump_files_by_task_id',
                        side_effect=CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)):
            ret = decode._json_summary(overflow_index, json_txt, debug_file_desc)
        print(ret)
        expect_result = '=================================================[1]' \
                        '==================================================\n' \
                        ' [AIV][TaskId:65][StreamId:396][Status:1]\n [timestamp:161233160]'
        self.assertEqual(ret, expect_result)

    def test_json_summary_case2(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)

        over_type = 'AI Core'
        json_txt = {
            'AI Core':
                {
                    'task_id': 1,
                    'stream_id': 25,
                    'status': 253
                },
            'L2 Atomic Add':
                {
                    'task_id': 2,
                    'stream_id': 35,
                    'status': 0
                }
        }
        overflow_index = 1
        debug_file = 'Op_debug_file'

        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'cov2d',
            "op_type": 'convolution',
            "task_id": int('11'),
            "stream_id": '25'
        }
        debug_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160.output.npy",
            "timestamp": int("161233160")
        }
        anchor = {
            "anchor_type":'output',
            "anchor_idx":'0',
            "format":'NHWC'
        }
        parsed_dump_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, anchor)
        np_summary_result = '[Shape: (3, 4)] [Dtype: int64] [Max: 6] [Min: 1] [Mean: 3.5]'
        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._find_dump_files_by_task_id',
                        return_value=debug_file_desc):
            with mock.patch('overflow.overflow_analyse.OverflowAnalyse._get_parsed_dump_file',
                            return_value=[parsed_dump_file_desc]):
                with mock.patch('overflow.overflow_analyse.OverflowAnalyse.npy_data_summary',
                                return_value=np_summary_result):
                    ret = decode._json_summary(overflow_index, json_txt, debug_file_desc)
        print(ret)
        expect_result = '=================================================[1]' \
                        '==================================================\n' \
                        '[convolution] cov2d\n' \
                        ' [AI Core][TaskId:1][StreamId:25][Status:253]\n' \
                        ' [timestamp:161233160]\n' \
                        ' convolution.cov2d.1.25.161233160.output.npy\n' \
                        ' -[Format: NHWC] [Shape: (3, 4)] [Dtype: int64] [Max: 6] [Min: 1] [Mean: 3.5]'
        self.assertEqual(ret, expect_result)

    def test_parse_overflow_file(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        with mock.patch('dump_parse.dump_data_parser.DumpDataParser.parse_dump_data',
                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
            ret = decode._parse_overflow_file("/home", "/home")
        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_get_parsed_debug_file_case1(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        file_desc = {
            "file_path": "/test/Opdebug.Node_OpDebug.1.25.161233160",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }
        debug_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        file_desc = {
            "file_path": "/test/Opdebug.Node_OpDebug.1.25.161233160.output.0.json",
            "timestamp": int("161233160")
        }
        anchor = {
            "anchor_type": 'output',
            "anchor_idx": '0',
        }
        parsed_debug_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, anchor)

        with mock.patch('os.path.basename', return_value='Opdebug.Node_OpDebug.1.25.161233160'):
            with mock.patch('overflow.overflow_analyse.OverflowAnalyse._parse_overflow_file',
                            return_value=''):
                with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_parsed_debug_files',
                                side_effect=iter([{},
                                                  {'Opdebug.Node_OpDebug.1.25.161233160.output'
                                                   '.0.json': parsed_debug_file_desc}])):
                    ret = decode._get_parsed_debug_file(debug_file_desc)
        self.assertEqual(ret, parsed_debug_file_desc)

    def test_get_parsed_debug_file_case2(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        file_desc = {
            "file_path": "/test/Opdebug.Node_OpDebug.1.25.161233160",
            "timestamp": int("161233160")
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int('11'),
            "stream_id": '25'
        }
        debug_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)

        with pytest.raises(CompareError) as err:
            with mock.patch('os.path.basename', return_value='Opdebug.Node_OpDebug.1.25.161233160'):
                with mock.patch('overflow.overflow_analyse.OverflowAnalyse._parse_overflow_file',
                                return_value=''):
                    with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_parsed_debug_files',
                                    side_effect=iter([{}, {}])):
                        ret = decode._get_parsed_debug_file(debug_file_desc)
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)

    def test_get_parsed_dump_file_case1(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160",
            "timestamp": int("161233260")
        }
        dump_attr = {
            "op_name": 'cov2d',
            "op_type": 'convolution',
            "task_id": int('12'),
            "stream_id": '25'
        }
        dump_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        file_desc = {
            "file_path": "/test/convolution.cov2d.1.25.161233160.output.0.npy",
            "timestamp": int("161233160")
        }
        anchor = {
            "anchor_type": 'output',
            "anchor_idx": '0',
            "format": 'NCHW'
        }
        parsed_dump_file_desc = file_utils.ParsedDumpFileDesc(file_desc, dump_attr, anchor)
        with mock.patch('overflow.overflow_analyse.OverflowAnalyse._parse_overflow_file',
                        return_value=''):
            with mock.patch('os.path.basename', return_value='convolution.cov2d.1.25.161233160'):
                with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_parsed_dump_files',
                                return_value={'convolution.cov2d.1.25.161233160.output'
                                              '.0.npy': parsed_dump_file_desc}):
                    ret = decode._get_parsed_dump_file(dump_file_desc)
        self.assertEqual(ret, [parsed_dump_file_desc])

        with pytest.raises(CompareError) as err:
            with mock.patch('overflow.overflow_analyse.OverflowAnalyse._parse_overflow_file',
                            return_value=''):
                with mock.patch('os.path.basename', return_value='convolution.cov2d.1.25.161233160'):
                    with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_parsed_dump_files',
                                    return_value={}):
                        ret = decode._get_parsed_dump_file(dump_file_desc)
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_UNKNOWN_ERROR)


    def test_find_dump_files_by_task_id(self):
        args = argparse.Namespace()
        args.dump_path = "/home"
        args.output_path = "/home"
        args.top_num = 2
        decode = overflow_analyse.OverflowAnalyse(args)
        file_desc = {
            "file_path": "/test/convolution.cov2d.12.25.161233160",
            "timestamp": int("161233260")
        }
        dump_attr = {
            "op_name": 'cov2d',
            "op_type": 'convolution',
            "task_id": int('12'),
            "stream_id": '25'
        }
        dump_file_desc = file_utils.DumpFileDesc(file_desc, dump_attr)
        with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_dump_files',
                        return_value=[dump_file_desc]):
            ret = decode._find_dump_files_by_task_id('/test/', (12, 25, None, None))
        self.assertEqual(ret, dump_file_desc)

        with pytest.raises(CompareError) as err:
            with mock.patch('cmp_utils.file_utils.OverflowFileUtils.list_dump_files',
                            return_value=[dump_file_desc]):
                ret = decode._find_dump_files_by_task_id('/test/', (12, 24, None, None))
        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)


class Analyse(overflow_analyse.OverflowAnalyse):
    def get_overflow_info(self):
        return self._gen_overflow_info
