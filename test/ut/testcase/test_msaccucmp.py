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

import os
import unittest
from unittest import mock
import tempfile

import pytest

import msaccucmp
from cmp_utils.constant.compare_error import CompareError
from pytorch_cmp.compare_pytorch import PytorchComparison


class TestUtilsMethods(unittest.TestCase):
    mock_stat_result = os.stat_result((0, 0, 0, 0, os.getuid(), 0, 0, 0, 0, 0))

    def test_main1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("msaccucmp._check_dump_path_exist"):
                        with mock.patch('os.stat', return_value=self.mock_stat_result):
                            msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=False):
                    with mock.patch("msaccucmp._check_dump_path_exist"):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)

    def test_main3(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main5(self):
        args = ['aaa.py']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_compare3(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare4(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare5(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-op', 'aaa', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_compare6(self):
        args = ['aaa.py', 'compare', '-m', '/home/left', '-g',
                '/home/right', '-cf', '/home/demo/xx.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isfile', return_value=False):
                with mock.patch('sys.argv', args):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_convert1(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-i', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert2(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-o', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert3(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-s', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_main_convert4(self):
        args = ['aaa.py', 'convert', '-d', '/home/left.bin', '-c', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_convert_when_dump_path_is_empty_then_error(self):
        args = ['aaa.py', 'convert', '-d', '/home/empty']
        with pytest.raises(SystemExit) as error:
            with mock.patch('os.path.isdir', return_value=True):
                with mock.patch('os.listdir', return_value=False):
                    with mock.patch('sys.argv', args):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_error_parameter1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', "-map"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_error_parameter2(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', "-f", "/home/a.json", "-map", "-op", "name"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_mapping_error_parameter3(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', "/home/a.json", "-map", "-i", "1"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_msaccucmp_alg_help(self):
        args = ['aaa.py', 'compare', "--help", '-alg', '1', '2', '3']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_msaccucmp_alg_help2(self):
        args = ['aaa.py', 'compare', "--help", '-alg', '1', '2', '9']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_msaccucmp_help(self):
        args = ['aaa.py', 'compare', "--help"]
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], 0)

    def test_start_compare(self):
        args = mock.Mock
        args.my_dump_path = "/home/my_dump_path"
        args.golden_dump_path = "/home/golden_dump_path"
        args.fusion_rule_file = "/home/fusion_rule_file"
        args.op_name = "data"
        args.post_process = 0
        with pytest.raises(CompareError) as error:
            with mock.patch("msaccucmp._check_hdf5_file_valid", return_value=False):
                with mock.patch("os.path.isfile", return_value=False):
                    with mock.patch("os.path.exists", return_value=False):
                        with mock.patch("cmp_utils.path_check.check_path_valid",
                                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
                            msaccucmp.start_compare(args)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
    
    @mock.patch('os.path.getsize', return_value=209715200)
    @mock.patch('msaccucmp._check_hdf5_file_valid', return_value=True)
    def test_start_compare_when_file_size_exceed_limit_then_pass(self,
                                                                 mock_check_hdf5_file_valid,
                                                                 mock_getsize):        
        args = mock.Mock
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp_file1, \
             tempfile.NamedTemporaryFile(suffix='.h5') as tmp_file2:

            args.my_dump_path = tmp_file1.name
            args.golden_dump_path = tmp_file2.name
            os.chmod(args.my_dump_path, 0o755)
            os.chmod(args.golden_dump_path, 0o755)
            warn_msg1 = "exceeds 100MB, it may task more time to run, please wait."
            warn_msg2 = "The size (209715200) of"
            with mock.patch.object(PytorchComparison, '__init__', return_value=None):
                with mock.patch.object(PytorchComparison, 'check_arguments_valid', return_value=None):
                    with mock.patch.object(PytorchComparison, 'compare', return_value=None):
                        with self.assertLogs() as log:
                            result = msaccucmp.start_compare(args)
                            self.assertIn(warn_msg1, log.output[0])
                            self.assertIn(warn_msg2, log.output[0])

        self.assertEqual(mock_getsize.call_count, 2)
    
    @mock.patch('os.path.getsize', return_value=5200)
    @mock.patch('msaccucmp._check_hdf5_file_valid', return_value=True)
    def test_start_compare_when_permission_not_valid(self,
                                                     mock_check_hdf5_file_valid,
                                                     mock_getsize):  
        args = mock.Mock
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp_file1, \
             tempfile.NamedTemporaryFile(suffix='.h5') as tmp_file2:

            args.my_dump_path = tmp_file1.name
            args.golden_dump_path = tmp_file2.name
            os.chmod(args.my_dump_path, 0o775)
            os.chmod(args.golden_dump_path, 0o775)
            with mock.patch.object(PytorchComparison, '__init__', return_value=None):
                with mock.patch.object(PytorchComparison, 'check_arguments_valid', return_value=None):
                    with mock.patch.object(PytorchComparison, 'compare', return_value=None):
                        with self.assertRaises(CompareError) as error:
                            result = msaccucmp.start_compare(args)
                        self.assertEqual(str(error.exception), "3")

    def test_main_overflow_case1(self):
        args = ['aaa.py', 'overflow', '-d', '/home/left.bin', '-out', '/home/output', '-n', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_main_overflow_case2(self):
        args = ['aaa.py', 'overflow', '-d', '/home/left.bin', '-out', '/home/output', '-n', '1']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('overflow.overflow_analyse.OverflowAnalyse.check_argument',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
                    with mock.patch('overflow.overflow_analyse.OverflowAnalyse.analyse',
                                    return_value=CompareError.MSACCUCMP_NONE_ERROR):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_NONE_ERROR)

    def test_main_file_compare_case1(self):
        args = ['aaa.py', 'file_compare', '-m', '/home/left.bin', '-g',
                '/home/right.npy', '-out', '/home/output']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("os.path.exists", return_value=False):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_main_file_compare_case2(self):
        args = ['aaa.py', 'file_compare', '-m', '/home/left.npy', '-g',
                '/home/right.bin', '-out', '/home/output']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("os.path.exists", return_value=False):
                    with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_main_file_compare_case3(self):
        args = ['aaa.py', 'file_compare', '-m', '/home/left.npy', '-g',
                '/home/right.npy', '-out', '/home/output']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
                    with mock.patch("cmp_utils.path_check.check_output_path_valid", return_value=0):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_DUMP_FILE_ERROR)

    def test_main_file_compare_case4(self):
        args = ['aaa.py', 'file_compare', '-m', '/home/left.npy', '-g',
                '/home/right.npy', '-out', '/home/output']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_check_range_effect1(self):
        args = ['aaa.py', 'compare', "-r", ',,', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect2(self):
        args = ['aaa.py', 'compare', "-r", ',,', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '/home/a.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect3(self):
        args = ['aaa.py', 'compare', "-s", ',,,', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_range_effect4(self):
        args = ['aaa.py', 'compare', "-s", ',,,', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '/home/a.json']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_max_line_effect1(self):
        args = ['aaa.py', 'compare', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '', '--max_line', '100']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_max_line_effect2(self):
        args = ['aaa.py', 'compare', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '', '--max_line', '10000000']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("cmp_utils.path_check.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


if __name__ == '__main__':
    unittest.main()
