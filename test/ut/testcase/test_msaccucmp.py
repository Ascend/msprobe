import unittest
import pytest
from unittest import mock

import msaccucmp
from cmp_utils.constant.compare_error import CompareError


class TestUtilsMethods(unittest.TestCase):

    def test_main1(self):
        args = ['aaa.py', 'compare', '-m', '/home/left.bin', '-g',
                '/home/right.bin']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("msaccucmp._check_dump_path_exist"):
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
                        with mock.patch("cmp_utils.path.check_path_valid",
                                        return_value=CompareError.MSACCUCMP_NONE_ERROR):
                            msaccucmp.start_compare(args)
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

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
                    with mock.patch("cmp_utils.path.check_path_valid", return_value=0):
                        msaccucmp.main()
        self.assertEqual(error.value.args[0],
                         CompareError.MSACCUCMP_INVALID_TYPE_ERROR)

    def test_main_file_compare_case3(self):
        args = ['aaa.py', 'file_compare', '-m', '/home/left.npy', '-g',
                '/home/right.npy', '-out', '/home/output']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("cmp_utils.path.check_path_valid", return_value=0):
                    with mock.patch("cmp_utils.path.check_output_path_valid", return_value=0):
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
                with mock.patch("cmp_utils.path.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_check_max_line_effect2(self):
        args = ['aaa.py', 'compare', '-op', 'prob', '-m', '/home/left.bin', '-g',
                '/home/right.bin', '-f', '', '--max_line', '10000000']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch("cmp_utils.path.check_path_valid", return_value=0):
                    msaccucmp.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


if __name__ == '__main__':
    unittest.main()
