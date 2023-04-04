import unittest
from unittest import mock
import pytest

from cmp_utils.constant.compare_error import CompareError
from src.compare.dump_parse import dump_parser


class TestUtilsMethods(unittest.TestCase):

    def test_save_log(self):
        args = ['aaa.py', 'save_log', '-d', '/home/result.csv', '-o',
                '/home/wangchao']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch("src.compare.dump_parse.dump_parser._do_save_log", return_value=0):
                    ret = dump_parser._do_cmd()
        self.assertEqual(ret, 0)

    def test_save_log_eror1(self):
        args = ['aaa.py']
        ret = 0
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch("src.compare.dump_parse.dump_parser._do_save_log", return_value=0):
                    try:
                        dump_parser._do_cmd()
                    except CompareError as error:
                        ret = error.code
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


if __name__ == '__main__':
    unittest.main()