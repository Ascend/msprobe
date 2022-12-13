import unittest
from unittest import mock
import pytest

from compare_error import CompareError
import dump_parser


class TestUtilsMethods(unittest.TestCase):

    def test_save_log(self):
        args = ['aaa.py', 'save_log', '-d', '/home/result.csv', '-o',
                '/home/wangchao']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("dump_parser._do_save_log", return_value=0):
                        dump_parser.main()
        self.assertEqual(error.value.args[0], 0)

    def test_save_log_eror1(self):
        args = ['aaa.py']
        with pytest.raises(SystemExit) as error:
            with mock.patch('sys.argv', args):
                with mock.patch('os.path.isfile', return_value=True):
                    with mock.patch("dump_parser._do_save_log", return_value=0):
                        dump_parser.main()
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


if __name__ == '__main__':
    unittest.main()