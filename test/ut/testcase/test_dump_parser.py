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

import unittest
from unittest import mock

from cmp_utils.constant.compare_error import CompareError
import dump_parser


class TestUtilsMethods(unittest.TestCase):

    def test_save_log(self):
        args = ['aaa.py', 'save_log', '-d', '/home/result.csv', '-o',
                '/home/wangchao']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch("dump_parser._do_save_log", return_value=0):
                    ret = dump_parser._do_cmd()
        self.assertEqual(ret, 0)

    def test_save_log_eror1(self):
        args = ['aaa.py']
        ret = 0
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch("dump_parser._do_save_log", return_value=0):
                    try:
                        dump_parser._do_cmd()
                    except CompareError as error:
                        ret = error.code
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


if __name__ == '__main__':
    unittest.main()