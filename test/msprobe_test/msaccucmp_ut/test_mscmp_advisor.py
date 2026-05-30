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
import pandas as pd

from advisor.advisor_const import AdvisorConst
from advisor.compare_advisor import CompareAdvisor
from advisor.advisor_result import AdvisorResult
import mscmp_advisor
from cmp_utils.constant.compare_error import CompareError

result = AdvisorResult()


class TestUtilsMethods(unittest.TestCase):

    def test_advisor_consistency(self):
        args = ['aaa.py', '-i', '/home/result.csv', '-o',
                '/home/wangchao']
        data = {"Index": ['0', '1', '2'],
                "CosineSimilarity": [0.99, 0.99, 0.99]}
        with mock.patch('sys.argv', args):
            with mock.patch('advisor.compare_advisor.CompareAdvisor._parse_input_file',
                            return_value=pd.DataFrame(data)):
                with mock.patch("advisor.advisor_result.AdvisorResult.gen_summary_file", return_value=True):
                    compare_advisor = CompareAdvisor("input_file.csv")
                    advisor_result = compare_advisor.advisor()
                    advisor_result.print_advisor_log()
        self.assertTrue(advisor_result.match_advisor)
        self.assertEqual(advisor_result.advisor_type, AdvisorConst.CONSISTENCY_DETECTION)
        self.assertEqual(advisor_result.operator_index, AdvisorConst.NO_ERROR_OP)
        self.assertEqual(advisor_result.advisor_message, AdvisorConst.CONSISTENCY_SUGGEST)

    def test_advisor_consistency_problem(self):
        args = ['aaa.py', '-i', '/home/result.csv', '-o',
                '/home/wangchao']
        data = {"Index": ['0', '1', '2'],
                "CosineSimilarity": [0.98, 0.99, 0.98]}
        with mock.patch('sys.argv', args):
            with mock.patch('advisor.compare_advisor.CompareAdvisor._parse_input_file',
                            return_value=pd.DataFrame(data)):
                with mock.patch("advisor.advisor_result.AdvisorResult.gen_summary_file", return_value=True):
                    compare_advisor = CompareAdvisor("input_file.csv")
                    advisor_result = compare_advisor.advisor()
                    advisor_result.print_advisor_log()
        self.assertTrue(advisor_result.match_advisor)
        self.assertEqual(advisor_result.advisor_type, AdvisorConst.CONSISTENCY_DETECTION)
        self.assertEqual(advisor_result.operator_index, "0")
        self.assertEqual(advisor_result.advisor_message, AdvisorConst.PROBLEM_SUGGEST)

    def test_advisor_consistency_ignore(self):
        args = ['aaa.py', '-i', '/home/result.csv', '-o',
                '/home/wangchao']
        data = {"Index": ['0', '1', '2'],
                "CosineSimilarity": [0.98, 0.99, 0.99]}
        with mock.patch('sys.argv', args):
            with mock.patch('advisor.compare_advisor.CompareAdvisor._parse_input_file',
                            return_value=pd.DataFrame(data)):
                with mock.patch("advisor.advisor_result.AdvisorResult.gen_summary_file", return_value=True):
                    compare_advisor = CompareAdvisor("input_file.csv")
                    advisor_result = compare_advisor.advisor()
                    advisor_result.print_advisor_log()
        self.assertTrue(advisor_result.match_advisor)
        self.assertEqual(advisor_result.advisor_type, AdvisorConst.CONSISTENCY_DETECTION)
        self.assertEqual(advisor_result.operator_index, "0")
        self.assertEqual(advisor_result.advisor_message, AdvisorConst.DEVIATION_SUGGEST)

    def test_advisor_overflow(self):
        args = ['aaa.py', '-i', '/home/result.csv', '-o',
                '/home/wangchao']
        data = {"Index": ['0', '1', '2'],
                "CosineSimilarity": [0.98, 0.99, 0.99],
                "OverFlow": ["NO", "YES", "NO"]}
        with mock.patch('sys.argv', args):
            with mock.patch('advisor.compare_advisor.CompareAdvisor._parse_input_file',
                            return_value=pd.DataFrame(data)):
                with mock.patch("advisor.advisor_result.AdvisorResult.gen_summary_file", return_value=True):
                    compare_advisor = CompareAdvisor("input_file.csv")
                    advisor_result = compare_advisor.advisor()
                    advisor_result.print_advisor_log()
        self.assertTrue(advisor_result.match_advisor)
        self.assertEqual(advisor_result.advisor_type, AdvisorConst.OVERFLOW_DETECTION)
        self.assertEqual(advisor_result.operator_index, "1")
        self.assertEqual(advisor_result.advisor_message, AdvisorConst.OVERFLOW_SUGGEST)

    def test_advisor_input(self):
        args = ['aaa.py', '-i', '/home/result.csv', '-o',
                '/home/wangchao']
        data = {"Index": ['0', '1', '2'],
                "CosineSimilarity": [0.98, 0.95, 0.99],
                "NPUDump": ["good", "error_node", "good"]}
        with mock.patch('sys.argv', args):
            with mock.patch('advisor.compare_advisor.CompareAdvisor._parse_input_file',
                            return_value=pd.DataFrame(data)):
                with mock.patch("advisor.advisor_result.AdvisorResult.gen_summary_file", return_value=True):
                    compare_advisor = CompareAdvisor("input_file.csv", ["error_node"])
                    advisor_result = compare_advisor.advisor()
                    advisor_result.print_advisor_log()
        self.assertTrue(advisor_result.match_advisor)
        self.assertEqual(advisor_result.advisor_type, AdvisorConst.INPUT_DETECTION)
        self.assertEqual(advisor_result.operator_index, "1")
        self.assertEqual(advisor_result.advisor_message, AdvisorConst.INPUT_SUGGEST)


if __name__ == '__main__':
    unittest.main()