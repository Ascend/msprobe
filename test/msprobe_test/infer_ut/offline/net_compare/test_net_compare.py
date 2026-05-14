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
import subprocess, io
from unittest.mock import patch, MagicMock, mock_open
from msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare import NetCompare


class TestNetCompare_init(unittest.TestCase):
    @patch("msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.check_file_size_valid", return_value=None)
    @patch("msprobe.infer.offline.compare.msquickcmp.common.utils.check_file_size_valid")
    def test_init_basic(self, mock_check_size, mock_check_file_size_valid):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False

        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args, "/golden/json")
        # check python_version取值正常
        self.assertIn("python", obj.python_version.lower())


class TestNetCompare_catch_compare_result(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_num(self, mock_logger):
        line = b"[INFO] 0.123 0.456"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(res)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_nan(self, mock_logger):
        line = b"[INFO] NaN something"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(res)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_header(self, mock_logger):
        line = b"[INFO] Cosine Something"
        res, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(res)
        self.assertTrue(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_exception(self, mock_logger):
        with self.assertRaises(Exception):  # AccuracyCompareException
            NetCompare._catch_compare_result(b"\xff", True)  # invalid decode


class TestNetCompare_accuracy_network_compare(unittest.TestCase):
    @patch.object(NetCompare, "execute_msaccucmp_command", return_value=(0, [], []))
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_accuracy_network_compare_success(self, mock_logger, mock_exec):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        obj.accuracy_network_compare()
        mock_exec.assert_called_once()
        mock_logger.info.assert_any_call("Finish compare the files in directory /npu/path with those in directory /cpu/path.")

    @patch.object(NetCompare, "execute_msaccucmp_command", return_value=(1, [], []))
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_accuracy_network_compare_fail(self,  mock_logger, mock_exec):
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            output_path = "/out/path"
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        with self.assertRaises(Exception):  # AccuracyCompareException
            obj.accuracy_network_compare()


class TestCatchCompareResult(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_valid_number(self, mock_logger):
        line = b'[INFO]123.456 other stuff'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(result)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_valid_nan(self, mock_logger):
        line = b'[INFO]NaN some other'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertTrue(result)
        self.assertFalse(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_header(self, mock_logger):
        line = b'[INFO]Cosine Error Distance'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(result)
        self.assertTrue(header)

    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.logger')
    def test_catch_compare_result_no_info(self, mock_logger):
        line = b'Some random log line'
        result, header = NetCompare._catch_compare_result(line, True)
        self.assertFalse(result)
        self.assertFalse(header)

    def test_catch_compare_result_exception(self):
        with self.assertRaises(Exception):
            NetCompare._catch_compare_result(123, True)


class TestExecuteMsaccucmpCommand(unittest.TestCase):

    @patch('subprocess.Popen')
    @patch.object(NetCompare, "_catch_compare_result")
    def test_execute_msaccucmp_command(self, mock_catch, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.stdout.readline.side_effect = ["[INFO]1.23 4.56\n", "[INFO]Cosine\n", ""]
        mock_process.returncode = 0
        mock_process.stdout.close = MagicMock()
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process
        mock_catch.side_effect = [
            (["1.23", "4.56"], []),
            ([], ["Cosine"])
        ]
        class Args:
            cann_path = "/fake/cann"
            quant_fusion_rule_file = None
            out_path = "/out/path"
            advisor = False
            max_cmp_size = None
            dump = False
        args = Args()
        obj = NetCompare("/npu/path", "/cpu/path", "/output/json", args)
        status, result, header = obj.execute_msaccucmp_command(["fake", "cmd"], catch=True)
        self.assertEqual(status, 0)
        self.assertEqual(result, ["1.23", "4.56"])
        self.assertEqual(header, ["Cosine"])
        self.assertEqual(mock_catch.call_count, 2)


class TestProcessResultOneLine(unittest.TestCase):
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.csv.writer')
    @patch('msprobe.core.common.log.logger')
    @patch('msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare.ms_open', new_callable=mock_open, read_data="Header,NPU_Dump,GroundTruth\nline1\nline2\n")
    def test_process_result_one_line(self, mock_file, mock_logger, mock_csv_writer):
        mock_fp_write = MagicMock()
        mock_fp_read = io.StringIO("NPU Dump,NPUDump,GroundTruth\nNode_Output,abc,*\nother,line,*\n")
        instance = NetCompare.__new__(NetCompare)
        # 模拟 _check_msaccucmp_compare_support_advisor 返回 False
        instance._check_msaccucmp_compare_support_advisor = MagicMock(return_value=False)
        # 调用
        instance._process_result_one_line(mock_fp_write, mock_fp_read, 'npu_file.npy', 'golden_file.npy', ['1','2'])
        self.assertTrue(mock_csv_writer.return_value.writerow.called)
