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
from collections import namedtuple
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

import numpy as np

from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import (
    DynamicInput,
    NpuDumpData,
)
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException

DynamicArgInfo = namedtuple('DynamicArgInfo', ['atc_arg', 'benchmark_arg'])


class TestDynamicInput(unittest.TestCase):

    def setUp(self):
        self.mock_om_parser = MagicMock()
        self.mock_om_parser.shape_range = None

        self.dym_batch_enum = Mock()
        self.dym_batch_enum.value = DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")

        self.dym_shape_enum = Mock()
        self.dym_shape_enum.value = DynamicArgInfo(atc_arg="--dym_shape", benchmark_arg="--shape")

        self.dym_dims_enum = Mock()
        self.dym_dims_enum.value = DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    def test_init_success(self, mock_get_arg_value, mock_get_dynamic_arg):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", "DYM_BATCH")
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertEqual(di.atc_dynamic_arg, "--dym_batch=10")
        self.assertEqual(di.cur_dynamic_arg, "DYM_BATCH")
        self.assertEqual(di.dynamic_arg_value, "8")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_found_exact(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims")),
        ]

        self.mock_om_parser.get_atc_cmdline.return_value = "atc --dym_batch 10 --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result[0], "--dym_batch=10")
        self.assertEqual(result[1], mock_enum.__iter__.return_value[0])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_found_with_equal(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="--batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="--dims"))
        ]
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --dym_batch=10 --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result[0], "--dym_batch=10")
        self.assertEqual(result[1], mock_enum.__iter__.return_value[0])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicArgumentEnum')
    def test_get_dynamic_arg_from_om_not_found(self, mock_enum):
        mock_enum.__iter__.return_value = [
            Mock(value=DynamicArgInfo(atc_arg="--dym_batch", benchmark_arg="batch")),
            Mock(value=DynamicArgInfo(atc_arg="--dym_dims", benchmark_arg="dims")),
        ]
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:1,3,224,224"

        result = DynamicInput.get_dynamic_arg_from_om(self.mock_om_parser)

        self.assertEqual(result, ("", None))

    def test_get_input_shape_from_om_with_space(self):
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:1,3,224,224"

        result = DynamicInput.get_input_shape_from_om(self.mock_om_parser)

        self.assertEqual(result, "input:1,3,224,224")

    def test_get_input_shape_from_om_with_equal(self):
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape=input:1,3,224,224"

        result = DynamicInput.get_input_shape_from_om(self.mock_om_parser)

        self.assertEqual(result, "input:1,3,224,224")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.logger')
    def test_get_arg_value_non_dynamic(self, mock_logger):
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (False, None)

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:1,3,224,224")

        self.assertEqual(result, "")
        mock_logger.info.assert_called_with("The input of model is not dynamic.")

    def test_get_arg_value_shape_range(self):
        """测试shape_range存在的情况"""
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (True, None)
        self.mock_om_parser.shape_range = "range_info"

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:1,3,224,224")

        self.assertEqual(result, "input:1,3,224,224")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_dim_values')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.INPUT_SHAPE', '--input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.logger')
    def test_get_arg_value_dynamic_batch_success(self, mock_logger, mock_get_dim, mock_parse):
        self.mock_om_parser.get_dynamic_scenario_info.return_value = (True, None)
        self.mock_om_parser.get_atc_cmdline.return_value = "atc --input_shape input:-1,3,224,224"

        mock_parse.side_effect = [
            {"input": ["-1", "3", "224", "224"]},
            {"input": ["8", "3", "224", "224"]}
        ]

        batch_set = set()
        batch_set.add("8")
        mock_get_dim.side_effect = lambda a, b, s: s.update(["8"])

        result = DynamicInput.get_arg_value(self.mock_om_parser, "input:8,3,224,224")

        self.assertEqual(result, "8")
        mock_logger.error.assert_not_called()

    def test_get_dynamic_dim_values_list(self):
        """测试get_dynamic_dim_values使用列表收集值"""
        dym_shape = ["-1", "3", "-1"]
        cur_shape = ["8", "3", "224"]
        shape_values = []

        DynamicInput.get_dynamic_dim_values(dym_shape, cur_shape, shape_values)
        self.assertEqual(shape_values, [8, 224])

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_input_dynamic_arg_valid')
    def test_add_dynamic_arg_for_benchmark_dynamic(self, mock_check_valid, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        benchmark_cmd = ["benchmark"]
        di.add_dynamic_arg_for_benchmark(benchmark_cmd)

        self.assertEqual(benchmark_cmd, ["benchmark", "--batch", "8"])
        mock_check_valid.assert_called_once()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_is_dynamic_shape_scenario_true(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertTrue(di.is_dynamic_shape_scenario())

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_is_dynamic_shape_scenario_false(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertFalse(di.is_dynamic_shape_scenario())

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_judge_dynamic_shape_scenario_match(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertTrue(di.judge_dynamic_shape_scenario("--dym_batch"))

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    def test_judge_dynamic_shape_scenario_not_match(self, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_dims=10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        self.assertFalse(di.judge_dynamic_shape_scenario("--dym_batch"))

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_dims_valid')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_batch_valid')
    def test_check_input_dynamic_arg_valid_dym_shape(
            self,
            mock_check_batch,
            mock_check_dims,
            mock_get_dynamic_arg,
            mock_get_arg_value
    ):
        mock_get_dynamic_arg.return_value = ("--dym_shape=", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_batch.assert_not_called()
        mock_check_dims.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_dims_valid')
    def test_check_input_dynamic_arg_valid_dym_dims(self, mock_check_dims, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_dims=5~10", self.dym_batch_enum)
        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_dims.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.check_dynamic_batch_valid')
    def test_check_input_dynamic_arg_valid_dym_batch(self, mock_check_batch, mock_get_dynamic_arg, mock_get_arg_value):
        mock_get_dynamic_arg.return_value = ("--dym_batch=5~10", self.dym_batch_enum)
        mock_get_arg_value.return_value = ""

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        di.check_input_dynamic_arg_valid()

        mock_check_batch.assert_not_called()

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_dynamic_arg_from_om')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_value_by_comma')
    def test_check_dynamic_batch_valid_success(
        self,
        mock_parse_comma,
        mock_parse_arg,
        mock_get_dynamic_arg,
        mock_get_arg_value
    ):
        mock_get_dynamic_arg.return_value = ("--dym_batch=10", self.dym_batch_enum)
        mock_get_arg_value.return_value = "8"

        di = DynamicInput(self.mock_om_parser, "input:8,3,224,224")

        mock_parse_arg.return_value = ["5", "6", "7", "8", "9", "10"]
        mock_parse_comma.return_value = "8"

        di.check_dynamic_batch_valid("5~10")

        mock_parse_arg.assert_called_once_with("5~10")
        mock_parse_comma.assert_called_once_with("8")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_arg_value')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_input_shape_from_om')
    def test_check_dynamic_dims_valid_success(
        self,
        mock_get_input_shape,
        mock_parse_input_shape,
        mock_parse_arg_value
    ):
        mock_get_input_shape.return_value = "input:-1,3,224,224"

        # 第一次调用 parse_input_shape: atc 输入 shape；第二次：当前输入 shape
        mock_parse_input_shape.side_effect = [
            {"input": ["-1", "3", "224", "224"]},
            {"input": ["8", "3", "224", "224"]},
        ]
        mock_parse_arg_value.return_value = [[8]]

        di = object.__new__(DynamicInput)
        di.om_parser = self.mock_om_parser
        di.dynamic_arg_value = "input:8,3,224,224"
        # 不应抛出异常
        di.check_dynamic_dims_valid("1~8")

        assert mock_parse_input_shape.call_count == 2
        mock_parse_arg_value.assert_called_once_with("1~8")

    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.parse_input_shape')
    @patch('msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput.get_input_shape_from_om')
    def test_check_dynamic_dims_valid_parse_error(self, mock_get_input_shape, mock_parse_input_shape):
        mock_get_input_shape.return_value = "input:-1,3,224,224"
        mock_parse_input_shape.side_effect = AccuracyCompareException(0)

        di = object.__new__(DynamicInput)
        di.om_parser = self.mock_om_parser
        di.dynamic_arg_value = "input:8,3,224,224"
        with self.assertRaises(AccuracyCompareException) as cm:
            di.check_dynamic_dims_valid("1~8")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR
        )


class TestNpuDumpData(unittest.TestCase):

    def _build_npu_dump_instance(self):
        """构造一个未调用 __init__ 的 NpuDumpData 实例，手动注入属性，避免真实依赖。"""
        inst = object.__new__(NpuDumpData)
        inst.target_path = "model.om"
        inst.output_path = "/tmp/out"
        inst.input_data = ""
        inst.input_shape = ""
        inst.output_size = ""
        inst.device = "0"
        inst.is_golden = False
        inst.dump = True
        inst.benchmark_input_path = ""
        inst.om_parser = MagicMock()
        inst.dynamic_input = MagicMock()
        inst.python_version = "python3"
        return inst

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.create_directory")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.OmParser")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.atc_utils.convert_model_to_json")
    def test_npu_dump_data_init_non_golden(
        self, mock_convert, mock_om_parser, mock_dynamic_input, mock_create_dir
    ):
        mock_convert.return_value = "out.json"
        mock_om_parser.return_value = "OM_PARSER"
        mock_dynamic_input.return_value = "DYN_INPUT"

        class Args:
            pass

        args = Args()
        args.golden_path = "golden.om"
        args.target_path = "model.om"
        args.output_path = "/out"
        args.input_data = ""
        args.input_shape = "input:1,3,224,224"
        args.output_size = ""
        args.rank = "0"
        args.cann_path = "/cann"

        inst = NpuDumpData(args, is_golden=False)

        self.assertEqual(inst.target_path, "model.om")
        self.assertEqual(inst.output_path, "/out")
        self.assertEqual(inst.input_data, "")
        self.assertEqual(inst.input_shape, "input:1,3,224,224")
        self.assertEqual(inst.output_size, "")
        self.assertEqual(inst.device, "0")
        self.assertFalse(inst.is_golden)
        self.assertEqual(inst.benchmark_input_path, "")
        self.assertEqual(inst.om_parser, "OM_PARSER")
        self.assertEqual(inst.dynamic_input, "DYN_INPUT")
        self.assertEqual(inst.data_dir, "/out/input")

        mock_convert.assert_called_once_with("/cann", "model.om", "/out")
        mock_om_parser.assert_called_once_with("out.json")
        mock_dynamic_input.assert_called_once_with("OM_PARSER", "input:1,3,224,224")
        mock_create_dir.assert_called_once_with("/out/input")

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.create_directory")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.DynamicInput")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.OmParser")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.atc_utils.convert_model_to_json")
    def test_npu_dump_data_init_golden(
        self, mock_convert, mock_om_parser, mock_dynamic_input, mock_create_dir
    ):
        mock_convert.return_value = "out.json"
        mock_om_parser.return_value = "OM_PARSER"
        mock_dynamic_input.return_value = "DYN_INPUT"

        class Args:
            pass

        args = Args()
        args.golden_path = "golden.om"
        args.target_path = "model.om"
        args.output_path = "/out"
        args.input_data = ""
        args.input_shape = ""
        args.output_size = ""
        args.rank = "1"
        args.cann_path = "/cann"

        inst = NpuDumpData(args, is_golden=True)

        self.assertEqual(inst.target_path, "golden.om")
        self.assertTrue(inst.is_golden)
        mock_convert.assert_called_once_with("/cann", "golden.om", "/out")

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.ms_open")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.remove")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.stat")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.access")
    def test_write_content_to_acl_json_success(
        self, mock_access, mock_stat, mock_remove, mock_ms_open
    ):
        mock_access.return_value = True
        mock_stat.return_value.st_uid = 1000
        with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.getuid", return_value=1000):
            mock_file = MagicMock()
            mock_ms_open.return_value.__enter__.return_value = mock_file
            with patch("json.dump") as mock_dump:
                NpuDumpData._write_content_to_acl_json(
                    "acl.json", "model", "/dump_dir", ["sub1", "sub2"]
                )

            mock_remove.assert_called_once_with("acl.json")
            mock_dump.assert_called_once()

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.access", return_value=False)
    def test_write_content_to_acl_json_no_permission(self, mock_access):
        with self.assertRaises(AccuracyCompareException) as cm:
            NpuDumpData._write_content_to_acl_json("acl.json", "model", "/dump_dir", [])
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR
        )

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.access", return_value=True)
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.stat")
    def test_write_content_to_acl_json_wrong_owner(self, mock_stat, mock_access):
        mock_stat.return_value.st_uid = 2000
        with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.getuid", return_value=1000):
            with self.assertRaises(AccuracyCompareException) as cm:
                NpuDumpData._write_content_to_acl_json("acl.json", "model", "/dump_dir", [])
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR
        )

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.ms_open")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.remove")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.stat")
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.access")
    def test_write_content_to_acl_json_value_error_in_dump(
        self, mock_access, mock_stat, mock_remove, mock_ms_open
    ):
        mock_access.return_value = True
        mock_stat.return_value.st_uid = 1000
        with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.getuid", return_value=1000):
            mock_file = MagicMock()
            mock_ms_open.return_value.__enter__.return_value = mock_file
            with patch("json.dump", side_effect=ValueError("bad json")):
                with self.assertRaises(AccuracyCompareException) as cm:
                    NpuDumpData._write_content_to_acl_json(
                        "acl.json", "model", "/dump_dir", []
                    )
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_WRITE_JSON_FILE_ERROR
        )

    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.access", return_value=True)
    @patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.stat")
    @patch(
        "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.ms_open",
        side_effect=IOError("open fail"),
    )
    def test_write_content_to_acl_json_io_error(
        self, mock_ms_open, mock_stat, mock_access
    ):
        mock_stat.return_value.st_uid = 1000
        with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.getuid", return_value=1000), \
                patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.remove"):
            with self.assertRaises(AccuracyCompareException) as cm:
                NpuDumpData._write_content_to_acl_json("acl.json", "model", "/dump_dir", [])
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR
        )

    def test_generate_inputs_data_with_invalid_input_file(self):
        inst = self._build_npu_dump_instance()
        inst.input_data = "not_exist.bin"
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(AccuracyCompareException) as cm:
                inst.generate_inputs_data()
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR
        )

    def test_generate_inputs_data_with_valid_files(self):
        inst = self._build_npu_dump_instance()
        inst.output_path = "/out"
        inst.input_data = "/in/a.bin,/in/b.bin"
        with patch("os.path.isfile", return_value=True), \
                patch("shutil.copy") as mock_copy, \
                patch("os.chmod") as mock_chmod:
            inst.generate_inputs_data()

        # 两个 input 文件各 copy 一次
        self.assertEqual(mock_copy.call_count, 2)
        self.assertEqual(mock_chmod.call_count, 2)

    def test_check_input_path_param_from_generated_input_dir(self):
        inst = self._build_npu_dump_instance()
        inst.output_path = "/out_dir"
        inst.input_data = ""
        fake_files = ["input_0.bin", "input_1.bin"]
        with patch("os.path.realpath", return_value="/out_dir/input"), \
                patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.check_file_or_directory_path"), \
                patch("os.listdir", return_value=fake_files):
            inst._check_input_path_param()

        self.assertEqual(
            inst.benchmark_input_path,
            "/out_dir/input/input_0.bin,/out_dir/input/input_1.bin",
        )

    def test_check_input_path_param_from_input_data_arg(self):
        inst = self._build_npu_dump_instance()
        inst.input_data = "/in/0.bin,/in/1.bin"
        with patch(
            "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.check_input_bin_file_path",
            return_value=["/in/0.bin", "/in/1.bin"],
        ):
            inst._check_input_path_param()
        self.assertEqual(inst.benchmark_input_path, "/in/0.bin,/in/1.bin")

    def test_get_inputs_info_from_aclruntime(self):
        inst = self._build_npu_dump_instance()
        inst.target_path = "model.om"
        inst.device = "0"

        fake_input = MagicMock()
        fake_input.shape = (1, 3, 224, 224)
        fake_dtype = MagicMock()
        fake_dtype.name = "float32"
        fake_input.datatype = fake_dtype

        class FakeSession:
            def __init__(self, *args, **kwargs):
                self._inputs = [fake_input]

            def get_inputs(self):
                return self._inputs

            def free_resource(self):
                self._freed = True

        fake_aclruntime = MagicMock()
        fake_aclruntime.session_options.return_value = "opts"
        fake_aclruntime.InferenceSession = FakeSession

        with patch.dict(
            "sys.modules", {"aclruntime": fake_aclruntime}, clear=False
        ), patch(
            "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.Rule.input_file"
        ) as mock_rule, patch(
            "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.load_file_to_read_common_check",
            return_value="model.om",
        ):
            mock_rule.return_value.check.return_value = None
            shapes, dtypes, names = inst._get_inputs_info_from_aclruntime()

        self.assertEqual(shapes, [(1, 3, 224, 224)])
        self.assertEqual(dtypes, ["float32"])

    def test_generate_inputs_data_without_aipp_dynamic_without_shape(self):
        inst = self._build_npu_dump_instance()
        inst.input_shape = ""
        inst.dynamic_input.is_dynamic_shape_scenario.return_value = True

        with patch("os.listdir", return_value=[]), \
                patch.object(
                    inst, "_get_inputs_info_from_aclruntime", return_value=([(1, 3, 224, 224)], ["float32"], ['aaa'])
                ):
            with self.assertRaises(AccuracyCompareException) as cm:
                inst._generate_inputs_data_without_aipp("/input_dir")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR
        )

    def test_generate_inputs_data_without_aipp_normal(self):
        inst = self._build_npu_dump_instance()
        inst.input_shape = "input:1,3,224,224"
        inst.dynamic_input.is_dynamic_shape_scenario.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir, \
                patch.object(
                    inst, "_get_inputs_info_from_aclruntime", return_value=([(1, 3, 224, 224)], ["float32"], ['aaa'])
                ), \
                patch(
                    "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.parse_input_shape_to_list",
                    return_value=([(1, 3, 224, 224)], None)
                ), \
                patch("numpy.random.random", return_value=np.ones((1, 3, 224, 224))), \
                patch("os.chmod") as mock_chmod:
            inst._generate_inputs_data_without_aipp(tmpdir)

        mock_chmod.assert_called_once()

    def test_generate_inputs_data_for_aipp_invalid_src_size(self):
        inst = self._build_npu_dump_instance()
        inst.om_parser.get_aipp_config_content.return_value = [
            'src_image_size_h:224,input_format:RGB888_U8'
        ]
        with self.assertRaises(AccuracyCompareException) as cm:
            inst._generate_inputs_data_for_aipp("/input_dir")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT
        )

    def test_generate_inputs_data_for_aipp_invalid_hw_number(self):
        inst = self._build_npu_dump_instance()
        inst.om_parser.get_aipp_config_content.return_value = [
            'src_image_size_h:224,src_image_size_w:224,src_image_size_h:112'
        ]
        with self.assertRaises(AccuracyCompareException) as cm:
            inst._generate_inputs_data_for_aipp("/input_dir")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT
        )

    def test_generate_inputs_data_for_aipp_mismatch_inputs_number(self):
        inst = self._build_npu_dump_instance()
        inst.input_shape = ""
        inst.om_parser.get_aipp_config_content.return_value = [
            'src_image_size_h:224,src_image_size_w:224,input_format:RGB888_U8'
        ]
        inst.om_parser.get_shape_list.return_value = [(1, 3, 224, 224), (1, 3, 112, 112)]

        with self.assertRaises(AccuracyCompareException) as cm:
            inst._generate_inputs_data_for_aipp("/input_dir")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT
        )

    def test_generate_inputs_data_for_aipp_invalid_input_format(self):
        inst = self._build_npu_dump_instance()
        inst.input_shape = "input:1,3,224,224"
        inst.om_parser.get_aipp_config_content.return_value = [
            'src_image_size_h:224,src_image_size_w:224,input_format:UNKNOWN'
        ]
        with self.assertRaises(AccuracyCompareException) as cm:
            inst._generate_inputs_data_for_aipp("/input_dir")
        self.assertEqual(
            cm.exception.error_info, utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT
        )

    def test_generate_inputs_data_for_aipp_normal(self):
        inst = self._build_npu_dump_instance()
        inst.input_shape = "input:1,3,224,224"
        inst.om_parser.get_aipp_config_content.return_value = [
            'src_image_size_h:224,src_image_size_w:224,input_format:RGB888_U8'
        ]

        with tempfile.TemporaryDirectory() as tmpdir, \
                patch(
                    "msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.parse_input_shape_to_list",
                    return_value=([[1, 3, 224, 224]], None)
                ), \
                patch("numpy.random.randint", return_value=np.ones((1, 3, 224, 224), dtype=np.uint8)), \
                patch("os.chmod") as mock_chmod:
            inst.output_path = tmpdir
            os.makedirs(os.path.join(tmpdir, "input"), exist_ok=True)
            inst._generate_inputs_data_for_aipp(tmpdir)

        mock_chmod.assert_called_once()
