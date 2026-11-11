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
import os
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import NpuDumpData

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
                patch("numpy.random.random", return_value=np.ones((1, 3, 224, 224))):
            inst._generate_inputs_data_without_aipp(tmpdir)

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
                patch("numpy.random.randint", return_value=np.ones((1, 3, 224, 224), dtype=np.uint8)):
            inst.output_path = tmpdir
            os.makedirs(os.path.join(tmpdir, "input"), exist_ok=True)
            inst._generate_inputs_data_for_aipp(tmpdir)



class TestNpuDumpDataExtra(unittest.TestCase):
    def _build(self):
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

    def test_collect_path_permissions(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            sub_dir = os.path.join(base, "sub")
            os.makedirs(sub_dir)
            file_path = os.path.join(sub_dir, "file.bin")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("x")
            modes = {}
            inst._collect_path_permissions(base, modes)
            self.assertIn(base, modes)
            self.assertIn(sub_dir, modes)
            self.assertIn(file_path, modes)

            unchanged = dict(modes)
            inst._collect_path_permissions(base, unchanged)
            self.assertEqual(modes, unchanged)
            inst._collect_path_permissions("", {})
            inst._collect_path_permissions(os.path.join(base, "missing"), {})

    def test_change_path_permissions_and_restore(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            sub_dir = os.path.join(base, "sub")
            os.makedirs(sub_dir)
            file_path = os.path.join(sub_dir, "file.bin")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("x")

            original_modes = inst._change_path_permissions([f"{base},{sub_dir}"])
            self.assertIn(base, original_modes)
            self.assertIn(sub_dir, original_modes)
            self.assertIn(file_path, original_modes)
            self.assertEqual(stat.S_IMODE(os.stat(base).st_mode), 0o750)
            self.assertEqual(stat.S_IMODE(os.stat(file_path).st_mode), 0o640)

            inst._change_path_permissions([], is_restore=True, original_modes=original_modes)
            self.assertNotEqual(stat.S_IMODE(os.stat(file_path).st_mode), 0o640)

    def test_change_path_permissions_restore_without_modes(self):
        inst = self._build()
        self.assertIsNone(inst._change_path_permissions([], is_restore=True, original_modes=None))

    def test_change_path_permissions_restores_on_error(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            file_a = os.path.join(base, "a.bin")
            file_b = os.path.join(base, "b.bin")
            for path in (file_a, file_b):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("x")
            calls = []

            def fake_chmod(path, mode):
                calls.append((path, mode))
                if len(calls) == 2:
                    raise OSError("boom")

            with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.chmod", side_effect=fake_chmod):
                with self.assertRaises(OSError):
                    inst._change_path_permissions([f"{file_a},{file_b}"])
            self.assertGreaterEqual(len(calls), 3)

    def test_get_file_size(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            bin_path = os.path.join(base, "a.bin")
            with open(bin_path, "wb") as f:
                f.write(b"abcd")
            npy_path = os.path.join(base, "b.npy")
            np.save(npy_path, np.array([1, 2, 3], dtype=np.float32))
            npy_upper_path = os.path.join(base, "c.NPY")
            with open(npy_upper_path, "wb") as f:
                np.save(f, np.array([1, 2], dtype=np.float32))
            inst.benchmark_input_path = f"{bin_path},{npy_path},{npy_upper_path}"
            with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.Rule.input_file"):
                self.assertEqual(inst._get_file_size(), [4, 3, 2])

    def test_get_file_size_invalid_suffix(self):
        inst = self._build()
        inst.benchmark_input_path = "a.txt"
        with self.assertRaises(utils.AccuracyCompareException):
            inst._get_file_size()

    def test_get_file_size_npy_failure(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            npy_path = os.path.join(base, "b.npy")
            with open(npy_path, "wb") as f:
                f.write(b"x")
            inst.benchmark_input_path = npy_path
            with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.Rule.input_file"), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.np.load",
                          side_effect=ValueError("bad")):
                with self.assertRaises(utils.AccuracyCompareException):
                    inst._get_file_size()

    def test_shape_size_vs_file_size_file_count_mismatch(self):
        inst = self._build()
        with self.assertRaises(utils.AccuracyCompareException):
            inst._shape_size_vs_file_size([1], [1, 2])

    def test_shape_size_vs_file_size_shape_range(self):
        inst = self._build()
        inst.om_parser.shape_range = True
        with self.assertRaises(utils.AccuracyCompareException):
            inst._shape_size_vs_file_size([1, 2], [3])
        inst._shape_size_vs_file_size([1, 3], [3])

    def test_shape_size_vs_file_size_dynamic(self):
        inst = self._build()
        inst.om_parser.shape_range = False
        inst.dynamic_input.is_dynamic_shape_scenario.return_value = True
        inst._shape_size_vs_file_size([2], [1])
        inst._shape_size_vs_file_size([1], [2])

    def test_shape_size_vs_file_size_normal(self):
        inst = self._build()
        inst.om_parser.shape_range = False
        inst.dynamic_input.is_dynamic_shape_scenario.return_value = False
        inst._shape_size_vs_file_size([0, 2], [0, 2])
        with self.assertRaises(utils.AccuracyCompareException):
            inst._shape_size_vs_file_size([1], [2])

    def test_compare_shape_vs_file(self):
        inst = self._build()
        inst.om_parser.contain_negative_1 = True
        with patch.object(inst, "_get_file_size") as get_size:
            inst._compare_shape_vs_file()
        get_size.assert_not_called()

        inst.om_parser.contain_negative_1 = False
        inst.om_parser.get_aipp_config_content = "aipp"
        inst.om_parser.get_shape_size.return_value = [1]
        with patch.object(inst, "_get_file_size", return_value=[1]), \
                patch.object(inst, "_shape_size_vs_file_size") as compare:
            inst._compare_shape_vs_file()
        compare.assert_not_called()

        inst.om_parser.get_aipp_config_content = None
        with patch.object(inst, "_get_file_size", return_value=[1]), \
                patch.object(inst, "_shape_size_vs_file_size") as compare:
            inst._compare_shape_vs_file()
        compare.assert_called_once_with([1], [1])

    def test_make_benchmark_cmd_for_shape_range(self):
        inst = self._build()
        inst.om_parser.get_net_output_count.return_value = 2
        inst.output_size = ""
        cmd = []
        inst._make_benchmark_cmd_for_shape_range(cmd)
        self.assertEqual(cmd, ["--outputSize", "90000000,90000000"])

        inst.output_size = "10,20"
        cmd = []
        inst._make_benchmark_cmd_for_shape_range(cmd)
        self.assertEqual(cmd, ["--outputSize", "10,20"])

        for output_size in ("10", "10,abc", "10,-1"):
            with self.subTest(output_size=output_size):
                inst.output_size = output_size
                with self.assertRaises(utils.AccuracyCompareException):
                    inst._make_benchmark_cmd_for_shape_range([])

    def test_convert_net_output_to_numpy(self):
        inst = self._build()
        with tempfile.TemporaryDirectory() as base:
            np.array([1.0, 2.0], dtype=np.float32).tofile(os.path.join(base, "data_0"))
            np.array([3.0, 4.0], dtype=np.float32).tofile(os.path.join(base, "data_1"))
            inst.om_parser.get_net_output_data_info.return_value = {
                0: [np.float32, (2,)],
                1: [np.float32, (2,)],
            }
            with patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.Rule.input_file"):
                inst._convert_net_output_to_numpy(base, "dump")
            self.assertTrue(os.path.exists(os.path.join(base, "output_0")))
            self.assertTrue(os.path.exists(os.path.join(base, "output_0.npy")))
            self.assertTrue(os.path.exists(os.path.join(base, "output_1")))
            self.assertTrue(os.path.exists(os.path.join(base, "output_1.npy")))

    def test_generate_dump_data(self):
        inst = self._build()
        with patch.object(inst, "_check_input_path_param") as check, \
                patch.object(inst, "benchmark_run", return_value=("dump", "net")) as run:
            self.assertEqual(inst.generate_dump_data(), ("dump", "net"))
        check.assert_called_once()
        run.assert_called_once()

    def test_get_expect_output_name(self):
        inst = self._build()
        inst.om_parser.get_expect_net_output_name.return_value = "output"
        self.assertEqual(inst.get_expect_output_name(), "output")

    def _run_benchmark(self, dump=True, dump_exist=True, net_exist=True):
        inst = self._build()
        inst.dump = dump
        with tempfile.TemporaryDirectory() as base:
            inst.output_path = base
            with patch.dict("sys.modules", {"ais_bench": MagicMock()}), \
                    patch.object(inst, "_compare_shape_vs_file"), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.create_directory"), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.path.exists",
                          return_value=False), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.os.mknod"), \
                    patch.object(inst, "_write_content_to_acl_json"), \
                    patch.object(inst, "_change_path_permissions", return_value={}), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils."
                          "get_model_name_and_extension", return_value=("model", "om")), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils."
                          "get_dump_data_path",
                          side_effect=([("dump_path", dump_exist), ("net_path", net_exist)]
                                       if dump else [("net_path", net_exist)])), \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.utils.execute_command") as execute, \
                    patch("msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data.filter_cmd",
                          side_effect=lambda cmd: cmd), \
                    patch.object(inst, "_make_benchmark_cmd_for_shape_range"), \
                    patch.object(inst, "_convert_net_output_to_numpy") as convert:
                result = inst.benchmark_run()
        return inst, result, execute, convert

    def test_benchmark_run_with_dump(self):
        _, result, execute, convert = self._run_benchmark(dump=True)
        self.assertEqual(result, ("dump_path", "net_path"))
        execute.assert_called_once()
        convert.assert_called_once()

    def test_benchmark_run_without_dump(self):
        _, result, _, _ = self._run_benchmark(dump=False)
        self.assertEqual(result, ("", "net_path"))

    def test_benchmark_run_dump_missing(self):
        with self.assertRaises(utils.AccuracyCompareException):
            self._run_benchmark(dump=True, dump_exist=False)

    def test_benchmark_run_net_output_missing(self):
        with self.assertRaises(utils.AccuracyCompareException):
            self._run_benchmark(dump=True, net_exist=False)

if __name__ == '__main__':
    unittest.main()
