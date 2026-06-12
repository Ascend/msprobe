import unittest
from unittest.mock import patch, MagicMock
import argparse

# 导入被测模块
from msprobe.infer.offline.compare.msquickcmp.main import offline_dump_cli, compare_offline_model_mode, \
    set_args_default, check_compare_args, install_offline_deps_cli


class TestOfflineDumpCli(unittest.TestCase):

    @patch("msprobe.infer.offline.compare.msquickcmp.main.dump_process")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.DumpArgsAdapter")
    def test_offline_dump_cli(self, mock_adapter, mock_dump):
        args = argparse.Namespace(
            model_path="/tmp/model.om",
            input_data="data.bin",
            output_path="./out",
            input_shape="a:1,2,3",
            rank="0",
            dym_shape_range="",
            onnx_fusion_switch=True,
            output_size=""
        )

        offline_dump_cli(args)

        mock_adapter.assert_called_once()
        mock_dump.assert_called_once()


class TestCompareOfflineModelMode(unittest.TestCase):

    @patch("msprobe.infer.offline.compare.msquickcmp.main.set_args_default")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_compare_args")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.print_compare_ends_info")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.cmp_process")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.CmpArgsAdapter")
    def test_compare(self, mock_adapter, mock_cmp_process,
                     mock_print_end, mock_check, mock_set_default):
        mock_set_default.side_effect = lambda x: x

        args = argparse.Namespace(
            golden_path="/tmp/golden.onnx",
            target_path="/tmp/model.om",
            input_data="",
            output_path="./out",
            input_shape="",
            rank="0",
            output_size="",
            dym_shape_range="",
            onnx_fusion_switch=True
        )

        compare_offline_model_mode(args)

        mock_set_default.assert_called_once()
        mock_check.assert_called_once()
        mock_adapter.assert_called_once()
        mock_cmp_process.assert_called_once()
        mock_print_end.assert_called_once()

    @patch("msprobe.infer.offline.compare.msquickcmp.main.set_args_default")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_compare_args")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.logger")
    def test_compare_missing_golden(self, mock_logger, mock_check, mock_set_default):
        mock_set_default.side_effect = lambda x: x
        args = argparse.Namespace(golden_path=None)

        compare_offline_model_mode(args)

        mock_logger.error.assert_called_once()


class TestSetArgsDefault(unittest.TestCase):

    def test_set_defaults(self):
        args = argparse.Namespace()
        result = set_args_default(args)

        self.assertEqual(result.rank, "0")
        self.assertEqual(result.input_data, "")
        self.assertEqual(result.input_shape, "")
        self.assertEqual(result.output_size, "")
        self.assertEqual(result.dym_shape_range, "")
        self.assertTrue(result.onnx_fusion_switch)


class TestCheckCompareArgs(unittest.TestCase):

    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_file_or_directory_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_output_path_legality")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_input_data_path")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_dict_kind_string")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_rank_range_valid")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_number_list")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.check_dym_range_string")
    @patch("msprobe.infer.offline.compare.msquickcmp.main.str2bool")
    def test_check_args(
        self, mock_bool, mock_dym, mock_num, mock_rank, mock_dict,
        mock_input, mock_out, mock_file
    ):
        mock_bool.return_value = False
        args = argparse.Namespace(
            target_path="/tmp/m.om",
            golden_path="/tmp/g.onnx",
            output_path="./out",
            input_data="abc.bin",
            input_shape="a:1,2",
            rank="0",
            output_size="1,2",
            dym_shape_range="a:1~2",
            onnx_fusion_switch="False"
        )

        check_compare_args(args)

        mock_file.assert_any_call("/tmp/m.om", False, False, [".om"])
        mock_file.assert_any_call("/tmp/g.onnx", False, False, [".onnx", ".om"])
        mock_out.assert_called_once_with("./out")
        mock_input.assert_called_once_with("abc.bin")
        mock_dict.assert_called_once_with("a:1,2")
        mock_rank.assert_called_once_with("0")
        mock_num.assert_called_once_with("1,2")
        mock_dym.assert_called_once_with("a:1~2")
        mock_bool.assert_called_once_with("False")


class TestInstallOfflineDepsCli(unittest.TestCase):

    @patch("msprobe.infer.offline.compare.msquickcmp.main.subprocess.run")
    def test_install_offline_deps(self, mock_run):
        args = argparse.Namespace(no_check=True)

        install_offline_deps_cli(args)

        self.assertEqual(mock_run.call_count, 2)

        pip_call_args = mock_run.call_args_list[0][0][0]
        self.assertEqual(pip_call_args[1], "-m")
        self.assertEqual(pip_call_args[2], "pip")
        self.assertEqual(pip_call_args[3], "install")
        self.assertIn("onnx>=1.14.0", pip_call_args)
        self.assertIn("onnxruntime>=1.14.1,!=1.16.0", pip_call_args)

        shell_call_args = mock_run.call_args_list[1][0][0]
        self.assertEqual(shell_call_args[0], "/bin/bash")
        self.assertEqual(shell_call_args[-1], "True")


if __name__ == "__main__":
    unittest.main()
