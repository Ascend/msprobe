# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# -------------------------------------------------------------------------

import os

from msprobe.core.common.cli_help import MindStudioArgumentParser


def test_unified_help_has_required_sections_in_order():
    parser = MindStudioArgumentParser(prog="msprobe parse")
    parser.add_argument("-d", "--dump_path", required=True, help="<Required> Dump data to parse")
    parser.add_argument("-t", "--type", choices=["npy", "pt"], default="pt", help="Output type")
    parser.add_argument(
        "-o", "--output_path", default="./output", help="<Optional> Output directory. Default path: ./output"
    )

    help_text = parser.format_help()

    section_offsets = [
        help_text.index("Description:"),
        help_text.index("Usage:"),
        help_text.index("Required arguments:"),
        help_text.index("Optional arguments:"),
        help_text.index("Examples:"),
        help_text.index("Output:"),
    ]
    assert section_offsets == sorted(section_offsets)
    assert "--dump_path <FILE_OR_DIR>" in help_text
    assert "--type {npy,pt}" in help_text
    assert "[default: pt]" in help_text
    assert "[default: ./output]" in help_text
    assert "<Required>" not in help_text
    assert "Default path:" not in help_text


def test_unified_help_omits_empty_output_and_required_sections():
    parser = MindStudioArgumentParser(prog="msprobe install_deps")
    parser.add_argument("--no_check", action="store_true", help="Skip certificate checks")

    help_text = parser.format_help()

    assert "Required arguments:" not in help_text
    assert "Output:" not in help_text
    assert "Examples:" in help_text
    assert "--no_check" in help_text
    assert "[default: off]" in help_text


def test_generated_usage_reuses_action_metavar():
    test_cases = (
        ("anomaly_processor", "-d", "--data_path", "data_path_dir"),
        ("gen_model_config", None, "--model-path", "model_path"),
    )

    for prog, short_option, long_option, dest in test_cases:
        parser = MindStudioArgumentParser(prog=prog)
        options = [long_option] if short_option is None else [short_option, long_option]
        parser.add_argument(*options, dest=dest, required=True, metavar="<DIR>")

        help_text = parser.format_help()
        usage_line = next(line for line in help_text.splitlines() if line.startswith(f"  {prog} "))
        required_section = help_text.split("Required arguments:\n", 1)[1].split("\n\n", 1)[0]
        argument_line = next(line for line in required_section.splitlines() if long_option in line)

        assert f"{long_option} <DIR>" in usage_line
        assert f"{long_option} <DIR>" in argument_line


def test_short_only_option_does_not_render_none():
    parser = MindStudioArgumentParser(prog="short_only")
    parser.add_argument("-i", metavar="<FILE>", help="Input file")

    help_text = parser.format_help()

    assert "None" not in help_text
    assert "-i <FILE>" in help_text


def test_parameter_descriptions_follow_terminal_width(monkeypatch):
    monkeypatch.setattr(
        "msprobe.core.common.cli_help.shutil.get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((60, 24)),
    )
    parser = MindStudioArgumentParser(prog="narrow_help")
    parser.add_argument(
        "--name",
        help="A deliberately long description that must wrap instead of overflowing a narrow terminal window.",
    )

    help_text = parser.format_help()
    optional_section = help_text.split("Optional arguments:\n", 1)[1].split("\n\n", 1)[0]

    assert max(len(line) for line in optional_section.splitlines()) <= 60


def test_acc_check_help_lists_aliases_examples_and_outputs():
    for command in ("acc_check", "multi_acc_check"):
        parser = MindStudioArgumentParser(prog=f"msprobe {command}")
        parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", required=True)

        help_text = parser.format_help()

        assert "-api_info, --api_info_file <FILE>" in help_text
        assert f"msprobe {command} -api_info ./dump_source/step0/rank0/dump.json -h" in help_text
        assert "<DIR>/accuracy_checking_result_{timestamp}.csv" in help_text
        assert "<DIR>/accuracy_checking_details_{timestamp}.csv" in help_text
        assert "<out_path>" not in help_text


def test_framework_specific_acc_check_help():
    test_cases = (
        ("msprobe acc_check pytorch", "PyTorch", False),
        ("msprobe acc_check mindspore", "MindSpore", False),
        ("msprobe multi_acc_check pytorch", "PyTorch", True),
        ("msprobe multi_acc_check mindspore", "MindSpore", True),
    )

    for help_spec_key, framework, is_multi in test_cases:
        prog = "msprobe multi_acc_check" if is_multi else "msprobe acc_check"
        parser = MindStudioArgumentParser(prog=prog, help_spec_key=help_spec_key)
        parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", required=True)
        parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path")
        if framework == "PyTorch":
            parser.add_argument("-config", "--config_path", dest="config_path")

        help_text = parser.format_help()

        assert f"Run {framework} API accuracy checks" in help_text
        assert "-api_info, --api_info_file <FILE>" in help_text
        assert "-csv_path, --result_csv_path <FILE>" in help_text
        assert "<DIR>/accuracy_checking_result_{timestamp}.csv" in help_text
        assert "<DIR>/accuracy_checking_details_{timestamp}.csv" in help_text
        if framework == "PyTorch":
            assert "-config,   --config_path <FILE>" in help_text


def test_integer_range_choices_use_numeric_metavar():
    parser = MindStudioArgumentParser(prog="range_choices")
    parser.add_argument(
        "-n",
        "--num_splits",
        type=int,
        choices=range(1, 65),
        default=8,
        help="Number of splits for parallel processing. Range: 1-64",
    )

    help_text = parser.format_help()

    assert "-n, --num_splits <N>" in help_text
    assert "{1,2,3" not in help_text
    assert "Range: 1-64." in help_text
    assert "[default: 8]" in help_text
