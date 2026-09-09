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

import argparse
import json
import os
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.common.cli_help import MindStudioArgumentParser


def _detect_framework_from_api_info(api_info_path: str) -> str:
    """从 -api_info 指定的 dump.json 中读取 framework 字段."""
    check_file_or_directory_path(api_info_path, False)
    if not api_info_path:
        raise ValueError("Argument -api_info is required to detect framework.")

    if not os.path.exists(api_info_path):
        raise FileNotFoundError(f"api_info file does not exist: {api_info_path}")

    with open(api_info_path, "r", encoding="utf-8") as f:
        try:
            info = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {api_info_path}: {e}") from e

    framework = info.get("framework")
    if not framework:
        raise ValueError(f'Key "framework" not found in {api_info_path}')

    framework = str(framework).lower()
    # 统一映射到 Const 中定义的名字，方便复用其他逻辑
    if framework in ("pytorch", "pt", Const.PT_FRAMEWORK.lower()):
        return Const.PT_FRAMEWORK
    if framework in ("mindspore", "ms", Const.MS_FRAMEWORK.lower(), Const.MT_FRAMEWORK.lower()):
        return Const.MS_FRAMEWORK

    raise ValueError(f"Unsupported framework in api_info: {framework}")


class _DeferredHelpAction(argparse._HelpAction):
    """将 -h/--help 延迟到未知参数校验之后处理，避免错误参数被帮助信息掩盖。"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "help_requested", True)


def _parse_api_info_file(argv):
    """预解析 -api_info/--api_info_file，用于确定校验框架。"""
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("-api_info", "--api_info_file", dest="api_info_file")
    pre_args, _ = pre_parser.parse_known_args(argv)
    return pre_args.api_info_file


def _validate_api_info_option(argv, prog):
    """只允许精确的 -api_info / --api_info_file，其它任何前缀缩写形式一律报错。

    argparse 的 allow_abbrev 只能关闭长选项（--）的前缀匹配，单短横线多字符选项
    （如 -api_info）仍会无条件做前缀匹配，因此这里对两个 api_info 选项的任意前缀做精确校验。
    """
    for token in argv:
        if token == "--":  # nosec B105
            break
        name = token.split("=", 1)[0]
        if name in ("-api_info", "--api_info_file"):
            continue
        if name.startswith("-") and name != "-":
            if "-api_info".startswith(name) or "--api_info_file".startswith(name):
                _build_precheck_parser(prog, prog).error(f"unrecognized arguments: {name}")


def _add_help_flag(parser):
    parser.add_argument("-h", "--help", action=_DeferredHelpAction)


def _build_precheck_parser(prog, help_spec_key):
    """构建预检 parser，保证非法/缺失 -api_info 时报错 usage 一致。"""
    parser = MindStudioArgumentParser(prog=prog, help_spec_key=help_spec_key, add_help=False, allow_abbrev=False)
    _add_help_flag(parser)
    parser.add_argument(
        "-api_info",
        "--api_info_file",
        dest="api_info_file",
        help="Path to the API info JSON file used to determine the check framework.",
    )
    return parser


def _parse_args_strict(parser, argv):
    """先校验未知参数（即使带 -h 也报错），再处理 -h，最后返回解析结果。"""
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if getattr(args, "help_requested", False):
        parser.print_help()
        parser.exit()
    return args


def _error_on_missing_api_info(prog, help_spec_key, argv):
    """缺少 -api_info 时触发标准报错；-h/--help 显示帮助，未知参数优先报错。"""
    parser = _build_precheck_parser(prog, help_spec_key)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if getattr(args, "help_requested", False):
        parser.print_help()
        parser.exit()
    parser.error("the following arguments are required: -api_info/--api_info_file")


def acc_check_cli(argv):
    """
    msprobe acc_check ... 的统一入口。
    1. 先解析出 -api_info
    2. 根据 dump.json 中的 framework 动态选择 PT/MS 的 parser + 命令。
    """
    _validate_api_info_option(argv, "msprobe acc_check")
    api_info_file = _parse_api_info_file(argv)

    # 缺少必填参数 -api_info：报错（而非静默打印帮助）；-h 仍显示帮助
    if api_info_file is None:
        _error_on_missing_api_info("msprobe acc_check", "msprobe acc_check", argv)
        return

    framework = _detect_framework_from_api_info(api_info_file)

    if framework == Const.PT_FRAMEWORK:
        # PyTorch 路径：使用原来的 PT acc_check 实现
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser, acc_check_command

        pt_parser = MindStudioArgumentParser(
            prog="msprobe acc_check",
            help_spec_key="msprobe acc_check pytorch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run PyTorch acc_check with msprobe.",
            add_help=False,
            allow_abbrev=False,
        )
        _add_help_flag(pt_parser)
        _acc_check_parser(pt_parser)  # 这里会给 parser 加上原来的所有 PT acc_check 参数（包括 -api_info）

        pt_args = _parse_args_strict(pt_parser, argv)
        acc_check_command(pt_args)

    elif framework == Const.MS_FRAMEWORK:
        # MindSpore 路径：复用原来的 MS api_checker_main
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import add_api_accuracy_checker_argument
        from msprobe.mindspore.api_accuracy_checker.main import api_checker_main

        ms_parser = MindStudioArgumentParser(
            prog="msprobe acc_check",
            help_spec_key="msprobe acc_check mindspore",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run MindSpore Check  with msprobe.",
            add_help=False,
            allow_abbrev=False,
        )
        _add_help_flag(ms_parser)
        add_api_accuracy_checker_argument(ms_parser)  # 给 acc_check 的 parser 加上原来 MS 的所有参数

        ms_args = _parse_args_strict(ms_parser, argv)
        api_checker_main(ms_args)


def multi_acc_check_cli(argv):
    """
    msprobe multi_acc_check ... 的统一入口。
    同样通过 -api_info -> dump.json -> framework 做分发。
    """
    _validate_api_info_option(argv, "msprobe multi_acc_check")
    api_info_file = _parse_api_info_file(argv)

    # 缺少必填参数 -api_info：报错（而非静默打印帮助）；-h 仍显示帮助
    if api_info_file is None:
        _error_on_missing_api_info("msprobe multi_acc_check", "msprobe multi_acc_check", argv)
        return

    framework = _detect_framework_from_api_info(api_info_file)

    if framework == Const.PT_FRAMEWORK:
        # PyTorch 多进程路径：沿用原来的 prepare_config + run_parallel_ut
        from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser
        from msprobe.pytorch.api_accuracy_checker.acc_check.multi_acc_check import prepare_config, run_parallel_ut

        pt_parser = MindStudioArgumentParser(
            prog="msprobe multi_acc_check",
            help_spec_key="msprobe multi_acc_check pytorch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run PyTorch acc_check in parallel with msprobe.",
            add_help=False,
            allow_abbrev=False,
        )
        _add_help_flag(pt_parser)
        _acc_check_parser(pt_parser)
        pt_parser.add_argument(
            "-n",
            "--num_splits",
            type=int,
            choices=range(1, 65),
            default=8,
            help="Number of splits for parallel processing. Range: 1-64",
        )

        pt_args = _parse_args_strict(pt_parser, argv)
        config = prepare_config(pt_args)
        run_parallel_ut(config)

    elif framework == Const.MS_FRAMEWORK:
        # MindSpore 多进程路径：沿用原来的 multi_add_api_accuracy_checker_argument + mul_api_checker_main
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import multi_add_api_accuracy_checker_argument
        from msprobe.mindspore.api_accuracy_checker.main import mul_api_checker_main

        ms_parser = MindStudioArgumentParser(
            prog="msprobe multi_acc_check",
            help_spec_key="msprobe multi_acc_check mindspore",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run MindSpore Check in parallel with msprobe.",
            add_help=False,
            allow_abbrev=False,
        )
        _add_help_flag(ms_parser)
        multi_add_api_accuracy_checker_argument(ms_parser)

        ms_args = _parse_args_strict(ms_parser, argv)
        mul_api_checker_main(ms_args)
