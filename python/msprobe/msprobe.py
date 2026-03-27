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
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'msaccucmp'))
from msprobe.core.common.log import logger
from msprobe.core.compare.utils import _compare_parser
from msprobe.core.compare.compare_cli import compare_cli
from msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli
from msprobe.core.config_check.config_check_cli import _config_checking_parser, _run_config_checking_command
from msprobe.overflow_check.analyzer import _overflow_check_parser, _run_overflow_check
from msprobe.core.acc_check.acc_check_cli import acc_check_cli, multi_acc_check_cli
from msprobe.visualization.graph_service import _graph_service_parser, _graph_service_command
from msprobe.core.compare.data2db import _data2db_service_parser, _data2db_command
from msprobe.infer.offline.compare.msquickcmp.main import _offline_dump_parser, offline_dump_cli
from msprobe.core.install_deps.install_deps import _install_deps_parser, install_deps_cli
from msprobe.core.parse.parse_cli import _parse_parser, parse_cli


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="msprobe(mindstudio probe), [Powered by MindStudio].\n"
                    "A full-process, all-scenario precision tool base on Ascend products.\n"
                    f"For any issue, refer README.md first",
    )

    parser.set_defaults(print_help=parser.print_help)
    subparsers = parser.add_subparsers()

    compare_parser = subparsers.add_parser('compare')
    _compare_parser(compare_parser)
    acc_check_cmd_parser = subparsers.add_parser('acc_check')
    multi_acc_check_cmd_parser = subparsers.add_parser('multi_acc_check')
    merge_result_parser = subparsers.add_parser('merge_result')
    _merge_result_parser(merge_result_parser)
    overflow_check_parse = subparsers.add_parser('overflow_check')
    _overflow_check_parser(overflow_check_parse)
    config_checking_parser = subparsers.add_parser('config_check')
    _config_checking_parser(config_checking_parser)
    # api_precision_compare 是 PyTorch 相关能力，延迟加载
    api_precision_compare_cmd_parser = subparsers.add_parser('api_precision_compare')
    try:
        from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import (
            _api_precision_compare_parser
        )
        _api_precision_compare_parser(api_precision_compare_cmd_parser)
    except ImportError as e:
        # torch 不存在时，parser 仍然存在，但给提示
        def _no_torch_parser(_):
            api_precision_compare_cmd_parser.set_defaults(
                func=lambda *_: (
                    logger.error(
                        "api_precision_compare requires PyTorch environment. "
                        "Please install torch / torch_npu first."
                    ),
                    sys.exit(1)
                )
            )
        _no_torch_parser(api_precision_compare_cmd_parser)
    graph_service_cmd_parser = subparsers.add_parser('graph_visualize')
    _graph_service_parser(graph_service_cmd_parser)
    data2db_parser = subparsers.add_parser('data2db')
    _data2db_service_parser(data2db_parser)
    
    parse_parser = subparsers.add_parser('parse')
    _parse_parser(parse_parser)

    offline_dump_parser = subparsers.add_parser('offline_dump')
    _offline_dump_parser(offline_dump_parser)

    install_deps_cmd_parser = subparsers.add_parser('install_deps')
    _install_deps_parser(install_deps_cmd_parser)

    if len(sys.argv) >= 2 and sys.argv[1] == "acc_check":
        acc_check_cli(sys.argv[2:])
        return
    elif len(sys.argv) >= 2 and sys.argv[1] == "multi_acc_check":
        multi_acc_check_cli(sys.argv[2:])
        return
    elif len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(sys.argv[1:])
    if sys.argv[1] == "compare":
        compare_cli(args, sys.argv[1:])
    elif sys.argv[1] == "merge_result":
        merge_result_cli(args)
    elif sys.argv[1] == "overflow_check":
        _run_overflow_check(args)
    elif sys.argv[1] == "graph_visualize":
        _graph_service_command(args)
    elif sys.argv[1] == "api_precision_compare":
        try:
            from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import (
                _api_precision_compare_command
            )
            _api_precision_compare_command(args)
        except ImportError:
            logger.error(
                "api_precision_compare requires PyTorch environment. "
                "Please install torch / torch_npu first."
            )
            sys.exit(1)

    elif sys.argv[1] == "config_check":
        _run_config_checking_command(args)
    elif sys.argv[1] == "data2db":
        _data2db_command(args)

    elif sys.argv[1] == "offline_dump":
        offline_dump_cli(args)
    elif sys.argv[1] == "install_deps":
        install_deps_cli(args)
    elif sys.argv[1] == "parse":
        parse_cli(args)


if __name__ == "__main__":
    main()
