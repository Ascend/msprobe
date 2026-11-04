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


import os
import re
import sys
import subprocess  # nosec B404

from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException


def compare_offline_data_mode(args):
    cmd_args = []

    # 必需参数
    if args.target_path:
        cmd_args.extend(['-m', args.target_path])

    if args.golden_path:
        cmd_args.extend(['-g', args.golden_path])

    # 可选参数
    if args.fusion_rule_file:
        cmd_args.extend(['-f', args.fusion_rule_file])

    if args.quant_fusion_rule_file:
        cmd_args.extend(['-q', args.quant_fusion_rule_file])

    if args.close_fusion_rule_file:
        cmd_args.extend(['-cf', args.close_fusion_rule_file])

    model_graph_file = getattr(args, 'model_graph_file', '')
    if model_graph_file:
        cmd_args.extend(['-mgf', model_graph_file])

    if args.output_path:
        cmd_args.extend(['-out', args.output_path])

    call_msaccucmp(cmd_args)


def _get_msaccucmp_script_path():
    """
    获取 msprobe 目录下的 msaccucmp.py 脚本路径

    Returns:
        str: msaccucmp.py 的绝对路径
    """
    current_file = os.path.abspath(__file__)
    # offline_data_compare.py -> compare -> core -> msprobe
    # 需要向上3级目录，然后进入 msaccucmp/msaccucmp.py
    msprobe_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    script_path = os.path.join(msprobe_dir, "msaccucmp", "msaccucmp.py")

    if os.path.exists(script_path):
        return os.path.abspath(script_path)

    error_msg = "msaccucmp.py file not found in msprobe package"
    logger.error(error_msg)
    raise CompareException(CompareException.INVALID_PATH_ERROR)


def _replace_log_parameters(log_line):
    """
    替换日志中的参数，将旧参数格式替换为新格式
    -q -> -qfr
    -f -> -fr
    -cf -> -cfr
    -m -> -tp
    -g -> -gp
    -out -> -o

    只替换独立的参数（前后有空格、行首或行尾），不替换单词中的字符

    Args:
        log_line: 原始日志行

    Returns:
        str: 替换后的日志行
    """
    replacements = [
        ('-out', r'(?<![-\w])-out(?![-\w])', '-o'),  # -out -> -o
        ('-cf', r'(?<![-\w])-cf(?![-\w])', '-cfr'),  # -cf -> -cfr
        ('-q', r'(?<![-\w])-q(?![-\w])', '-qfr'),  # -q -> -qfr
        ('-f', r'(?<![-\w])-f(?![-\w])', '-fr'),  # -f -> -fr
        ('-m', r'(?<![-\w])-m(?![-\w])', '-tp'),  # -m -> -tp
        ('-g', r'(?<![-\w])-g(?![-\w])', '-gp'),  # -g -> -gp
    ]

    result = log_line
    for search_str, pattern, replacement in replacements:
        # 只有当日志行中包含要搜索的字符串时，才进行正则替换
        if search_str in result:
            result = re.sub(pattern, replacement, result)

    return result


def call_msaccucmp(cmd_args):
    """
    调用 msaccucmp.py 工具，透传从 args 中提取的参数。

    Args:
        cmd_args: 命令行参数列表

    Returns:
        subprocess.Popen: 命令执行进程对象
    """
    msaccucmp_script_path = _get_msaccucmp_script_path()

    # 构建完整命令
    python_cmd = sys.executable  # 使用当前 Python 解释器
    full_cmd = [python_cmd, msaccucmp_script_path, "compare"] + cmd_args

    logger.info(f"Calling msaccucmp with command: {' '.join(full_cmd)}")

    # 执行命令
    try:
        # The command uses an argument list with shell disabled; the process is closed and waited below.
        # pylint: disable=consider-using-with
        process = subprocess.Popen(  # nosec B603
            full_cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ''):
            if not re.match(r'^\d{4}-\d{2}-\d{2}', line):
                continue
            # 替换日志中的参数格式
            replaced_line = _replace_log_parameters(line)
            logger.raw(replaced_line)

        process.stdout.close()
        process.wait()

        return_code = process.returncode

        if return_code not in (0, 2):
            logger.error(f"msaccucmp execution failed with return code {return_code}")
        else:
            logger.info("msaccucmp execution completed successfully")

        return process

    except Exception as e:
        logger.error(f"Failed to execute msaccucmp: {str(e)}")
        raise CompareException(CompareException.UNKNOWN_ERROR) from e
