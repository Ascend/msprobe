# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# ==============================================================================

from enum import Enum

# 允许执行的命令白名单
ALLOWED_COMMANDS = {"msprobe"}
# 禁止的参数模式（注入攻击特征）
FORBIDDEN_ARG_PATTERNS = [
    r"\|",  # 管道符
    r";",  # 命令分隔符
    r"&&",  # 逻辑与
    r"\|\|",  # 逻辑或
    r"`",  # 反引号命令替换
    r"\$\(",  # 命令替换
    r"\n",  # 换行符注入
    r"\r",  # 回车符注入
    r">>",  # 追加重定向
    r"[<>]",  # 重定向
    r"\\x",  # 十六进制转义
]

security_headers = {
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; script-src 'unsafe-inline'; "
        "style-src 'unsafe-inline'; img-src 'self' data:; font-src data:;"
    ),
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

ADD_MATCH_KEYS = [
    "MaxAbsErr",
    "MinAbsErr",
    "MeanAbsErr",
    "NormAbsErr",
    "MaxRelativeErr",
    "MinRelativeErr",
    "MeanRelativeErr",
    "NormRelativeErr",
]
MAX_FILE_SIZE = 100 * 1024 * 1024 * 1024  # 最大文件大小限制
NPU_PREFIX = "N___"
BENCH_PREFIX = "B___"
FILE_NAME_REGEX = r"^[a-zA-Z0-9_\-\.]+$"  # 文件名正则表达式
COLOR_PATTERN = r"^#[0-9A-Fa-f]{6}$"
# 未匹配节点值
UN_MATCHED_VALUE = -1
# 图类型
NPU = "NPU"
BENCH = "Bench"
SINGLE = "Single"

# 前端节点类型
EXPAND_MODULE = 0
UNEXPAND_NODE = 1

# 权限码
PERM_GROUP_WRITE = 0o020
PERM_OTHER_WRITE = 0o002

# 后端节点类型
MODULE = 0
API = 1
MULTI_COLLECTION = 8
API_LIST = 9

# 计算指标
MAX_RELATIVE_ERR = "0"
MIN_RELATIVE_ERR = "1"
MEAN_RELATIVE_ERR = "2"
NORM_RELATIVE_ERR = "3"


class Extension(Enum):
    DB = ".vis.db"


class DataType(Enum):
    DB = "db"
