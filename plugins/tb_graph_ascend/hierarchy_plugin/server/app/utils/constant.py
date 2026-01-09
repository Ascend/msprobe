# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from enum import Enum
ADD_MATCH_KEYS = [
    'MaxAbsErr',
    'MinAbsErr',
    'MeanAbsErr',
    'NormAbsErr',
    'MaxRelativeErr',
    'MinRelativeErr',
    'MeanRelativeErr',
    'NormRelativeErr',
]
MAX_FILE_SIZE = 15 * 1024 * 1024 * 1024  # 最大文件大小限制
NPU_PREFIX = 'N___'
BENCH_PREFIX = 'B___'
FILE_NAME_REGEX = r'^[a-zA-Z0-9_\-\.]+$'  # 文件名正则表达式
COLOR_PATTERN = r'^#[0-9A-Fa-f]{6}$'
# 未匹配节点值
UN_MATCHED_VALUE = -1
# 图类型
NPU = 'NPU'
BENCH = 'Bench'
SINGLE = 'Single'

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
    DB = '.vis.db'
    JSON = '.vis'


class DataType(Enum):
    DB = 'db'
    JSON = 'json'
