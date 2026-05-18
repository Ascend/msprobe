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

from enum import Enum
import re
from msprobe.visualization.builder.msprobe_adapter import op_patterns
from msprobe.core.common.log import logger


class NodeOp(Enum):
    module = 0
    function_api = 1
    api_collection = 9

    @staticmethod
    def get_node_op(node_name: str):
        """
        基于代表节点的字符串，解析节点种类
        """
        for op in NodeOp:
            index = op.value
            if index < 0 or index >= len(op_patterns):
                continue
            pattern = op_patterns[index]
            if re.match(pattern, node_name):
                return op
        logger.debug(f"Cannot parse node_name {node_name} into NodeOp, default parsing as module.")
        return NodeOp.module
