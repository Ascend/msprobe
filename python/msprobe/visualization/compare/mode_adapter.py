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

import math
import json
from msprobe.core.common.const import CompareConst, Const
from msprobe.visualization.utils import ToolTip, GraphConst


class ModeAdapter:
    def __init__(self, compare_mode, parallel_merge=False):
        self.compare_mode = compare_mode
        self.csv_data = []
        self.compare_nodes = []
        self.parallel_merge = parallel_merge

    @staticmethod
    def _is_invalid(value):
        if not isinstance(value, float):
            return False
        return math.isnan(value) or math.isinf(value)

    @staticmethod
    def _add_compare_data(node_data, compare_data_dict, key_list, headers):
        for key, data_info in node_data.items():
            if not isinstance(data_info, dict):
                continue
            compare_data = compare_data_dict.get(key)
            if compare_data:
                # 对应比对结果csv的列
                id_list = [headers.index(x) for x in key_list]
                ModeAdapter._match_data(data_info, compare_data, key_list, id_list)

    @staticmethod
    def _match_data(data_dict, compare_data, key_list, id_list):
        """
        绑定精度指标到node的input_data和output_data
        """
        if len(key_list) != len(id_list):
            return
        for id_val, key in zip(id_list, key_list):
            data_dict[key] = compare_data[id_val]

    @staticmethod
    def _check_list_len(data_list, len_num):
        if len(data_list) < len_num:
            raise ValueError(f"compare_data_dict_list must contain at least {len_num} items.")

    def parse_result(self, node, compare_data_dict, api_indicator):
        """
        根据结果返回precision_index
        """
        if self.compare_mode == GraphConst.MD5_COMPARE:
            key_list = GraphConst.MD5_INDEX_LIST
            headers = CompareConst.MD5_COMPARE_RESULT_HEADER
        elif self.compare_mode == GraphConst.SUMMARY_COMPARE:
            key_list = GraphConst.SUMMARY_INDEX_LIST
            headers = CompareConst.SUMMARY_COMPARE_RESULT_HEADER
        else:
            key_list = GraphConst.REAL_DATA_INDEX_LIST
            headers = CompareConst.COMPARE_RESULT_HEADER
        ModeAdapter._add_compare_data(node.input_data, compare_data_dict, key_list, headers)
        ModeAdapter._add_compare_data(node.output_data, compare_data_dict, key_list, headers)
        precision_index = GraphConst.COMPARE_INDICATOR_TO_PRECISION_INDEX_MAPPING.get(api_indicator, CompareConst.PASS)
        precision_index = self._ignore_precision_index(node.id, precision_index)
        return precision_index

    def prepare_real_data(self, node):
        """
        为真实数据比较模式准备节点信息
        """
        if self.compare_mode == GraphConst.REAL_DATA_COMPARE:
            self.compare_nodes.append(node)
            return True
        return False

    def add_csv_data(self, compare_result_list):
        if self.compare_mode != GraphConst.REAL_DATA_COMPARE:
            return
        self.csv_data.extend(compare_result_list)

    def get_tool_tip(self):
        """
        用于前端展示字段的具体含义
        """
        if self.compare_mode == GraphConst.SUMMARY_COMPARE:
            tips = {
                CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
                CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
                CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
                CompareConst.NORM_DIFF: ToolTip.NORM_DIFF,
            }
        elif self.compare_mode == GraphConst.MD5_COMPARE:
            tips = {Const.MD5: ToolTip.MD5}
        else:
            # tips 前端已经不展示，全部设置为""
            algorithm_columns = CompareConst.ALL_COMPARE_INDEX
            tips = {col: "" for col in algorithm_columns}
        return json.dumps(tips)

    def _ignore_precision_index(self, node_id, precision_index):
        node_id_split = node_id.split(Const.SEP)
        if len(node_id_split) < 2:
            return precision_index
        if node_id.split(Const.SEP)[1] in GraphConst.IGNORE_PRECISION_INDEX:
            return GraphConst.MAX_INDEX_KEY if self.compare_mode == GraphConst.MD5_COMPARE else GraphConst.MIN_INDEX_KEY
        return precision_index
