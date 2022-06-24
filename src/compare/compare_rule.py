#!/usr/bin/env python
# coding=utf-8
"""
Function:
VectorComparison class. This class mainly involves the compare function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import os
import sys
import utils
from fusion_rule_parser import FusionRuleParser
from fusion_rule_parser import merge_fusion_rule
from fusion_rule_parser import merge_close_and_open_fusion_rule
from fusion_op import FusionOp
from fusion_op import OpAttr
from const_manager import ConstManager
from dump import CompareData
from compare_error import CompareError


class CompareRule:
    """
    The class for compare rule
    """

    def __init__(self: any, fusion_json_file_path: str, quant_fusion_rule_file_path: str,
                 close_fusion_rule_file_path: str = '') -> None:
        self.fusion_json_file_path = ""
        if fusion_json_file_path != "":
            self.fusion_json_file_path = os.path.realpath(fusion_json_file_path)
        self.quant_fusion_rule_file_path = ""
        if quant_fusion_rule_file_path != "":
            self.quant_fusion_rule_file_path = os.path.realpath(quant_fusion_rule_file_path)
        self.close_fusion_rule_file_path = ""
        if close_fusion_rule_file_path != "":
            self.close_fusion_rule_file_path = os.path.realpath(close_fusion_rule_file_path)
        self.fusion_info = None

    @staticmethod
    def _sort_file_by_timestamp(op_name_to_file_map: dict) -> list:
        origin_list = []
        for (op_name, file_list) in list(op_name_to_file_map.items()):
            for item in file_list:
                index = item.rfind(".")
                if index == -1:
                    # when index is 0, item is dump file ,the name is hash value.
                    # Example: the file name is only numeric.
                    timestamp = int(os.path.basename(item))
                else:
                    # when index is not 0, item is dump file,the name meet the data format.
                    # Example: {op_type}.{op_name}.{task_id}.{timestamp}
                    timestamp = int(item[index + 1:])
                origin_list.append([timestamp, op_name, item])
        return sorted(origin_list, key=lambda s: s[0])

    @staticmethod
    def _make_my_output_map(my_output_sort_list: list, op_name_to_op_map: dict) -> None:
        attr = OpAttr([], '', False, 0)
        # make my output fusion op by my_output_sort_list
        for item in my_output_sort_list:
            op_name = item[1]
            if op_name not in op_name_to_op_map:
                fusion_op = FusionOp(len(op_name_to_op_map), op_name, [], ConstManager.LEFT_TYPE, [item[2]], attr)
                op_name_to_op_map[op_name] = [fusion_op]
            else:
                op_name_to_op_map[op_name][0].output_desc.append(item[2])

    @staticmethod
    def _make_ground_truth_map(ground_truth_sort_list: list, op_name_to_op_map: dict) -> None:
        attr = OpAttr([], '', False, 0)
        # make ground truth fusion op by ground_truth_sort_list
        for item in ground_truth_sort_list:
            op_name = item[1]
            if op_name not in op_name_to_op_map:
                fusion_op = FusionOp(len(op_name_to_op_map), op_name, [], ConstManager.RIGHT_TYPE, [item[2]], attr)
                op_name_to_op_map[op_name] = [fusion_op]
            else:
                fusion_op_info = op_name_to_op_map[op_name][-1]
                if fusion_op_info.op_type == ConstManager.RIGHT_TYPE:
                    op_name_to_op_map[op_name][-1].output_desc.append(item[2])
                else:
                    op_id = op_name_to_op_map[op_name][0].op_id
                    fusion_op = FusionOp(op_id, op_name, [], ConstManager.RIGHT_TYPE, [item[2]], attr)
                    op_name_to_op_map[op_name].append(fusion_op)

    def parse_fusion_rule(self: any, compare_data: CompareData) -> None:
        """
        Parse fusion rule
        :param compare_data: compare data
        """

        if self.fusion_json_file_path != "" and self.quant_fusion_rule_file_path != "":
            offline_fusion_rule = FusionRuleParser(self.fusion_json_file_path)
            offline_fusion_rule.analysis_fusion_rule()
            quant_fusion_rule = FusionRuleParser(self.quant_fusion_rule_file_path)
            quant_fusion_rule.analysis_fusion_rule()
            self.fusion_info = merge_fusion_rule(offline_fusion_rule, quant_fusion_rule)
        elif self.fusion_json_file_path != "" and self.close_fusion_rule_file_path != "":
            open_fusion_rule = FusionRuleParser(self.fusion_json_file_path)
            open_fusion_rule.analysis_fusion_rule()
            close_fusion_rule = FusionRuleParser(self.close_fusion_rule_file_path)
            close_fusion_rule.analysis_fusion_rule()
            self.fusion_info = merge_close_and_open_fusion_rule(open_fusion_rule, close_fusion_rule)
        elif self.fusion_json_file_path != "":
            self.fusion_info = FusionRuleParser(self.fusion_json_file_path)
            self.fusion_info.analysis_fusion_rule()
        elif self.quant_fusion_rule_file_path != "":
            self.fusion_info = FusionRuleParser(self.quant_fusion_rule_file_path)
            self.fusion_info.analysis_fusion_rule()
        else:
            self._make_npu_vs_npu_fusion_rule(compare_data)

    def check_arguments_valid(self: any) -> None:
        """
        Check arguments valid, if invalid, throw exception
        """
        if self.fusion_json_file_path != "":
            ret = utils.check_path_valid(self.fusion_json_file_path, True, False, utils.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
        if self.quant_fusion_rule_file_path != "":
            ret = utils.check_path_valid(self.quant_fusion_rule_file_path, True, False, utils.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
        if self.close_fusion_rule_file_path != "":
            ret = utils.check_path_valid(self.close_fusion_rule_file_path, True, False, utils.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)

    def _make_npu_vs_npu_fusion_rule(self: any, compare_data: CompareData) -> None:
        """
        make fusion rule by npu vs npu
        """
        op_name_to_op_map = {}
        # sort my output dump file by timestamp
        my_output_sort_list = self._sort_file_by_timestamp(compare_data.left_dump_info.op_name_to_file_map)
        self._make_my_output_map(my_output_sort_list, op_name_to_op_map)

        # sort ground truth dump file by timestamp
        ground_truth_sort_list = self._sort_file_by_timestamp(compare_data.right_dump_info.op_name_to_file_map)
        self._make_ground_truth_map(ground_truth_sort_list, op_name_to_op_map)

        self.fusion_info = FusionRuleParser('')
        self.fusion_info.fusion_op_name_to_op_map = op_name_to_op_map
