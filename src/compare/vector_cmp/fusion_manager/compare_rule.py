# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
VectorComparison class. This class mainly involves the compare function.
"""

import os
from cmp_utils import utils, utils_type, path_check
from vector_cmp.fusion_manager.fusion_rule_parser import FusionRuleParser
from vector_cmp.fusion_manager.fusion_rule_parser import merge_fusion_rule
from vector_cmp.fusion_manager.fusion_rule_parser import merge_close_and_open_fusion_rule
from vector_cmp.fusion_manager.fusion_op import FusionOp
from vector_cmp.fusion_manager.fusion_op import OpAttr
from cmp_utils.constant.const_manager import ConstManager
from dump_parse.dump import CompareData, DumpInfo
from cmp_utils.constant.compare_error import CompareError
from dump_parse import dump_utils


class CompareRule:
    """
    The class for compare rule
    """
    def __init__(self: any, fusion_json_file_path: str, quant_fusion_rule_file_path: str,
                 close_fusion_rule_file_path: str = '') -> None:
        self.fusion_json_file_path = self._get_real_path_with_default(fusion_json_file_path)
        self.quant_fusion_rule_file_path = self._get_real_path_with_default(quant_fusion_rule_file_path)
        self.close_fusion_rule_file_path = self._get_real_path_with_default(close_fusion_rule_file_path)
        self.fusion_info = None

    @staticmethod
    def _sort_file_by_timestamp(dump_info: DumpInfo) -> dict:
        op_name_to_file_map = dump_info.op_name_to_file_map
        op_name_to_task_mode_map = dump_info.op_name_to_task_mode_map
        dump_utils.SortMode.hash_to_file_name_map = dump_info.hash_to_file_name_map
        origin_dic = {}
        for op_name, dump_file_list in op_name_to_file_map.items():
            dump_task_mode = op_name_to_task_mode_map.get(op_name)
            if len(dump_file_list) > 1:
                dump_file_list = dump_utils.sort_dump_file_list(dump_task_mode, dump_file_list)
            if dump_task_mode == ConstManager.NORMAL_MODE:
                timestamp = dump_utils.get_normal_timestamp(dump_file_list[-1])
            else:
                timestamp = dump_utils.get_ffts_timestamp(dump_file_list[-1])
            origin_dic[(op_name, timestamp)] = dump_file_list
        return dict(sorted(origin_dic.items(), key=lambda s: s[0][1]))

    @staticmethod
    def _make_my_output_map(my_output_sort_map: dict, op_name_to_op_map: dict) -> None:
        attr = OpAttr([], '', False, 0)
        # make my output fusion op by my_output_sort_list
        for key, values in my_output_sort_map.items():
            op_name = key[0]
            if op_name not in op_name_to_op_map:
                fusion_op = FusionOp(len(op_name_to_op_map), op_name, [], ConstManager.LEFT_TYPE, values, attr)
                op_name_to_op_map[op_name] = [fusion_op]
            else:
                op_name_to_op_map[op_name][0].output_desc.extend(values)

    @staticmethod
    def _make_ground_truth_map(ground_truth_sort_map: dict, op_name_to_op_map: dict) -> None:
        attr = OpAttr([], '', False, 0)
        # make ground truth fusion op by ground_truth_sort_list
        for key, values in ground_truth_sort_map.items():
            op_name = key[0]
            if op_name not in op_name_to_op_map:
                fusion_op = FusionOp(len(op_name_to_op_map), op_name, [], ConstManager.RIGHT_TYPE, values, attr)
                op_name_to_op_map[op_name] = [fusion_op]
            else:
                fusion_op_info = op_name_to_op_map[op_name][-1]
                if fusion_op_info.op_type == ConstManager.RIGHT_TYPE:
                    op_name_to_op_map[op_name][-1].output_desc.extend(values)
                else:
                    op_id = op_name_to_op_map[op_name][0].op_id
                    fusion_op = FusionOp(op_id, op_name, [], ConstManager.RIGHT_TYPE, values, attr)
                    op_name_to_op_map[op_name].append(fusion_op)

    @staticmethod
    def _get_real_path_with_default(file_path: str) -> str:
        return os.path.realpath(file_path) if file_path else ""

    def parse_fusion_rule(self: any, compare_data: CompareData) -> None:
        """
        Parse fusion rule
        :param compare_data: compare data
        """
        if self.fusion_json_file_path != "":
            if self.quant_fusion_rule_file_path != "":
                offline_fusion_rule = FusionRuleParser(self.fusion_json_file_path)
                offline_fusion_rule.analysis_fusion_rule()
                quant_fusion_rule = FusionRuleParser(self.quant_fusion_rule_file_path)
                quant_fusion_rule.analysis_fusion_rule()
                self.fusion_info = merge_fusion_rule(offline_fusion_rule, quant_fusion_rule)
            elif self.close_fusion_rule_file_path != "":
                open_fusion_rule = FusionRuleParser(self.fusion_json_file_path)
                open_fusion_rule.analysis_fusion_rule()
                close_fusion_rule = FusionRuleParser(self.close_fusion_rule_file_path)
                close_fusion_rule.analysis_fusion_rule()
                self.fusion_info = merge_close_and_open_fusion_rule(open_fusion_rule, close_fusion_rule)
            else:
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
        if self.fusion_json_file_path:
            ret = path_check.check_path_valid(self.fusion_json_file_path, True, False, path_check.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
        if self.quant_fusion_rule_file_path:
            ret = path_check.check_path_valid(self.quant_fusion_rule_file_path, True, False, path_check.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)
        if self.close_fusion_rule_file_path:
            ret = path_check.check_path_valid(self.close_fusion_rule_file_path, True, False, path_check.PathType.File)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                raise CompareError(ret)

    def _make_npu_vs_npu_fusion_rule(self: any, compare_data: CompareData) -> None:
        """
        make fusion rule by npu vs npu
        """
        op_name_to_op_map = {}
        # sort my output dump file by timestamp
        my_output_sort_map = self._sort_file_by_timestamp(compare_data.left_dump_info)
        self._make_my_output_map(my_output_sort_map, op_name_to_op_map)

        # sort ground truth dump file by timestamp
        ground_truth_sort_map = self._sort_file_by_timestamp(compare_data.right_dump_info)
        self._make_ground_truth_map(ground_truth_sort_map, op_name_to_op_map)

        self.fusion_info = FusionRuleParser('')
        self.fusion_info.fusion_op_name_to_op_map = op_name_to_op_map
