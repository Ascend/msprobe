
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
"""
Function:
SelectMode class.
This class mainly involves functions for selecting operators.
"""

from cmp_utils.reg_manager import RegManager
from cmp_utils import log
from vector_cmp.fusion_manager.compare_rule import CompareRule
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.range_manager import range_manager


class SelectMode(range_manager.RangeManager):
    """
    The subclass of range manager
    """

    def __init__(self: any, input_str: str) -> None:
        super(range_manager.RangeManager, self).__init__()
        self.selected_op = self._parse_input_str(input_str)

    @staticmethod
    def _parse_input_str(input_str: str) -> list:
        input_operators = []
        index_list = input_str.split(',')
        for item in index_list:
            value = item.strip()
            if not value:
                continue
            if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, value):
                log.print_error_log('The index (%s) is invalid, just supports '
                                    '"index_1, index_2, ...", the value is zero or a positive number.' % input_str)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            input_operators.append(int(value))
        selected_operators = list(set(input_operators))
        index_list_sorted = sorted(selected_operators)
        return index_list_sorted

    def get_all_ops(self: any, compare_rule: CompareRule) -> list:
        """
        Get all operators according to indexes:
        :param compare_rule: the compare rule
        :return: operator list
        """
        op_list = [op for op in compare_rule.fusion_info.op_list if op.attr.get_op_sequence() in self.selected_op]
        return self._get_op_list(op_list, compare_rule)

    def check_input_valid(self: any, op_count: int) -> None:
        """
        Check range valid:
        :param op_count: the op count
        """
        op_list = []
        out_of_range_list = []
        for value in self.selected_op:
            if value <= op_count:
                op_list.append(value)
            else:
                out_of_range_list.append(value)
        if out_of_range_list:
            invalid_ops = str(out_of_range_list)
            message = 'Index %s out of range ' % invalid_ops + '[1,%d].' % op_count
            if not op_list:
                log.print_error_log(message)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            log.print_warn_log(message)
        self.selected_op = op_list
        log.print_info_log('Operator list %s is valid.' % (str(self.selected_op)))
