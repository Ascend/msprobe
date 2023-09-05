
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
"""
Function:
RangeMode class.
This class mainly involves functions for selecting operators.
"""

from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.reg_manager import RegManager
from cmp_utils import log
from vector_cmp.fusion_manager.compare_rule import CompareRule
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.range_manager import range_manager


class RangeMode(range_manager.RangeManager):
    """
    The subclass of range manager
    """

    def __init__(self: any, input_str: str) -> None:
        super(range_manager.RangeManager, self).__init__()
        self.start, self.end, self.step = self._parse_input_str(input_str)

    @staticmethod
    def _parse_input_str(input_str: str) -> (int, int, int):
        cur_range = [ConstManager.DEFAULT_START, ConstManager.DEFAULT_END, ConstManager.DEFAULT_STEP]
        range_list = input_str.split(',')
        if len(range_list) != len(cur_range):
            log.print_error_log('The range (%s) is invalid, just supports "start,end,step".' % input_str)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        for index, item in enumerate(range_list):
            value = item.strip()
            if not value:
                continue
            if index == ConstManager.END_INDEX and value == '-1':
                continue
            if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, value):
                log.print_error_log('The range (%s) is invalid, just supports '
                                    '"start,end,step", the value is number.' % input_str)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            cur_range[index] = int(value)
        return (x for x in cur_range)

    def get_all_ops(self: any, compare_rule: CompareRule) -> list:
        """
        Get all operators according to [start, end, step]:
        :param compare_rule: the compare rule
        :return: operator list
        """
        op_list = []
        for op in compare_rule.fusion_info.op_list:
            op_sequence = op.attr.get_op_sequence()
            if op_sequence > self.end:
                break
            if op_sequence < self.start:
                continue
            if self.start == op_sequence or self.end == op_sequence or ((op_sequence - self.start) % self.step) == 0:
                op_list.append(op)
        return self._get_op_list(op_list, compare_rule)

    def check_input_valid(self: any, op_count: int) -> None:
        """
        Check range valid:
        :param op_count: the op count
        """
        if self.start < 1 or self.start > op_count:
            log.print_out_of_range_error('', 'start', self.start, '[1, %d]' % op_count)
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
        if self.end == -1:
            self.end = op_count
        if self.end < self.start or self.end > op_count:
            log.print_out_of_range_error('', 'end', self.end, '[1, %d]' % op_count)
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
        if self.step < 1 or self.step >= op_count:
            log.print_out_of_range_error('', 'step', self.step, '[1, %d)' % op_count)
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
        log.print_info_log('The range compare for [%d,%d,%d].' % (self.start, self.end, self.step))
