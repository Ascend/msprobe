
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
RangeManager class.
This class mainly involves the get function.
"""

import sys
from abc import ABC, abstractmethod

from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.fusion_manager.compare_rule import CompareRule


class RangeManager(ABC):
    """
    The class for range manager
    """

    def __init__(self: any, input_str: str) -> None:
        self.input_str = input_str

    @staticmethod
    def adjust_header(header: list) -> None:
        """
        adjust header by range
        :param header: the header
        """
        if RangeManager._has_cmd():
            header.insert(ConstManager.OP_SEQUENCE_INDEX, ConstManager.OP_SEQUENCE)

    @staticmethod
    def adjust_data(data: list, op_sequence: int) -> None:
        """
        adjust data by range
        :param data: the data
        :param op_sequence: the op_sequence
        """
        if RangeManager._has_cmd():
            data.insert(ConstManager.OP_SEQUENCE_INDEX, str(op_sequence))

    @staticmethod
    def _get_op_list(op_list: list, compare_rule: CompareRule) -> list:
        range_op_name = []
        for op in op_list:
            fusion_op_name = compare_rule.fusion_info.op_name_to_fusion_op_name_map.get(op.op_name)
            if fusion_op_name not in range_op_name:
                range_op_name.append(fusion_op_name)
        return range_op_name

    @staticmethod
    def _has_cmd() -> bool:
        """
        Check the argument has range
        :return: bool
        """
        # excluding compare_vector.py commands of old versions
        support_cmd_list = ['-r', '-s', '--range', '--select']
        match = False
        for item in sys.argv:
            if item in support_cmd_list:
                match = True
                break
        return match and 'compare_vector.py' not in sys.argv[0]

    @staticmethod
    def _parse_input_str(input_str: str):
        pass

    @abstractmethod
    def get_all_ops(self: any, compare_rule: CompareRule):
        """
        Get all operators:
        :param compare_rule: the compare rule
        """
        pass

    @abstractmethod
    def check_input_valid(self: any, op_count: int):
        """
        Check range valid:
        :param op_count: the op count
        """
        pass
