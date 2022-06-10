#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves xxxx function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import log
import utils
import numpy as np

import dump_data_pb2 as DD
from const_manager import ConstManager


class OverflowDetection:
    """
    The class for op overflow detection info
    """

    def __init__(self: any, *args: any) -> None:
        compare_data, op_name = args
        self.dump_version = compare_data.dump_version
        self.left_dump_info = compare_data.left_dump_info
        self.overflow_tensor_list = []
        self.op_name = op_name

    @staticmethod
    def _judge_overflow_data_by_array(tensor_data: any) -> str:
        if tensor_data is not None:
            absolute_value_data = np.absolute(tensor_data)
            up_overflow_flag = (absolute_value_data.max() >= ConstManager.OVERFLOW_MAX_VALUE)
            if up_overflow_flag:
                return 'YES'
            return 'NO'
        return ConstManager.NAN

    @staticmethod
    def process_model_overflow_detection(op_name: str, index: int, is_input: bool, tensor: any) -> str:
        """
        process model overflow detection
        """
        if tensor and tensor.data_type == DD.DT_FLOAT16:
            tensor_data_array = utils.deserialize_dump_data_to_array(tensor)
            overflow_result = OverflowDetection._judge_overflow_data_by_array(tensor_data_array)
            if overflow_result == 'YES':
                log.print_warn_log(
                    "{} operator {}:{} is overflow.".format(op_name, 'input' if is_input else 'output', index))
            return overflow_result
        return ConstManager.NAN

    def process_op_overflow_detection(self: any) -> None:
        """
        process op overflow detection
        """
        log.print_info_log("Checking {} operator overflow".format(self.op_name))
        input_tensor_data_info, output_tensor_data_info = self.parse_dump_file()
        if len(input_tensor_data_info) == 0 and len(output_tensor_data_info) == 0:
            return
        self._check_overflow_tensor(input_tensor_data_info, output_tensor_data_info)
        self._print_overflow_info_to_console(self.overflow_tensor_list)

    def parse_dump_file(self: any) -> (list, list):
        """
        process op overflow detection
        """
        dump_file_path = self.left_dump_info.get_op_dump_file(self.op_name)
        if not dump_file_path:
            return [], []
        dump_data = utils.parse_dump_file(dump_file_path, self.dump_version)
        input_tensor_data_info = self._get_tensor_data_info("input", dump_data.input, dump_file_path)
        output_tensor_data_info = self._get_tensor_data_info("output", dump_data.output, dump_file_path)
        return input_tensor_data_info, output_tensor_data_info

    def _get_tensor_data_info(self: any, tensor_type: str, tensor_list: list, dump_file_path: str) -> list:
        tensor_data_info = []
        for (index, tensor) in enumerate(tensor_list):
            if tensor.data_type == DD.DT_FLOAT16:
                log.print_info_log('Start to parse the data of %s:%d in "%s".' % (tensor_type, index, dump_file_path))
                array = utils.deserialize_dump_data_to_array(tensor)
                tensor_data_info.append(
                    {"tensor_type": tensor_type, "index": str(index), "tensor_data": array, "tensor_info": tensor,
                     "dump_file_path": dump_file_path})
        if len(tensor_data_info) == 0:
            log.print_warn_log("The {} data type of {} operator is not float16. "
                               "The overflow check supports only float16.".format(tensor_type, self.op_name))
        return tensor_data_info

    def _check_overflow_tensor(self: any, input_data: list, output_data: list) -> None:
        self._judge_overflow_data(input_data, "input")
        self._judge_overflow_data(output_data, "output")

    def _judge_overflow_data(self: any, tensor_list: list, tensor_type: str) -> None:
        if len(tensor_list) == 0:
            log.print_warn_log('There is no %s in "%s".' % (
                tensor_type, self.left_dump_info.get_op_dump_file(self.op_name)))
            return
        for item in tensor_list:
            tensor_data = item.get("tensor_data")
            absolute_value_data = np.absolute(tensor_data)
            up_overflow_flag = (absolute_value_data.max() >= ConstManager.OVERFLOW_MAX_VALUE)
            if up_overflow_flag:
                self.overflow_tensor_list.append(item)

    def _print_overflow_info_to_console(self: any, overflow_tensor_list: list) -> None:
        if len(overflow_tensor_list) == 0:
            log.print_info_log("The input and output of the op: {} is not overflow.".format(self.op_name))
        else:
            tensor_index_len = []
            tensor_index_info = []
            for item in overflow_tensor_list:
                tensor_index = "".join([item.get("tensor_type"), ":", item.get("index")])
                tensor_index_info.append(tensor_index)
                tensor_index_len.append(len(tensor_index))
            spacing_distance = max(tensor_index_len)
            if spacing_distance < 20:
                spacing_distance = 20
            format_line = "".join(['{:<', str(spacing_distance), '}', '{:<', str(spacing_distance), '}'])
            print(format_line.format("TensorIndex", "Overflow"))
            for item in tensor_index_info:
                print(format_line.format(item, "yes"))
            log.print_info_log("%s operator has overflowed." % self.op_name)
