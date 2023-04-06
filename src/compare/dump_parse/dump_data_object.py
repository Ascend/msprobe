#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.

import collections
from functools import reduce
import json

import numpy as np
from dump_data_pb2 import DumpData

from src.compare.cmp_utils.constant.compare_error import CompareError
from src.compare.cmp_utils import log


CommonAttr = collections.namedtuple('CommonAttr', ['data_type', 'tensor_format', 'address', 'original_shape'])

class DumpTensor:
    """
    The class of DumpTensor, replace the class of DD.DumpData.input or output.
    Include the data detail: index, data_type, tensor_format, shape, data, size, original_shape
    """

    def __init__(self: any, index: int = None, data_type: int = None, tensor_format: int = None,
                 shape: list = None, data: np.ndarray = None, size: int = None, original_shape: list = None,
                 address: int = None, sub_format: int = 0) -> None:

        self.index = index
        self.data_type = data_type
        self.tensor_format = tensor_format
        self.shape = shape if shape else []
        self.data = data
        self.size = size
        self.original_shape = original_shape
        self.address = address
        self.sub_format = sub_format

    @property
    def get_common_attr(self: any) -> tuple:
        """
        get common attr
        @return: tuple of common attr
        """
        common_attr = CommonAttr(self.data_type, self.tensor_format, self.address, self.original_shape)
        return common_attr


class DumpDataObj:
    """
    The class of DumpDataObject, replace the class DD.DumpData.
    Include dump_file information
    """
    def __init__(self: any, dump_data: DumpData = DumpData()) -> None:
        self.version = dump_data.version
        self.op_name = dump_data.op_name
        self.dump_time = dump_data.dump_time
        self.buffer = dump_data.buffer
        self.space = [_space_data for _space_data in dump_data.space]
        self.attr = json.loads(dump_data.attr[0].value) if dump_data.attr else None
        self.input_data = [_input_data for _input_data in dump_data.input]
        self.output_data = [_output_data for _output_data in dump_data.output]
        self.ffts_file_check = True

    @property
    def get_output_data(self: any) -> list:
        """
        Get output data
        @return: list of output data
        """
        output_data_list = []
        for output in self.output_data:
            if self.check_shape_match(output.data, output.shape):
                output_data_list.append(output.data.reshape(output.shape))
        return output_data_list

    @property
    def get_dump_time(self: any) -> int:
        """
        get dump_time
        @return: dump time
        """
        return self.dump_time

    @property
    def get_thread_num(self: any) -> int:
        """
        get slice_instance_num
        @return: slice number
        """
        return self.attr["slice_instance_num"] if self.attr else None

    @property
    def get_cut_axis_manual(self: any) -> list:
        """
        calculate the cut axis of manual mode
        @return: cut axis
        """
        cut_axis = []
        if not self.attr or not self.attr["outputCutList"]:
            return cut_axis
        for output in self.attr["outputCutList"]:
            output_index = []
            for index, value in enumerate(output):
                if value != 1:
                    output_index.append(index)
            cut_axis.append(output_index)
        return cut_axis

    @property
    def get_cut_axis_auto(self: any) -> list:
        """
        calculate the cut axis of auto mode
        @return: cut axis
        """
        output_shape = self.calculate_auto_mode_shape
        cut_axis = []
        return cut_axis

    @property
    def calculate_auto_mode_shape(self: any) -> list:
        """
        calculate the output data shape of auto mode
        @return: output shape
        """
        output_shape = []
        if self.attr["output_tensor_slice"]:
            for output in self.attr["output_tensor_slice"][0]:
                output_index = []
                for addr in output:
                    dim = addr.get("higher") - addr.get("lower")
                    output_index.append(dim)
                output_shape.append(output_index)
        return output_shape

    @property
    def get_ffts_mode(self: any) -> any:
        """
        get ffts+ mode
        @return: mode
        """
        return self.attr["threadMode"] if self.attr else None

    @staticmethod
    def check_shape_match(output_data: np.ndarray, shape: list) -> bool:
        if output_data.shape[-1] == reduce(lambda x, y: x * y, shape):
            return True
        else:
            log.print_error_log(
                f"The output_data shape {output_data.shape[-1]} doesn't match the shape in dump file {shape}")
            raise CompareError(CompareError.MSACCUCMP_UNMATCH_DATA_SHAPE_ERROR)

    def set_op_attr(self: any, op_name: str, ffts_file_check: bool) -> None:
        """
        set op_name and ffts_file_check
        @param op_name: op name
        @param ffts_file_check: if file num doesn't match thread num
        @return: none
        """
        self.op_name = op_name
        self.ffts_file_check = ffts_file_check


