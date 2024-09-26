
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.

import collections
from functools import reduce
import json

import numpy as np
from dump_data_pb2 import DumpData

from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils import log
from cmp_utils import common


CommonAttr = collections.namedtuple('CommonAttr', ['data_type', 'tensor_format', 'address', 'original_shape'])


def _deserialize_dump_data_to_array(data, data_type, shape: list = None) -> any:
    """
    Deserialize dump data to array
    :param tensor: the dump data for input or output
    :return: the numpy array
    """
    if shape is not None and 0 in shape:
        return np.array([]).reshape(shape)

    if shape is not None:  # shape can be empty [] for scalar data
        cnt = 1
        for ii in shape:
            cnt *= ii
        cur_type = common.get_dtype_by_data_type(data_type)
        cur_byte = np.dtype(cur_type).itemsize
        result = np.frombuffer(data[:cur_byte * cnt], dtype=cur_type)
    else:
        result = np.frombuffer(data, dtype=common.get_dtype_by_data_type(data_type))
    if data_type in ConstManager.UNPACK_DTYPE:
        return np.unpackbits(result)
    elif data_type in ConstManager.CAST_FP32_DTYPE:
        return result.astype('float32')
    else:
        return result


def build_dump_tensor(dump_data_object_data: list, is_input: bool, is_ffts: bool) -> None:
    """
    replace the input or output object of DumpDataObj to DumpyTensor
    @param dump_data_object_data: input or output object of DumpDataObj
    @param is_input: if the tensor is input data
    @param is_ffts: if the tensor is ffts plus mode
    @return: None
    """
    for index, tensor in enumerate(dump_data_object_data):
        if not tensor.HasField('shape') and tensor.size:
            log.print_info_log(f"Tensor shape is empty, using size {tensor.size} as shape")
            tensor.shape.dim.append(tensor.size) # Ignore dtype size, just set a value large enough
        data_to_np = _deserialize_dump_data_to_array(tensor.data, tensor.data_type, list(tensor.shape.dim))
        dump_tensor = DumpTensor(index, tensor.data_type, tensor.format, list(tensor.shape.dim),
                                 data_to_np, tensor.size, list(tensor.original_shape.dim),
                                 tensor.address, tensor.sub_format, is_input, is_ffts)
        dump_data_object_data[index] = dump_tensor


def build_nano_dump_tensor(dump_data_object_data: list, is_input: bool) -> None:
    """
    replace the input or output object of DumpDataObj to DumpyTensor
    @param dump_data_object_data: input or output object of DumpDataObj
    @param is_input: if the tensor is input data
    @param is_ffts: if the tensor is ffts plus mode
    @return: None
    """
    for index, tensor in enumerate(dump_data_object_data):
        data_to_np = _deserialize_dump_data_to_array(tensor.data, tensor.data_type, tensor.shape_dims)
        dump_tensor = DumpTensor(index, tensor.data_type, tensor.format, tensor.shape_dims,
                                 data_to_np, tensor.size, tensor.original_shape_dims,
                                 tensor.address, is_input=is_input, is_ffts=False)
        dump_data_object_data[index] = dump_tensor


class DumpTensor:
    """
    The class of DumpTensor, replace the class of DD.DumpData.input or output.
    Include the data detail: index, data_type, tensor_format, shape, data, size, original_shape
    """

    def __init__(self: any, index: int = None, data_type: int = None, tensor_format: int = None,
                 shape: list = None, data: np.ndarray = None, size: int = None, original_shape: list = None,
                 address: int = None, sub_format: int = 0, is_input: bool = False, is_ffts: bool = False) -> None:

        self.index = index
        self.data_type = data_type
        self.tensor_format = tensor_format
        self.shape = shape if shape else []
        self.data = data
        self.size = size
        self.original_shape = original_shape
        self.address = address
        self.sub_format = sub_format
        self.is_input = is_input
        self.is_ffts = is_ffts

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
    The class of DumpDataObject, replace the class DD.DumpData or NanoDumpData.
    Include dump_file information
    """
    def __init__(self: any, dump_data: DumpData = DumpData(), nano_dump_data=None) -> None:
        if nano_dump_data:
            self.version = nano_dump_data.version_id
            self.op_name = nano_dump_data.op_name
            self.dump_time = nano_dump_data.dump_time
            self.buffer = None
            self.space = None
            self.attr = None
            self.input_data = nano_dump_data.inputs
            self.output_data = nano_dump_data.outputs
            self.ffts_file_check = False

            build_nano_dump_tensor(self.output_data, is_input=False)
            build_nano_dump_tensor(self.input_data, is_input=True)

        else:
            self.version = dump_data.version
            self.op_name = dump_data.op_name
            self.dump_time = dump_data.dump_time
            self.buffer = dump_data.buffer
            self.space = [_space_data for _space_data in dump_data.space]
            self.attr = json.loads(dump_data.attr[0].value) if dump_data.attr else None
            self.input_data = [_input_data for _input_data in dump_data.input]
            self.output_data = [_output_data for _output_data in dump_data.output]

            is_ffts = False if self.get_ffts_mode is None else True
            build_dump_tensor(self.output_data, is_input=False, is_ffts=is_ffts)
            build_dump_tensor(self.input_data, is_input=True, is_ffts=is_ffts)

            self.ffts_file_check = True
            self.ffts_auto_input_shape_list = []
            self.ffts_auto_output_shape_list = []

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
        return self.attr.get("slice_instance_num")

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

    def get_auto_output_data(self: any, output_shape: list) -> list:
        """
        Get output data
        @return: list of output data
        """
        output_data_list = []
        for index, output in enumerate(self.output_data):
            if self.check_shape_match(output.data, output_shape[index]):
                output_data_list.append(output.data.reshape(output_shape[index]))
        return output_data_list
    
    def calculate_auto_mode_shape(self: any, thread_id: int, tensor_type: str) -> list:
        """
        calculate the output data shape of auto mode
        @return: output shape
        """
        output_shape = []
        if tensor_type == ConstManager.INPUT:
            attr_name = "input_tensor_slice"
        elif tensor_type == ConstManager.OUTPUT:
            attr_name = "output_tensor_slice"
        if self.attr is not None and self.attr[attr_name]:
            for output in self.attr[attr_name][thread_id]:
                output_index = []
                for addr in output:
                    dim = addr.get("higher") - addr.get("lower")
                    output_index.append(dim)
                output_shape.append(output_index)
        return output_shape

    def set_op_attr(self: any, op_name: str, ffts_file_check: bool) -> None:
        """
        set op_name and ffts_file_check
        @param op_name: op name
        @param ffts_file_check: if file num doesn't match thread num
        @return: none
        """
        self.op_name = op_name
        self.ffts_file_check = ffts_file_check


