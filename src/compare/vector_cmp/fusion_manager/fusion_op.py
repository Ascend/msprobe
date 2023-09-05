# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
FusionOp class. This class mainly involves the fusion op info.
"""
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.reg_manager import RegManager
from cmp_utils.constant.compare_error import CompareError


class OutputDesc:
    """
    The class for fusion op output desc
    """

    def __init__(self: any, origin_name: str, origin_output_index: int, origin_format: str,
                 origin_shape: list) -> None:
        self.origin_name = origin_name
        self.origin_output_index = origin_output_index
        self.origin_format = origin_format
        self.origin_shape = origin_shape
        self.data_type = ""

    def get_origin_name(self: any) -> str:
        """
        Get origin name
        """
        return self.origin_name

    def get_origin_shape(self: any) -> list:
        """
        Get origin shape
        """
        return self.origin_shape

    def set_data_type(self: any, data_type: str) -> None:
        """
        Set the data type of output
        """
        self.data_type = data_type


class OpAttr:
    """
    The class for op attr
    """

    def __init__(self: any, original_op_names: list, l1_fusion_no: str, is_multi_op: bool, op_sequence: int) -> None:
        self.original_op_names = original_op_names
        self.l1_fusion_no = l1_fusion_no
        self._is_multi_op = is_multi_op
        self._op_sequence = op_sequence
        self.quant_filter = False

    def is_multi_op(self: any) -> bool:
        """
        is multi op
        """
        return self._is_multi_op

    def get_op_sequence(self: any) -> int:
        """
        Get op sequence
        """
        return self._op_sequence


class Tensor:
    """
    The class for tensor info
    """

    def __init__(self: any, name: str, index: int, tensor_format: str, shape: list) -> None:
        self.name = name
        self.index = index
        self.tensor_format = tensor_format
        self.shape = shape
        self.path = ''
        self.data = None

    def set_data(self: any, data: any) -> None:
        """
        Set data
        """
        self.data = data

    def set_path(self: any, path: str) -> None:
        """
        Set path
        """
        self.path = path


class FusionOp:
    """
    The class for fusion op
    """

    def __init__(self: any, *args: any) -> None:
        op_id, op_name, input_list, op_type, output_desc, attr = args
        self.op_id = op_id
        self.op_name = op_name
        self.input_list = input_list
        self.op_type = op_type
        self.output_desc = output_desc
        self.attr = attr

    def get_origin_tensor(self: any, index: int) -> Tensor:
        """
        Get the origin tensor
        :param index: the index
        :return: the Tensor
        """
        if len(self.output_desc) == 0:
            message = 'The output description of the fusion operator "%s" is empty.' % self.op_name
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR, message)
        real_index = index
        if real_index >= len(self.output_desc):
            real_index = 0
        origin_name = self.output_desc[real_index].origin_name
        if origin_name == "":
            message = 'The origin name of the %s:%d is empty.' % (self.op_name, index)
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR, message)
        origin_output_index = self.output_desc[real_index].origin_output_index
        if origin_output_index is None:
            origin_output_index = index
        return Tensor(origin_name, origin_output_index, self.output_desc[real_index].origin_format,
                      self.output_desc[real_index].origin_shape)

    def get_input_tensor(self: any, index: int) -> Tensor:
        """
        Get the input tensor
        :param index: the input index
        :return: the Tensor
        """
        if index >= len(self.input_list):
            message = 'The input index (%d) is out of range (%d) for the operator list.' % (index, len(self.input_list))
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

        input_op_info = self.input_list[index].split(':')
        if len(input_op_info) != ConstManager.INPUT_INFO_COUNT:
            message = 'The input information (%s) is invalid. It only supports op_name:index.' % self.input_list[index]
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR, message)

        input_op_name = input_op_info[ConstManager.INPUT_OP_NAME_INDEX].strip()
        if not input_op_name:
            message = 'The input information (%s) is invalid. The input operator name is empty.' \
                      % self.input_list[index]
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR, message)

        input_op_index = input_op_info[ConstManager.INPUT_INDEX_INDEX].strip()
        if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, input_op_index):
            message = 'The input information (%s) is invalid. The input index only supports numbers.' \
                      % self.input_list[index]
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR, message)

        return Tensor(input_op_name, int(input_op_index), '', [])

    def is_inner_node(self: any) -> bool:
        """
        Check the fusion op is inner node.
        :return True, if the node is inner node
        """
        inner_node = False
        if self.attr.is_multi_op():
            inner_node = True
            for output_desc in self.output_desc:
                if output_desc.origin_name != '':
                    inner_node = False
                    break
        return inner_node

    def get_real_op_type(self: any) -> str:
        """
        if op_type is not Left or Right, there is real op type
        """
        if self.op_type not in [ConstManager.LEFT_TYPE, ConstManager.RIGHT_TYPE]:
            op_type = self.op_type
        else:
            op_type = ConstManager.NAN
        return op_type
