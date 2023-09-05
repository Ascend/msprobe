# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the common function.
"""
import dump_data_pb2 as DD

from cmp_utils import common
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.reg_manager import RegManager
from cmp_utils.path_check import check_name_valid
from vector_cmp.fusion_manager.fusion_op import FusionOp
from vector_cmp.fusion_manager.fusion_rule_parser import FusionRuleParser
from vector_cmp.fusion_manager import fusion_rule_parser


class TensorId:
    """
    The class for tensor id
    """

    def __init__(self: any, op_name: str, tensor_type: str, index: str) -> None:
        self.op_name = op_name
        self.tensor_type = tensor_type
        if not RegManager.match_pattern(RegManager.NUMBER_PATTERN, index):
            log.print_only_support_error('detail index', index, 'natural number')
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        self.index = int(index)

    def check_arguments_valid(self: any) -> None:
        """
        check arguments valid, if invalid, throw exception
        """
        ret = check_name_valid(self.op_name)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        if self.tensor_type not in ConstManager.SUPPORT_DETAIL_TYPE:
            log.print_only_support_error('detail type', self.tensor_type,
                                         ConstManager.SUPPORT_DETAIL_TYPE)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def get_tensor_id(self: any) -> str:
        """
        Get tensor id
        """
        return "%s:%s:%d" % (self.op_name, self.tensor_type, self.index)

    def is_input(self: any) -> bool:
        """
        is input
        :return: true if input
        """
        return self.tensor_type == ConstManager.INPUT

    def get_file_prefix(self: any) -> str:
        """
        Get detail file name prefix
        :return str
        """
        return "%s_%s_%d" % (self.op_name.replace('/', '_'), self.tensor_type, self.index)

    def get_tensor_type_index(self: any) -> str:
        """
        Get detail file name prefix, like _input_0
        :return str
        """
        return "_%s_%d" % (self.tensor_type, self.index)


class DetailInfo:
    """
    The class for detail info
    """

    def __init__(self: any, tensor_id: TensorId, top_n: int, ignore_result: bool, max_line: int) -> None:
        self.tensor_id = tensor_id
        self.my_output_ops = ''
        self.ground_truth_ops = ''
        self.detail_format = ''
        self.top_n = top_n
        self.ignore_result = ignore_result
        self.max_line = max_line

    def check_arguments_valid(self: any) -> None:
        """
        check arguments valid, if invalid, throw exception
        """
        self.tensor_id.check_arguments_valid()
        const_manager = ConstManager()
        if self.top_n < const_manager.min_top_n or self.top_n > const_manager.max_top_n:
            log.print_out_of_range_error('', 'top n', self.top_n,
                                         '[%d, %d]' % (const_manager.min_top_n, const_manager.max_top_n))
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def get_detail_op(self: any, fusion_rule: FusionRuleParser) -> (FusionOp, list):
        """
        Get detail op by fusion rule
        :param: fusion_rule: the fusion rule
        :return: the fusion op, the fusion op list
        """
        fusion_op_list, fusion_op = fusion_rule.get_fusion_op_list(self.tensor_id.op_name)
        # get the map for {original_op_names, op_list}
        right_to_left_map = fusion_rule_parser.make_right_to_left_multi_map(
            fusion_op_list)
        my_output_ops_str, ground_truth_ops_str = fusion_rule_parser.make_left_and_right_string(right_to_left_map)
        # if right ops is empty, mark '*' indicates that the left op is a
        # new operator, and there is no operator on the right that matches it
        if ground_truth_ops_str == "":
            ground_truth_ops_str = '*'
        self.my_output_ops = my_output_ops_str.replace(',', ' ')
        self.ground_truth_ops = ground_truth_ops_str.replace(',', ' ')
        return fusion_op, fusion_op_list

    def set_detail_format(self: any, shape_str: str, tensor_format: int, ground_truth_format: str) -> None:
        """
        Set detail format by shape
        :param shape_str: the shape string
        :param tensor_format: tensor format
        :param ground_truth_format: ground truth format
        """
        if common.contain_depth_dimension(tensor_format):
            self.detail_format = 'N C D H W'
        else:
            if tensor_format == DD.FORMAT_FRACTAL_Z:
                self.detail_format = ' '.join(ground_truth_format)
            else:
                self.detail_format = "N C H W" if shape_str != "()" else "ID"

    def check_and_set_format(self: any, shape_str: str, tensor_format: int, ground_truth_format: int) -> None:
        """
        Check and Set detail format by shape
        :param shape_str: the shape string
        :param tensor_format: tensor format
        :param ground_truth_format: ground truth format
        """
        if tensor_format != ground_truth_format:
            log.print_error_log("NPUDump tensor format not match Ground truth tensor format!"
                                "Cannot be directly compared.")
            raise CompareError(CompareError.MSACCUCMP_INVALID_FORMAT_ERROR)
        if common.contain_depth_dimension(tensor_format):
            self.detail_format = 'N C D H W'
        else:
            self.detail_format = "N C H W" if shape_str != "()" else "ID"

    def set_detail_ops(self: any, my_out_ops: str, ground_truth_ops: str) -> None:
        """
        Check and Set detail format by shape
        :param my_out_ops: the opname of my out tensor
        :param ground_truth_ops: the opname of ground truth tensor
        """
        self.my_output_ops = my_out_ops.replace(',', ' ')
        self.ground_truth_ops = ground_truth_ops.replace(',', ' ')

    def make_detail_header(self: any) -> str:
        """
        Make detail header
        """
        return "Index,%s,NPUDump,GroundTruth,AbsoluteError,RelativeError\n" % self.detail_format

    def get_detail_info(self: any) -> str:
        """
        Get detail_info
        """
        return "NPUDump:%s\nGroundTruth:%s\nFormat:%s\n" % (self.my_output_ops, self.ground_truth_ops,
                                                            self.detail_format)
