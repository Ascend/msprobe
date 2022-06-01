#!/usr/bin/env python
# coding=utf-8
"""
Function:
NpuVsNpuComparison class. This class mainly involves the compare function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""

import utils
import common
import log
from const_manager import ConstManager
from dump import CompareData
from algorithm_manager import AlgorithmManager
from fusion_op import FusionOp
from fusion_op import Tensor
import compare_result
from compare_error import CompareError
from overflow_detection import OverflowDetection


class NpuVsNpuComparison:
    """
    The class for npu vs npu comparison
    """

    def __init__(self: any, compare_data: CompareData, fusion_op_list: list, algorithm_manager: AlgorithmManager,
                 overflow_detection: bool = False) -> None:
        self.compare_data = compare_data
        self.fusion_op_list = fusion_op_list
        self.algorithm_manager = algorithm_manager
        self.op_name = fusion_op_list[0].op_name
        self.overflow_detection = overflow_detection

    def _make_one_dump_file_result(self: any) -> (int, bool, list):
        error_msg = []
        # if only left or right has dump file, the result is NaN
        if self.fusion_op_list[0].op_type == ConstManager.LEFT_TYPE:
            message = '[%s] There is no the ground truth dump file for the op "%s".' % (self.op_name, self.op_name)
            log.print_warn_log(message)
            error_msg.append(message)
        elif self.fusion_op_list[0].op_type == ConstManager.RIGHT_TYPE:
            message = '[%s] There is no the my output dump file for the op "%s".' % (self.op_name, self.op_name)
            log.print_warn_log(message)
            error_msg.append(message)
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager,
                                                            overflow_detection=self.overflow_detection)
        result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg, no_dump_file=True)
        return CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR, False, result

    def _get_dump_data(self: any, fusion_op: FusionOp, dump_path: str, dump_type: str) -> Tensor:
        """
        get dump data by fusion op output_desc
        """
        dump_file_list = fusion_op.output_desc
        match_count = len(fusion_op.output_desc)
        dump_file_path = dump_file_list[-1]
        if match_count > 1:
            log.print_warn_log(
                'There are %d dump files of the "%s" in the path "%s". Choose the file "%s" to compare.'
                % (match_count, fusion_op.op_name, dump_path, dump_file_path))
        log.print_info_log('[%s] [%s] %s' % (fusion_op.op_name, dump_type, dump_file_path))
        dump_data = utils.parse_dump_file(dump_file_path, self.compare_data.dump_version)
        if dump_data.op_name:
            fusion_op.op_name = dump_data.op_name
        tensor = Tensor(fusion_op.op_name, 0, '', [])
        tensor.set_path(dump_file_path)
        tensor.set_data(dump_data)
        return tensor

    def _check_op_data_valid(self: any, my_output_list: any, ground_truth_list: any, tensor_type: str) -> (int, str):
        """
        check format and shape of each tensor valid
        """
        message = ""
        tensor_id_prefix = "%s:%s" % (self.op_name, tensor_type)
        for index, (my_output_tensor, ground_truth_tensor) in enumerate(zip(my_output_list, ground_truth_list)):
            tensor_id = '%s:%d' % (tensor_id_prefix, index)
            # check format valid
            if my_output_tensor.format != ground_truth_tensor.format:
                message = log.print_not_match_error(
                    self.op_name, 'format',
                    common.get_format_string(my_output_tensor.format),
                    common.get_format_string(ground_truth_tensor.format), tensor_id)
                return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message

            # check the length of shape is the same
            if len(my_output_tensor.shape.dim) != len(ground_truth_tensor.shape.dim):
                message = log.print_not_match_error(
                    self.op_name, 'shape',
                    utils.convert_shape_to_string(my_output_tensor.shape.dim),
                    utils.convert_shape_to_string(ground_truth_tensor.shape.dim), tensor_id)
                return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message
            # check each dim in shape is the same
            for my_output_dim, ground_truth_dim in zip(my_output_tensor.shape.dim, ground_truth_tensor.shape.dim):
                if my_output_dim != ground_truth_dim:
                    message = log.print_not_match_error(
                        self.op_name, 'shape',
                        utils.convert_shape_to_string(my_output_tensor.shape.dim),
                        utils.convert_shape_to_string(ground_truth_tensor.shape.dim), tensor_id)
                    return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message
        return CompareError.MSACCUCMP_NONE_ERROR, message

    def check_tensor_valid(self: any, my_output_tensor_list: any, ground_truth_tensor_list: any,
                           tensor_type: str) -> (int, str):
        """
        check tensor valid
        """
        # check the length is same
        if len(my_output_tensor_list) != len(ground_truth_tensor_list):
            message = log.print_not_match_error(
                self.op_name, 'number of %s' % tensor_type, str(len(my_output_tensor_list)),
                str(len(ground_truth_tensor_list)))
            return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message
        if len(my_output_tensor_list) != 0:
            # check each tensor format and shape valid
            return self._check_op_data_valid(my_output_tensor_list, ground_truth_tensor_list, tensor_type)
        message = '[%s] There is no %s. Skip the %s:%s.' % (self.op_name, tensor_type, self.op_name, tensor_type)
        log.print_info_log(message)
        return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message

    def _compare_by_one_tensor(self: any, my_output_dump_data: Tensor, ground_truth_dump_data: Tensor,
                               my_output_tensor: any, ground_truth_tensor: any) -> (list, list):
        error_msg = []
        try:
            # 1. deserialize output data to array
            if my_output_tensor and ground_truth_tensor:
                my_output_data_array = utils.deserialize_dump_data_to_array(my_output_tensor)
                ground_truth_data_array = utils.deserialize_dump_data_to_array(ground_truth_tensor)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError):
            error_msg.append("deserialize_dump_data_to_array failed in _compare_by_one_tensor")
            algorithm_result = self.algorithm_manager.make_nan_result()
            return algorithm_result, error_msg
        
        try:
            # 2. compare by support algorithm
            algorithm_result, error_msg = self.algorithm_manager.compare(
                my_output_data_array, ground_truth_data_array,
                {'my_output_dump_file': my_output_dump_data.path,
                 'ground_truth_dump_file': ground_truth_dump_data.path,
                 'shape_type': utils.get_shape_type(my_output_tensor.shape.dim)})
        except CompareError as compare_error:
            if isinstance(compare_error, CompareError):
                error_msg.append(compare_error.message)
            algorithm_result = self.algorithm_manager.make_nan_result()

        return algorithm_result, error_msg

    def _compare_by_tensor(self: any, my_output_dump_data: Tensor, ground_truth_dump_data: Tensor,
                           tensor_type: str) -> list:
        tensor_result_list = []
        if tensor_type == ConstManager.INPUT:
            my_output_tensor_list = my_output_dump_data.data.input
            ground_truth_tensor_list = ground_truth_dump_data.data.input
            is_input = True
        else:
            my_output_tensor_list = my_output_dump_data.data.output
            ground_truth_tensor_list = ground_truth_dump_data.data.output
            is_input = False
        # compare each tensor
        for index, (my_output_tensor, ground_truth_tensor) in enumerate(
                zip(my_output_tensor_list, ground_truth_tensor_list)):
            tensor_id = '%s:%s:%d' % (my_output_dump_data.name, tensor_type, index)
            log.print_info_log('[%s] compare %s %s for %s.'
                               % (self.fusion_op_list[0].op_name,
                                  common.get_format_string(my_output_tensor.format),
                                  utils.convert_shape_to_string(my_output_tensor.shape.dim),
                                  tensor_id))
            algorithm_result, error_msg = self._compare_by_one_tensor(my_output_dump_data, ground_truth_dump_data,
                                                                      my_output_tensor, ground_truth_tensor)
            # Check whether the current input/output data overflows
            overflow_result = ''
            if self.overflow_detection:
                overflow_result = OverflowDetection.process_model_overflow_detection(my_output_dump_data.name,
                                                                                     index, is_input, my_output_tensor)
            my_output_tensor_dtype = utils.get_data_type(my_output_tensor.data_type)
            ground_truth_tensor_dtype = utils.get_data_type(ground_truth_tensor.data_type)
            my_output_tensor_address = my_output_tensor.address \
                if hasattr(my_output_tensor, 'address') else "NaN*"
            ground_truth_tensor_address = ground_truth_tensor.address \
                if hasattr(ground_truth_tensor, 'address') else "NaN*"

            # 3. merge result
            tensor_result_list.append(
                compare_result.TensorResult(tensor_id, my_output_tensor.shape.dim, [algorithm_result, overflow_result],
                                            error_msg, my_output_tensor_dtype, ground_truth_tensor_dtype,
                                            my_output_tensor_address, ground_truth_tensor_address))
        return tensor_result_list

    def compare(self: any) -> (int, bool, list):
        """
        Compare for npu vs npu by op_name
        :return ret: return code
        :return dump_match: True, at least one operator match;False, no operator match
        :return result: the compare result by the fusion op list
        """
        if len(self.fusion_op_list) == 1:
            return self._make_one_dump_file_result()
        # get my output and ground truth tensor
        my_output_dump_data = self._get_dump_data(
            self.fusion_op_list[0], self.compare_data.left_dump_info.path, ConstManager.LEFT_TYPE)
        ground_truth_dump_data = self._get_dump_data(
            self.fusion_op_list[1], self.compare_data.right_dump_info.path, ConstManager.RIGHT_TYPE)
        compare_vector_result = []
        # check npu input data valid
        input_ret, input_error_msg = self.check_tensor_valid(
            my_output_dump_data.data.input, ground_truth_dump_data.data.input, ConstManager.INPUT)
        if input_ret == CompareError.MSACCUCMP_NONE_ERROR:
            # compare input
            compare_vector_result += self._compare_by_tensor(my_output_dump_data, ground_truth_dump_data,
                                                             ConstManager.INPUT)

        # check npu output data valid
        output_ret, output_error_msg = self.check_tensor_valid(
            my_output_dump_data.data.output, ground_truth_dump_data.data.output, ConstManager.OUTPUT)

        if output_ret == CompareError.MSACCUCMP_NONE_ERROR:
            # compare output
            compare_vector_result += self._compare_by_tensor(my_output_dump_data, ground_truth_dump_data,
                                                             ConstManager.OUTPUT)
        error_msg = []
        # if no input and output, result is NaN
        if input_ret != CompareError.MSACCUCMP_NONE_ERROR and \
                output_ret != CompareError.MSACCUCMP_NONE_ERROR:
            error_msg.append(input_error_msg)
            error_msg.append(output_error_msg)
            compare_vector_result = None
        else:
            output_ret = CompareError.MSACCUCMP_NONE_ERROR
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager)
        result = fusion_op_result.get_result(self.fusion_op_list[0], compare_vector_result, error_msg)
        return output_ret, True, result
