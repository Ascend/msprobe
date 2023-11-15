# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This FusionOpComResult class. This file mainly involves the get_result function.
"""
import collections

from vector_cmp.fusion_manager import fusion_rule_parser
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.range_manager.range_manager import RangeManager
from algorithm_manager.algorithm_manager import AlgorithmManager
from vector_cmp.fusion_manager.fusion_op import FusionOp
from cmp_utils.constant.compare_error import CompareError


class TensorResult:
    """
    The class for tensor compare result
    """

    def __init__(self: any, tensor_info: dict, result: list, error_msg: list, is_ffts: bool) -> None:
        self.tensor_info = tensor_info
        self.algorithm_result = result[0]
        self.error_msg = error_msg
        self.overflow_result = result[1]
        self.is_ffts = is_ffts

    def get_result(self: any) -> list:
        """
        Get tensor result list
        :return [tensor_id, shape, algorithm_result, error_msg]
        """
        shape_str = '[%s]' % ",".join(map(str, self.tensor_info.get("shape", ConstManager.NAN)))
        if self.overflow_result:
            result = [shape_str] + [self.overflow_result] + self.algorithm_result + [",".join(self.error_msg)]
        else:
            result = [shape_str] + self.algorithm_result + [",".join(self.error_msg)]

        if self.tensor_info.get("tensor_id"):
            result = [self.tensor_info.get("tensor_id")] + result
        return result

    def get_algorithm_result(self: any) -> list:
        """
        Get algorithm result
        """
        return self.algorithm_result

    def get_my_output_dtype(self):
        return self.tensor_info.get("my_output_dtype", ConstManager.NAN)

    def get_ground_truth_dtype(self):
        return self.tensor_info.get("ground_truth_dtype", ConstManager.NAN)

    def get_my_output_address(self):
        return self.tensor_info.get("my_output_address", ConstManager.NAN)

    def get_ground_truth_address(self):
        return self.tensor_info.get("ground_truth_address", ConstManager.NAN)

    def get_op_type(self):
        return self.tensor_info.get("op_type", ConstManager.NAN)


class PytorchOpInfo:
    """
    The class for pytorch op info
    """

    def __init__(self: any, index: int, op_name: str, my_dump_path: str, ground_truth_dump_path: str) -> None:
        self.index = index
        self.op_name = op_name
        self.my_dump_path = my_dump_path
        self.ground_truth_dump_path = ground_truth_dump_path

    def get_result(self: any) -> list:
        """
        Get op info result
        :return [index, op_name, my_dump_path, ground_truth_dump_path]
        """
        return [str(self.index), self.op_name, self.my_dump_path, self.ground_truth_dump_path]

    def get_op_name(self: any) -> str:
        """
        Get op name
        """
        return self.op_name


class FusionOpComResult:
    """
    The class for fusion op compare result
    """

    def __init__(self: any, algorithm_manager: AlgorithmManager, ground_truth_to_my_output_map: any = None,
                 overflow_detection: bool = False, dump_is_cpu_or_gpu_data: list = None) -> None:
        self.algorithm_manager = algorithm_manager
        self.ground_truth_to_my_output_map = ground_truth_to_my_output_map
        self.overflow_detection = overflow_detection
        self.is_ground_truth_gpu_or_cpu = dump_is_cpu_or_gpu_data[0] if dump_is_cpu_or_gpu_data else False
        self.is_my_dump_gpu_or_cpu = dump_is_cpu_or_gpu_data[1] if dump_is_cpu_or_gpu_data else False

    @staticmethod
    def _make_ops_without_map(fusion_op: FusionOp, no_dump_file: bool) -> (str, str):
        my_output_op = fusion_op.op_name
        ground_truth_op = fusion_op.op_name
        # if only left or right has dump file
        if fusion_op.op_type in [ConstManager.LEFT_TYPE, ConstManager.RIGHT_TYPE]:
            if no_dump_file:
                if fusion_op.op_type == ConstManager.LEFT_TYPE:
                    ground_truth_op = '*'
                elif fusion_op.op_type == ConstManager.RIGHT_TYPE:
                    my_output_op = '*'
        else:
            ground_truth_op = ','.join(fusion_op.attr.original_op_names)
            if ground_truth_op == '':
                ground_truth_op = '*'
        return my_output_op, ground_truth_op

    @staticmethod
    def _process_input_and_output(result, input_result_list, output_result_list):
        if ConstManager.INPUT_PATTERN in result[ConstManager.TENSOR_INDEX]:
            input_result_list.append(result)
        elif ConstManager.OUTPUT_PATTERN in result[ConstManager.TENSOR_INDEX]:
            output_result_list.append(result)
        return input_result_list, output_result_list

    def get_result(self: any, fusion_op: FusionOp, tensor_result: any, error_msg: list,
                   no_dump_file: bool = False) -> any:
        """
        Get fusion op compare result list
        :param fusion_op: the fusion op
        :param tensor_result: the tensor result list
        :param error_msg: error message
        :param no_dump_file: the result no dump file
        :return [op_id, my_output_op, ground_truth_op, tensor_id, shape, algorithm_result, error_msg]
        """
        result_list = []
        input_result_list = []
        output_result_list = []
        is_ffts = False
        my_output_op, ground_truth_op = self._make_my_output_op_and_ground_truth_op(fusion_op, no_dump_file)
        if tensor_result:
            for item in tensor_result:
                current_tensor_info = [
                    str(fusion_op.op_id), item.get_op_type(),
                    my_output_op, str(item.get_my_output_dtype()),
                    str(item.get_my_output_address()), ground_truth_op,
                    str(item.get_ground_truth_dtype()), str(item.get_ground_truth_address())
                ]
                self._pre_handle_result(current_tensor_info)
                result = current_tensor_info + item.get_result()
                if item.is_ffts:
                    is_ffts = True
                    input_result_list, output_result_list = \
                        self._process_input_and_output(result, input_result_list, output_result_list)
                RangeManager.adjust_data(result, fusion_op.attr.get_op_sequence())
                log.print_info_log('[{}] Result: {}'.format(fusion_op.op_name, " ".join(result)))
                result_list.append(result)
        else:
            current_tensor_info = [
                str(fusion_op.op_id), fusion_op.get_real_op_type(),
                my_output_op, ConstManager.NAN, ConstManager.NAN,
                ground_truth_op, ConstManager.NAN, ConstManager.NAN,
                ConstManager.NAN, ConstManager.NAN
            ]
            self._pre_handle_result(current_tensor_info)
            if self.overflow_detection:
                # using 'NaN' as an overflow detection for 'no tensor_result'
                # and insert it after the column 'Shape'.
                result = current_tensor_info + ['NaN'] + self.algorithm_manager.make_nan_result() \
                         + [",".join(error_msg)]
            else:
                result = current_tensor_info + self.algorithm_manager.make_nan_result() + [",".join(error_msg)]
            RangeManager.adjust_data(result, fusion_op.attr.get_op_sequence())
            log.print_info_log('[{}] Result: {}'.format(fusion_op.op_name, " ".join(result)))
            result_list.append(result)

        Result = collections.namedtuple("Result", ["result_list", "input_result_list", "output_result_list", "is_ffts"])
        result = Result(result_list, input_result_list, output_result_list, is_ffts)
        return result

    def get_pytorch_result(self: any, op_info: PytorchOpInfo, tensor_result: any, error_msg: list) -> list:
        """
        Get fusion op compare result list
        :param op_info: the pytorch op info
        :param tensor_result: the tensor result list
        :param error_msg: error message
        :return [op_id, my_output_op, ground_truth_op, tensor_id, shape, algorithm_result, error_msg]
        """
        result_list = []
        if tensor_result:
            for item in tensor_result:
                result = op_info.get_result() + [item.get_my_output_dtype()] + item.get_result()
                log.print_info_log('[%s:%d] Result: %s' % (op_info.op_name, op_info.index, " ".join(result)))
                result_list.append(result)
        else:
            result = op_info.get_result() + [ConstManager.NAN] + [ConstManager.NAN] + \
                     self.algorithm_manager.make_nan_result() + [",".join(error_msg)]
            log.print_info_log('[%s:%d] Result: %s' % (op_info.op_name, op_info.index, " ".join(result)))
            result_list.append(result)
        return result_list

    def _make_my_output_op_and_ground_truth_op(self: any, fusion_op: FusionOp, no_dump_file: bool) -> (str, str):
        if self.ground_truth_to_my_output_map:
            my_output_op, ground_truth_op = fusion_rule_parser. \
                make_left_and_right_string(self.ground_truth_to_my_output_map)
        else:
            my_output_op, ground_truth_op = self._make_ops_without_map(fusion_op, no_dump_file)
        return my_output_op, ground_truth_op

    def _pre_handle_result(self: any, current_tensor_info: list) -> None:
        """
        if dump data is not NPU data, the result will be popped.
        the index of my dump data address is 3, the index of ground truth data address is 6.
        args: result list
        """
        if self.is_ground_truth_gpu_or_cpu:
            # op id is inserted as index in header, so address index should plus one.
            current_tensor_info.pop(ConstManager.GROUND_TRUTH_ADDRESS_INDEX + 1)
        if self.is_my_dump_gpu_or_cpu:
            current_tensor_info.pop(ConstManager.MY_OUTPUT_ADDRESS_INDEX + 1)


def get_result_title(algorithm_manager: AlgorithmManager, op_header: list, overflow_detection: bool = False) -> list:
    """
    Get result title
    :param algorithm_manager: the algorithm manager
    :param op_header: the op header
    :param overflow_detection: whether to display overflow info
    :return  [Index, op_header, shape, algorithm_result, error_msg]
    """

    header = ['Index'] + op_header + ['Shape'] + algorithm_manager.get_result_title() + ["CompareFailReason"]
    # add 'OverFlow' after 'Shape'.
    if overflow_detection:
        header.insert(header.index('Shape') + 1, 'OverFlow')
    RangeManager.adjust_header(header)
    return header


class SingleOpCmpResult:
    """
    The class for single op result
    """
    def __init__(self: any) -> None:
        self.op_name = ""
        self.dump_match = False
        self.result_list = None
        self.ret = 0
        self.input_list = None
        self.input_result_list = None
        self.output_result_list = None
        self.is_ffts = False
        self.op_name_origin_output_index_map = None
        self.npu_vs_npu = False

    @staticmethod
    def get_pre_op_output(op_name: str, index: int, result_mapping: dict) -> list:
        pre_op_result = result_mapping.get(op_name)
        if pre_op_result:
            output_result = pre_op_result.output_result_list[index]
        else:
            output_result = None
            message = "The result of '%s' is not in result mapping" % op_name
            log.print_warn_log(message)
        return output_result

    def update_attr(self: any, result_info: collections.namedtuple) -> None:
        self.op_name = result_info.op_name
        self.dump_match = result_info.dump_match
        self.result_list = result_info.result_list
        self.ret = result_info.ret
        self.input_list = result_info.input_list
        self.input_result_list = result_info.input_result_list
        self.output_result_list = result_info.output_result_list
        self.is_ffts = result_info.is_ffts
        self.op_name_origin_output_index_map = result_info.op_name_origin_output_index_map
        self.npu_vs_npu = result_info.npu_vs_npu

    def check_result_list_valid(self: any) -> None:
        if len(self.result_list) < len(self.input_result_list):
            message = "The length of input result list is greater than result list, '%s'" % self.op_name
            log.print_error_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_INPUT_MAPPING)

    def find_pre_op(self: any, result_mapping: dict) -> None:
        """
        Replace the input of the current operator with the previous output result
        """
        self.check_result_list_valid()
        for index, input_result in enumerate(self.input_result_list):
            if len(input_result) <= ConstManager.TENSOR_INDEX:
                log.print_warn_log(f"Broken result, id {index}, skip")
                continue

            tensor_id = input_result[ConstManager.TENSOR_INDEX]
            pre_op = self.op_name_origin_output_index_map.get(tensor_id)
            if not pre_op or len(pre_op) < 2:
                message = "The tensor index '%s' is invalid, no input mapping information" % tensor_id
                log.print_error_log(message)
                raise CompareError(CompareError.MSACCUCMP_INVALID_INPUT_MAPPING)
            pre_op_name = pre_op[0]
            pre_op_index = pre_op[1]
            output_result = self.get_pre_op_output(pre_op_name, pre_op_index, result_mapping)
            if not output_result:
                continue
            origin_result = self.result_list[index][:ConstManager.TENSOR_INDEX + 1]
            self.result_list[index] = origin_result + output_result[ConstManager.TENSOR_INDEX + 1:]
