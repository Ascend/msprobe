
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
DetailComparison class. This class mainly involves the compare function.
"""

import os

from vector_cmp.fusion_manager import fusion_rule_parser
from cmp_utils import utils, utils_type, path_check
from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from dump_parse import dump, mapping
from vector_cmp.compare_detail.detail_writer import DetailWriter
from vector_cmp.compare_detail.detail import DetailInfo
from vector_cmp.fusion_manager.compare_fusion_op import FusionOpComparison
from conversion.tensor_conversion import TensorConversion


class DetailComparison:
    """
    The class for fusion op compare
    """

    def __init__(self: any, detail_info: DetailInfo, fusion_op_comparison: FusionOpComparison,
                 output_path: str) -> None:
        self.detail_info = detail_info
        self.fusion_op_comparison = fusion_op_comparison
        self.detail_writer = DetailWriter(output_path, detail_info)
        self.fusion_op = None

    def check_index_valid(self: any, my_output_data: any) -> None:
        """
        Check index valid
        :param my_output_data: my output data
        """
        if self.detail_info.tensor_id.is_input():
            count = len(my_output_data.input_data)
        else:
            count = len(my_output_data.output_data)
        if self.detail_info.tensor_id.index >= count:
            log.print_out_of_range_error(self.fusion_op.op_name, self.detail_info.tensor_id.tensor_type,
                                         self.detail_info.tensor_id.index, '[0, %d)' % count)
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def compare(self: any) -> int:
        """
        Compare detail by op name
        :return error_code
        """
        tensor_id = self.detail_info.tensor_id.get_tensor_id()
        log.print_info_log('[%s] Start to compare detail for %s.'
                           % (self.detail_info.tensor_id.op_name, tensor_id))
        # get fusion op list by op name
        tensor_list, dump_file_name = self._get_my_output_tensor_list()

        # delete old result
        try:
            self.detail_writer.delete_old_detail_result_files()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as error:
            log.print_error_log('Failed to delete the old detail result file. %s' % error)
            raise CompareError(CompareError.MSACCUCMP_DELETE_FILE_ERROR) from error

        # get right dump data by original op name and index
        try:
            ground_truth_tensor = \
                self.fusion_op_comparison.get_right_dump_data(
                    self.fusion_op, self.detail_info.tensor_id.index, self.detail_info.tensor_id.is_input())
        except CompareError as compare_error:
            if compare_error.code == CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
                log.print_no_right_dump_file_error(self.fusion_op.op_name, tensor_id, is_error=True)
            raise compare_error

        # get format by right output info
        self.detail_info.set_detail_format(utils.convert_shape_to_string(
            tensor_list[self.detail_info.tensor_id.index].shape),
            tensor_list[self.detail_info.tensor_id.index].tensor_format,
            ground_truth_tensor.tensor_format
        )
        # deserialize output data to array
        tensor_conversion = TensorConversion(self.fusion_op, self.fusion_op_comparison.format_manager,
                                             is_detail=True)
        my_output_array, ground_truth_array, my_output_shape = \
            tensor_conversion.get_my_output_and_ground_truth_data(
                self.fusion_op_comparison.compare_data, tensor_list[self.detail_info.tensor_id.index],
                ground_truth_tensor)
        self.detail_writer.write(my_output_shape, my_output_array, ground_truth_array, dump_file_name)
        return CompareError.MSACCUCMP_NONE_ERROR

    def _print_l1_fusion_warning(self: any) -> None:
        timestamp_list = self.fusion_op_comparison.sort_l1_fusion_dump_file()
        max_timestamp_op_name = timestamp_list[-1][ConstManager.FUSION_OP_INDEX].op_name
        if self.fusion_op.op_name != max_timestamp_op_name:
            log.print_warn_log(
                'The dump data of %s is incomplete, the comparison may be far away. '
                'The dump data of %s may be complete, It is best to use %s for comparison.'
                % (self.fusion_op.op_name, max_timestamp_op_name, max_timestamp_op_name))

    def _get_my_output_tensor_by_op(self: any, relation: int) -> any:
        my_output_data_path, my_output_data = \
            self.fusion_op_comparison.compare_data.get_left_dump_data(self.fusion_op.op_name)
        dump_file = os.path.basename(my_output_data_path)
        self.check_index_valid(my_output_data)
        if self.detail_info.tensor_id.is_input():
            tensor_list = my_output_data.input_data
        else:
            if relation == utils_type.FusionRelation.L1Fusion:
                self._print_l1_fusion_warning()
            tensor_list = my_output_data.output_data
        return tensor_list, dump_file

    def _get_my_output_tensor_list(self: any) -> list:
        self.fusion_op, fusion_op_list = self.detail_info.get_detail_op(
            self.fusion_op_comparison.compare_rule.fusion_info)
        if self.fusion_op.is_inner_node():
            log.print_skip_inner_op_msg(self.fusion_op.op_name, is_error=True)
            raise CompareError(CompareError.MSACCUCMP_UNSUPPORTED_COMPARE_ERROR)
        if self.fusion_op.attr.quant_filter:
            log.print_error_log("[{}] The -op param should not specify an operator in quant/dequant pair."
                                .format(self.fusion_op.op_name))
            raise CompareError(CompareError.MSACCUCMP_UNSUPPORTED_COMPARE_ERROR)
        relation = fusion_rule_parser.get_relation_for_fusion(fusion_op_list)
        try:
            tensor_list, dumpfile = self._get_my_output_tensor_by_op(relation)
        except CompareError as compare_error:
            if compare_error.code == CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
                log.print_no_left_dump_file_error(self.fusion_op.op_name, self.fusion_op.op_type, True)
            raise compare_error
        finally:
            pass
        return tensor_list, dumpfile


class DumpDetailComparison:
    """
    The class for Npu dump op compare
    """

    def __init__(self: any, detail_info: DetailInfo, compare_data: dump.CompareData,
                 output_path: str) -> None:
        self.detail_info = detail_info
        self.compare_data = compare_data
        self.detail_writer = DetailWriter(output_path, detail_info)
        self.fusion_op = None

    @staticmethod
    def _check_index_valid(tensor_data, op_name, tensor_index, tensor_type):
        if tensor_index >= len(tensor_data):
            log.print_out_of_range_error(op_name, tensor_type, tensor_index, '[0, %d)' % len(tensor_data))
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def compare(self: any) -> int:
        """
        Compare detail by op name
        :return error_code
        """
        tensor_id = self.detail_info.tensor_id.get_tensor_id()
        op_name = self.detail_info.tensor_id.op_name
        tensor_type = self.detail_info.tensor_id.tensor_type
        tensor_index = self.detail_info.tensor_id.index
        log.print_info_log('[%s] Start to compare detail for %s.'
                           % (op_name, tensor_id))

        left_path, left_tensor_data, right_tensor_data = self.get_tensor_data(op_name, tensor_type)

        self._check_index_valid(left_tensor_data, op_name, tensor_index, tensor_type)
        self._check_index_valid(right_tensor_data, op_name, tensor_index, tensor_type)

        my_output_shape = tuple(left_tensor_data[tensor_index].shape)
        ground_truth_shape = tuple(right_tensor_data[tensor_index].shape)
        if my_output_shape != ground_truth_shape:
            log.print_error_log("My Output data shape %s not equal to Ground Truth data shape %s."
                                "Can not compare." % (my_output_shape, ground_truth_shape))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

        my_output_array = left_tensor_data[tensor_index].data
        ground_truth_array = right_tensor_data[tensor_index].data

        self.detail_info.set_detail_ops(op_name, op_name)
        self.detail_info.check_and_set_format(utils.convert_shape_to_string(
            left_tensor_data[tensor_index].shape),
            left_tensor_data[tensor_index].tensor_format,
            right_tensor_data[tensor_index].tensor_format
        )

        # delete old result
        try:
            self.detail_writer.delete_old_detail_result_files()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as error:
            log.print_error_log('Failed to delete the old detail result file. %s' % error)
            raise CompareError(CompareError.MSACCUCMP_DELETE_FILE_ERROR) from error

        dump_file = os.path.basename(left_path)
        self.detail_writer.write(my_output_shape, my_output_array, ground_truth_array, dump_file)
        return CompareError.MSACCUCMP_NONE_ERROR

    def get_tensor_data(self, op_name, tensor_type):
        """
        Get tensor form dump data
        :return left_path
        :return left_tensor_data
        :return right_tensor_data
        """
        try:
            left_path, left_data = self.compare_data.get_left_dump_data(op_name)
        except CompareError as compare_error:
            log.print_error_log('Failed to find %s dump data from -m dump data path.' % op_name)
            raise compare_error
        finally:
            pass
        try:
            right_path, right_data = self.compare_data.get_right_dump_data(op_name)
        except CompareError as compare_error:
            log.print_error_log('Failed to find %s dump data from -g dump data path.' % op_name)
            raise compare_error
        finally:
            pass
        log.print_info_log("My Output data path: " + left_path)
        log.print_info_log("Ground Truth data path: " + right_path)
        if tensor_type == ConstManager.INPUT:
            left_tensor_data = left_data.input_data
            right_tensor_data = right_data.input_data
        else:
            left_tensor_data = left_data.output_data
            right_tensor_data = right_data.output_data
        return left_path, left_tensor_data, right_tensor_data
