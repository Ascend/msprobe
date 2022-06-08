#!/usr/bin/env python
# coding=utf-8
"""
Function:
FusionOpComparison class. This class mainly involves the compare function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""
import collections
import fusion_rule_parser
import utils
import log

from compare_npu_vs_npu import NpuVsNpuComparison
import compare_result
from dump import DumpType
from overflow_detection import OverflowDetection
from range_manager import RangeManager
from fusion_op import FusionOp
from fusion_op import Tensor

from const_manager import ConstManager

from tensor_conversion import TensorConversion
from compare_error import CompareError
import dump_data_pb2 as DD

import common


class FusionOpComparison:
    """
    The class for fusion op compare
    """
    TIMESTAMP_INDEX = 0
    ORIGINAL_NAMES_INDEX = 2

    def __init__(self: any, *args: any) -> None:
        fusion_op_name, compare_rule, compare_data, format_manager, arguments = args
        self.compare_data = compare_data
        self.compare_rule = compare_rule
        self.format_manager = format_manager
        self.algorithm_manager = arguments.get('algorithm_manager')
        self.fusion_op_list = \
            self.compare_rule.fusion_info.fusion_op_name_to_op_map[fusion_op_name]
        self.left_dump_file_path = ''
        self.left_dump_data = None
        self.overflow_detection = arguments.get('overflow_detection', False)
        self.is_ground_truth_gpu_or_cpu = \
            True if utils.dump_path_contains_npy(arguments.get("golden_dump_path")) else False
        self.is_my_dump_gpu_or_cpu = \
            True if utils.dump_path_contains_npy(arguments.get("my_dump_path")) else False

    def _compare_for_any_to_one(self: any) -> (bool, list, int):
        """
        Compare for any to one relation fusion op
        """
        dump_match = False
        # compare each fusion op
        result_list = []
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager,
                                                            overflow_detection=self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])
        ret = CompareError.MSACCUCMP_NONE_ERROR
        for fusion_op in self.fusion_op_list:
            # skip not support compare op
            if fusion_op.attr.quant_filter:
                log.print_skip_quant_info(fusion_op.op_name)
                continue
            result = None
            # 1. compare op
            error_msg = []
            try:
                result = self._compare_by_operator(fusion_op)
            except CompareError as compare_error:
                ret = compare_error.code
                error_msg.append(compare_error.message)
            if result:
                dump_match = True
            # 2. make compare result to string
            result_list += fusion_op_result.get_result(fusion_op, result, error_msg)
        return dump_match, result_list, ret

    def _compare_for_any_to_multi(self: any, right_to_left_map: dict) -> (bool, list, int):
        """
        Compare for any to multi relation fusion op
        """
        dump_match = False
        has_result_for_any_to_multi = False
        result_list = []
        error_msg = []
        ret = CompareError.MSACCUCMP_NONE_ERROR
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager, right_to_left_map,
                                                            self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])
        # compare each fusion op
        for fusion_op in self.fusion_op_list:
            if fusion_op.attr.quant_filter:
                log.print_skip_quant_info(fusion_op.op_name)
                continue
            result = None
            if fusion_op.is_inner_node():
                warn_message = '[%s] The op is inner node for multi to multi relation.Skip the op.' \
                               % fusion_op.op_name
                log.print_warn_log(warn_message)
                error_msg.append(warn_message)
                continue
            # 1. compare op
            try:
                result = self._compare_by_operator(fusion_op)
            except CompareError as compare_error:
                ret = compare_error.code
                error_msg.append(compare_error.message)
                if ret == CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
                    continue
            finally:
                pass
            has_result_for_any_to_multi = True
            dump_match = True
            error_msg.clear()
            # 2. write compare result to file
            result_list += fusion_op_result.get_result(fusion_op, result, error_msg)
        # no result for any to multi, write empty result to file
        if not has_result_for_any_to_multi:
            result_list += fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)
        return dump_match, result_list, ret

    def sort_l1_fusion_dump_file(self: any) -> list:
        """
        Sort l1 fusion dump file by timestamp
        :return: the sorted dump file by timestamp
        """
        timestamp_list = []
        for fusion_op in self.fusion_op_list:
            try:
                file_path = self.compare_data.left_dump_info.get_op_dump_file(
                    fusion_op.op_name, print_log=False)
            except CompareError:
                continue
            finally:
                pass
            index = file_path.rfind(".") + 1
            timestamp = int(file_path[index:])
            original_names = utils.get_string_from_list(
                fusion_op.attr.original_op_names)
            timestamp_list.append([timestamp, fusion_op, original_names])
        # sort by timestamp
        sort_timestamp_list = sorted(timestamp_list, key=lambda s: s[self.TIMESTAMP_INDEX])
        return sort_timestamp_list

    def _compare_for_l1_fusion(self: any, right_to_left_map: dict) -> (bool, list, int):
        timestamp_list = self.sort_l1_fusion_dump_file()
        error_msg = []
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager, right_to_left_map,
                                                            self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])
        if not timestamp_list:
            error_msg.append("The fusion operator list is empty")
            return False, fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg), \
                   CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR

        # if the dump data is only input,
        # the dump file of min and max timestamp for input is the same
        # if the dump data is only output,
        # the dump file of max timestamp for output is complete
        # if the dump data contains input and output,
        # the dump file of max timestamp for output is complete
        # the dump file of min timestamp for input is complete
        l1_fusion_list = []
        if timestamp_list[-1][self.ORIGINAL_NAMES_INDEX] != \
                timestamp_list[0][self.ORIGINAL_NAMES_INDEX]:
            l1_fusion_list.append(timestamp_list[0][ConstManager.FUSION_OP_INDEX])
        l1_fusion_list.append(timestamp_list[-1][ConstManager.FUSION_OP_INDEX])
        dump_match = False
        has_result_for_any_to_multi = False
        result_list = []
        ret = CompareError.MSACCUCMP_NONE_ERROR
        # compare each fusion op
        for fusion_op in l1_fusion_list:
            if fusion_op.attr.quant_filter:
                log.print_skip_quant_info(fusion_op.op_name)
                continue
            result = None
            # 1. compare op
            try:
                result = self._compare_by_operator(fusion_op)
            except CompareError as compare_error:
                ret = compare_error.code
                error_msg.append(compare_error.message)
                if ret == CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
                    continue
            finally:
                pass
            error_msg.clear()
            has_result_for_any_to_multi = True
            dump_match = True
            # 2. write compare result to file
            result_list += fusion_op_result.get_result(fusion_op, result, error_msg)

        # no result for any to multi, write empty result to file
        if not has_result_for_any_to_multi:
            result_list += fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)
        return dump_match, result_list, ret

    def _find_pre_op(self: any, fusion_op: FusionOp, index: int = 0) -> any:
        input_tensor = fusion_op.get_input_tensor(index)
        _, compare_fusion_op = \
            self.compare_rule.fusion_info.get_fusion_op_list(input_tensor.name)
        compare_index = input_tensor.index
        return compare_fusion_op, compare_index

    def get_right_dump_data(self: any, fusion_op: FusionOp, index: int, is_input: bool = False,
                            parse: bool = True) -> Tensor:
        """
        Get the right dump file path and data by left dump data
        :param fusion_op: the fusion op
        :param index: the index
        :param is_input: the left data is input or not
        :param parse: need parse dump data or not
        :return tensor and dump data
        """
        compare_fusion_op = fusion_op
        compare_index = index
        if is_input:
            compare_fusion_op, compare_index = self._find_pre_op(fusion_op, index)
        while compare_fusion_op.op_type in ConstManager.SPECIAL_OPS_TYPE:
            # Assume that the input and output indexes of the special op are in one-to-one correspondence.
            compare_fusion_op, compare_index = self._find_pre_op(compare_fusion_op, compare_index)

        origin_tensor = compare_fusion_op.get_origin_tensor(compare_index)
        left_data_type = ConstManager.INPUT if is_input else ConstManager.OUTPUT
        log.print_info_log('[%s] Left(%s:%s:%d) <======> Right(%s:%s:%d)'
                           % (fusion_op.op_name, fusion_op.op_name, left_data_type, index, origin_tensor.name,
                              ConstManager.OUTPUT, origin_tensor.index))
        if not parse:
            dump_file_path = self.compare_data.right_dump_info.get_op_dump_file(origin_tensor.name,
                                                                                origin_tensor.index)
            origin_tensor.set_path(dump_file_path)
            return origin_tensor
        dump_file_path, dump_data = self.compare_data.get_right_dump_data(
            origin_tensor.name, origin_tensor.index)
        origin_tensor.set_path(dump_file_path)
        if self.compare_data.right_dump_info.type == DumpType.Offline:
            if origin_tensor.index >= len(dump_data.output):
                log.print_out_of_range_error(fusion_op.op_name, "output", origin_tensor.index,
                                             '[0, %d)' % len(dump_data.output))
                raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)
            origin_tensor.set_data(dump_data.output[origin_tensor.index])
            origin_tensor.format = common.get_format_string(dump_data.output[origin_tensor.index].format)
            origin_tensor.shape = list(dump_data.output[origin_tensor.index].shape.dim)
        else:
            origin_tensor.set_data(dump_data.output[0])
        return origin_tensor

    def _compare_by_one_tensor(self: any, fusion_op: FusionOp, index: int,
                               is_input: bool, tensor: any) -> (list, list, list):
        # 1. get ground truth dump data by original op name and index
        ground_truth_tensor = self.get_right_dump_data(fusion_op, index, is_input)
        if len(tensor.original_shape.dim) != 0:
            ground_truth_tensor.shape = tensor.original_shape.dim
        # 2. deserialize output data to array
        tensor_conversion = TensorConversion(fusion_op, self.format_manager, is_detail=False)
        my_output_array, ground_truth_array, my_output_shape = \
            tensor_conversion.get_my_output_and_ground_truth_data(self.compare_data, tensor, ground_truth_tensor)
        my_output_dtype = my_output_array.dtype
        ground_truth_dtype = ground_truth_array.dtype

        # 3. compare by support algorithm
        result, compare_fail_message = self.algorithm_manager.compare(
            my_output_array, ground_truth_array,
            {'my_output_dump_file': self.left_dump_file_path,
             'ground_truth_dump_file': ground_truth_tensor.path,
             'shape_type': utils.get_shape_type(my_output_shape)})
        return result, compare_fail_message, [my_output_shape, my_output_dtype, ground_truth_dtype]

    def _compare_by_tensor(self: any, fusion_op: FusionOp, is_input: bool) -> (bool, list):
        tensor_list = self.left_dump_data.input if is_input else self.left_dump_data.output
        match = False
        tensor_result_list = []
        # compare each tensor
        for (index, tensor) in enumerate(tensor_list):
            result = self._compare(fusion_op, index, is_input, tensor)
            # Check whether the current input/output data overflows
            overflow_result = ''
            if self.overflow_detection:
                overflow_result = OverflowDetection.process_model_overflow_detection(fusion_op.op_name, index, is_input,
                                                                                     tensor)
            match = match or result.match
            # 4. merge result
            tensor_info = {"tensor_id": result.tensor_id, "shape": result.shape,
                           "my_output_dtype": result.my_output_dtype,
                           "ground_truth_dtype": result.ground_truth_dtype,
                           "my_output_address": result.my_output_address,
                           "ground_truth_address": result.ground_truth_address}
            tensor_result = compare_result.TensorResult(tensor_info, [result.algorithm_result, overflow_result],
                                                        result.error_msg)
            tensor_result_list.append(tensor_result)
        return match, tensor_result_list

    def _compare(self: any, fusion_op: FusionOp, index: int, is_input: bool, tensor: any) -> tuple:
        tensor_type = ConstManager.INPUT if is_input else ConstManager.OUTPUT
        tensor_id = "%s:%s:%d" % (fusion_op.op_name, tensor_type, index)
        match = True
        my_output_dtype = "NaN"
        ground_truth_dtype = "NaN"
        try:
            algorithm_result, error_msg, [shape, my_output_dtype, ground_truth_dtype] = self._compare_by_one_tensor(
                fusion_op, index, is_input, tensor)
        except CompareError as compare_error:
            if compare_error.code == \
                    CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
                error_msg = [log.print_no_right_dump_file_error(fusion_op.op_name, tensor_id)]
            else:
                error_msg = [compare_error.message]
            algorithm_result = self.algorithm_manager.make_nan_result()
            shape = tensor.shape.dim
            match = False

        CompareResult = collections.namedtuple(
            'Result',
            ['tensor_id', 'shape', 'algorithm_result', 'error_msg', 'match', 'my_output_dtype', 'ground_truth_dtype',
             'my_output_address', 'ground_truth_address'])
        result = CompareResult(tensor_id, shape, algorithm_result, error_msg, match, my_output_dtype,
                               ground_truth_dtype,
                               my_output_address=utils.get_address_from_tensor(tensor),
                               ground_truth_address=ConstManager.NAN)
        return result

    @staticmethod
    def _check_dequant_op(fusion_op: FusionOp) -> bool:
        if fusion_op.op_type == 'AscendDequant':
            return True
        for suffix in ConstManager.DEQUANT_OP_NANE_SUFFIX_LIST:
            if fusion_op.op_name.endswith(suffix):
                return True
        return False

    def _compare_by_operator(self: any, fusion_op: FusionOp) -> list:
        """
        Compare for one operator
        """
        log.print_start_to_compare_op(fusion_op.op_name)
        try:
            self.left_dump_file_path, self.left_dump_data = \
                self.compare_data.get_left_dump_data(fusion_op.op_name)
        except CompareError as error:
            error.message = log.print_no_left_dump_file_error(fusion_op.op_name, fusion_op.op_type)
            raise error
        finally:
            pass
        compare_vector_result = []
        if not self._check_dequant_op(fusion_op):
            _, result = self._compare_by_tensor(fusion_op, True)
            compare_vector_result += result
        if fusion_op.op_type != 'AscendQuant':
            _, result = self._compare_by_tensor(fusion_op, False)
            compare_vector_result += result
        return compare_vector_result

    def compare(self: any) -> (int, bool, list):
        """
        Compare for one the fusion op
        :return error_code:VectorComparisonErrorCode
        :return dump_match: True, at least one operator match;False, no operator match
        :return result: the compare result by the fusion op
        """
        ret = CompareError.MSACCUCMP_NONE_ERROR
        dump_match = False
        result = None
        try:
            if self.compare_rule.quant_fusion_rule_file_path == '' \
                    and self.compare_rule.fusion_json_file_path == '':
                npu_vs_npu_comparison = NpuVsNpuComparison(self.compare_data, self.fusion_op_list,
                                                           self.algorithm_manager, self.overflow_detection)
                return npu_vs_npu_comparison.compare()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError,
                AttributeError) as error:
            log.print_error_log('Failed to compare %s. %s' % (self.fusion_op_list[0].op_name, error))
            return CompareError.MSACCUCMP_UNKNOWN_ERROR, False, []

        # 1. get the fusion op relation
        relation = fusion_rule_parser.get_relation_for_fusion(self.fusion_op_list)
        # 2. get the map for {original_op_names, op_list}
        right_to_left_map = fusion_rule_parser.make_right_to_left_multi_map(self.fusion_op_list)
        # 3. compare by relation
        if relation in (utils.FusionRelation.OneToOne, utils.FusionRelation.MultiToOne):
            dump_match, result, ret = self._compare_for_any_to_one()
        elif relation in (utils.FusionRelation.OneToMulti, utils.FusionRelation.MultiToMulti):
            dump_match, result, ret = self._compare_for_any_to_multi(right_to_left_map)
        elif relation == utils.FusionRelation.L1Fusion:
            dump_match, result, ret = self._compare_for_l1_fusion(right_to_left_map)
        return ret, dump_match, result

    def _get_my_output_dump_path(self: any, fusion_op: FusionOp, table_content: list) -> (bool, str):
        output_index = None
        left_dump_info = self.compare_data.left_dump_info
        if left_dump_info.type == DumpType.Quant:
            output_index = 0
        try:
            return True, self.compare_data.left_dump_info.get_op_dump_file(fusion_op.op_name, output_index)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError,
                AttributeError, CompareError):
            message = '[{0}] There is no left dump file for the op "{0}".'.format(fusion_op.op_name)
            log.print_warn_log(message)
            table_content.append(self._make_mapping_table_content(
                ConstManager.NAN, ConstManager.NAN, ConstManager.NAN, fusion_op))
        return False, ''

    def _get_ground_truth_dump_path(self: any, fusion_op: FusionOp, index: int, is_input: bool) -> (bool, str):
        ground_truth_tensor = None
        try:
            ground_truth_tensor = self.get_right_dump_data(fusion_op, index, is_input, parse=False)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError,
                AttributeError, CompareError):
            message = '[{0}] There is no right dump file for the op "{0}".'.format(fusion_op.op_name)
            log.print_warn_log(message)

        if ground_truth_tensor:
            return ground_truth_tensor.path
        else:
            return ConstManager.NAN

    def _parse_dump_file(self: any, fusion_op: FusionOp, my_output_dump_path: str,
                         table_content: list) -> (bool, DD.DumpData):
        try:
            return True, utils.parse_dump_file(my_output_dump_path, self.compare_data.dump_version)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError,
                AttributeError, CompareError):
            log.print_error_log("{} file parse error".format(my_output_dump_path))
            table_content.append(self._make_mapping_table_content(
                ConstManager.NAN, my_output_dump_path, ConstManager.NAN, fusion_op))
        finally:
            pass
        return False, DD.DumpData()

    def make_gpu_and_npu_mapping_table(self: any) -> list:
        """
        Generate table content by op name
        :return table content
        """
        table_content = []
        for fusion_op in self.fusion_op_list:
            ok, my_output_dump_path = self._get_my_output_dump_path(fusion_op, table_content)
            if not ok:
                continue
            ok, my_output_dump_data = self._parse_dump_file(fusion_op, my_output_dump_path, table_content)
            if not ok:
                continue
            self._make_mapping_by_input(fusion_op, my_output_dump_data.input, my_output_dump_path, table_content)
            self._make_mapping_by_output(fusion_op, my_output_dump_data.output, my_output_dump_path, table_content)
        return table_content

    def _make_mapping_table_content(self: any, tensor_id: str, my_output_dump_path: str, ground_truth_dump_path: str,
                                    fusion_op: FusionOp) -> list:
        relation = fusion_rule_parser.get_relation_for_fusion(self.fusion_op_list)
        ground_truth_to_my_output_map = fusion_rule_parser.make_right_to_left_multi_map(self.fusion_op_list)
        if relation in (utils.FusionRelation.OneToOne, utils.FusionRelation.MultiToOne):
            ground_truth_op_str = utils.get_string_from_list(fusion_op.attr.original_op_names)
            if ground_truth_op_str == "":
                ground_truth_op_str = "*"
            my_output_op_str = fusion_op.op_name
        else:
            my_output_op_str, ground_truth_op_str = fusion_rule_parser.make_left_and_right_string(
                ground_truth_to_my_output_map)
        row_content = [str(fusion_op.op_id), my_output_op_str, ground_truth_op_str, tensor_id, my_output_dump_path,
                       ground_truth_dump_path]
        RangeManager.adjust_data(row_content, fusion_op.attr.get_op_sequence())
        return row_content

    def _make_mapping_by_input(self: any, fusion_op: FusionOp, input_tensor: any,
                               my_output_dump_path: str, table_content: list) -> None:
        if not self._check_dequant_op(fusion_op):
            for index, _ in enumerate(input_tensor):
                tensor_id = "%s:%s:%d" % (fusion_op.op_name, ConstManager.INPUT, index)
                ground_truth_dump_path = self._get_ground_truth_dump_path(fusion_op, index, is_input=True)
                table_content.append(self._make_mapping_table_content(
                    tensor_id, my_output_dump_path, ground_truth_dump_path, fusion_op))

    def _make_mapping_by_output(self: any, fusion_op: FusionOp, output_tensor: any,
                                my_output_dump_path: str, table_content: list) -> None:
        if fusion_op.op_type != 'AscendQuant':
            for index, _ in enumerate(output_tensor):
                tensor_id = "%s:%s:%d" % (fusion_op.op_name, ConstManager.OUTPUT, index)
                ground_truth_dump_path = self._get_ground_truth_dump_path(fusion_op, index, is_input=False)
                table_content.append(self._make_mapping_table_content(
                    tensor_id, my_output_dump_path, ground_truth_dump_path, fusion_op))
