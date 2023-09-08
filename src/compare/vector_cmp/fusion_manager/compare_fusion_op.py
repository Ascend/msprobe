
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
FusionOpComparison class. This class mainly involves the compare function.
"""
import collections

from vector_cmp.fusion_manager import fusion_rule_parser
from cmp_utils import utils, utils_type
from cmp_utils import log
from vector_cmp.fusion_manager.compare_npu_vs_npu import NpuVsNpuComparison
from vector_cmp.fusion_manager import compare_result
from dump_parse.dump import DumpType
from overflow.overflow_detection import OverflowDetection
from vector_cmp.range_manager.range_manager import RangeManager
from vector_cmp.fusion_manager.fusion_op import FusionOp
from vector_cmp.fusion_manager.fusion_op import Tensor
from cmp_utils.constant.const_manager import ConstManager
from conversion.tensor_conversion import TensorConversion
from cmp_utils.constant.compare_error import CompareError
from dump_parse.dump_data_object import DumpDataObj
from dump_parse import dump_utils
from cmp_utils import common


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
        self.fusion_op_list = self.compare_rule.fusion_info.fusion_op_name_to_op_map[fusion_op_name]
        self.left_dump_file_path = ''
        self.left_dump_data = None
        self.overflow_detection = arguments.get('overflow_detection', False)
        self.is_ground_truth_gpu_or_cpu = \
            True if utils.dump_path_contains_npy(arguments.get("golden_dump_path")) else False
        self.is_my_dump_gpu_or_cpu = \
            True if utils.dump_path_contains_npy(arguments.get("my_dump_path")) else False
        self.op_name_origin_output_index_map = {}
        self.max_cmp_size = arguments.get('max_cmp_size')

    @staticmethod
    def _check_dequant_op(fusion_op: FusionOp) -> bool:
        if fusion_op.op_type == 'AscendDequant':
            return True
        for suffix in ConstManager.DEQUANT_OP_NANE_SUFFIX_LIST:
            if fusion_op.op_name.endswith(suffix):
                return True
        return False

    @staticmethod
    def _handle_error_msg(compare_error: CompareError, fusion_op: FusionOp, tensor_id: str) -> str:
        if compare_error.code == CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR:
            error_msg = [log.print_no_right_dump_file_error(fusion_op.op_name, tensor_id)]
        elif compare_error.code == CompareError.MSACCUCMP_INVALID_FRACTAL_NZ_DUMP_DATA_ERROR:
            error_msg = [log.print_invalid_nz_dump_data(compare_error.message, op_name=fusion_op.op_name)]
        else:
            error_msg = [compare_error.message]
        return error_msg

    def sort_l1_fusion_dump_file(self: any) -> list:
        """
        Sort l1 fusion dump file by timestamp
        :return: the sorted dump file by timestamp
        """
        timestamp_list = []
        for fusion_op in self.fusion_op_list:
            try:
                file_path_list, data_mode = self.compare_data.left_dump_info.get_op_dump_file(
                    fusion_op.op_name, print_log=False)
            except CompareError:
                continue
            finally:
                pass
            file_path = file_path_list[-1]
            if data_mode == ConstManager.NORMAL_MODE:
                timestamp = dump_utils.get_normal_timestamp(file_path)
            else:
                timestamp = dump_utils.get_ffts_timestamp(file_path)
            original_names = utils.get_string_from_list(
                fusion_op.attr.original_op_names)
            timestamp_list.append([timestamp, fusion_op, original_names])
        # sort by timestamp
        sort_timestamp_list = sorted(timestamp_list, key=lambda s: s[self.TIMESTAMP_INDEX])
        return sort_timestamp_list

    def get_right_dump_data(self: any, fusion_op: FusionOp, index: int, is_input: bool = False,
                            parse: bool = True, tensor_id: str = None) -> Tensor:
        """
        Get the right dump file path and data by left dump data
        :param fusion_op: the fusion op
        :param index: the index
        :param is_input: the left data is input or not
        :param parse: need parse dump data or not
        :param tensor_id: tensor index
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
        if is_input and tensor_id:
            self.op_name_origin_output_index_map[tensor_id] = (compare_fusion_op.op_name, compare_index)
        left_data_type = ConstManager.INPUT if is_input else ConstManager.OUTPUT
        log.print_info_log('[%s] Left(%s:%s:%d) <======> Right(%s:%s:%d)'
                           % (fusion_op.op_name, fusion_op.op_name, left_data_type, index, origin_tensor.name,
                              ConstManager.OUTPUT, origin_tensor.index))
        if not parse:
            index = None if self.compare_data.right_dump_info.type == DumpType.Offline else origin_tensor.index
            dump_file_list, _ = self.compare_data.right_dump_info.get_op_dump_file(origin_tensor.name, index)
            dump_file_path = dump_file_list[-1]
            origin_tensor.set_path(dump_file_path)
            return origin_tensor
        dump_file_path, dump_data = self.compare_data.get_right_dump_data(
            origin_tensor.name, origin_tensor.index)
        origin_tensor.set_path(dump_file_path)
        if self.compare_data.right_dump_info.type == DumpType.Offline:
            if origin_tensor.index >= len(dump_data.output_data):
                log.print_out_of_range_error(fusion_op.op_name, "output", origin_tensor.index,
                                             '[0, %d)' % len(dump_data.output_data))
                raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

            origin_tensor.set_data(dump_data.output_data[origin_tensor.index])
            origin_tensor.tensor_format = \
                common.get_format_string(dump_data.output_data[origin_tensor.index].tensor_format)
            origin_tensor.shape = dump_data.output_data[origin_tensor.index].shape

        else:
            origin_tensor.set_data(dump_data.output_data[0])
        return origin_tensor

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
        if relation in (utils_type.FusionRelation.OneToOne, utils_type.FusionRelation.MultiToOne):
            dump_match, result, ret = self._compare_for_any_to_one()
        elif relation in (utils_type.FusionRelation.OneToMulti, utils_type.FusionRelation.MultiToMulti):
            dump_match, result, ret = self._compare_for_any_to_multi(right_to_left_map)
        elif relation == utils_type.FusionRelation.L1Fusion:
            dump_match, result, ret = self._compare_for_l1_fusion(right_to_left_map)
        return ret, dump_match, result

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
            self._make_mapping_by_input(fusion_op, my_output_dump_data.input_data, my_output_dump_path, table_content)
            self._make_mapping_by_output(fusion_op, my_output_dump_data.output_data, my_output_dump_path, table_content)
        return table_content

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

    def _compare_for_any_to_one(self: any) -> (bool, list, int):
        """
        Compare for any to one relation fusion op
        """
        dump_match = False
        cmp_result_list = []
        # compare each fusion op
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager,
                                                            overflow_detection=self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])
        ret = CompareError.MSACCUCMP_NONE_ERROR

        for fusion_op in self.fusion_op_list:
            single_op_cmp_result = compare_result.SingleOpCmpResult()
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
            _result = fusion_op_result.get_result(fusion_op, result, error_msg)

            result_info = utils.ResultInfo(
                fusion_op.op_name, dump_match, _result.result_list, ret,
                fusion_op.input_list, _result.input_result_list, _result.output_result_list,
                _result.is_ffts, self.op_name_origin_output_index_map, False)

            single_op_cmp_result.update_attr(result_info)
            cmp_result_list.append(single_op_cmp_result)
        return dump_match, cmp_result_list, ret

    def _compare_for_any_to_multi(self: any, right_to_left_map: dict) -> (bool, list, int):
        """
        Compare for any to multi relation fusion op
        """
        dump_match = False
        has_result_for_any_to_multi = False
        error_msg = []
        cmp_result_list = []
        ret = CompareError.MSACCUCMP_NONE_ERROR
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager, right_to_left_map,
                                                            self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])

        # compare each fusion op
        for fusion_op in self.fusion_op_list:
            single_op_cmp_result = compare_result.SingleOpCmpResult()
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
            _result = fusion_op_result.get_result(fusion_op, result, error_msg)

            result_info = utils.ResultInfo(
                fusion_op.op_name, dump_match, _result.result_list, ret,
                fusion_op.input_list, _result.input_result_list, _result.output_result_list,
                _result.is_ffts, self.op_name_origin_output_index_map, False)

            single_op_cmp_result.update_attr(result_info)
            cmp_result_list.append(single_op_cmp_result)
        # no result for any to multi, write empty result to file
        if not has_result_for_any_to_multi:
            single_op_cmp_result = compare_result.SingleOpCmpResult()
            _result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)

            result_info = utils.ResultInfo(
                self.fusion_op_list[0].op_name, dump_match, _result.result_list, ret,
                self.fusion_op_list[0].input_list, _result.input_result_list, _result.output_result_list,
                _result.is_ffts, self.op_name_origin_output_index_map, False)

            single_op_cmp_result.update_attr(result_info)
            cmp_result_list.append(single_op_cmp_result)

        return dump_match, cmp_result_list, ret

    def _compare_for_l1_fusion_not_timestamp(self, fusion_op_result):
        error_msg = []

        single_op_cmp_result = compare_result.SingleOpCmpResult()
        error_msg.append("The fusion operator list is empty")
        _result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)
        result_info = utils.ResultInfo(self.fusion_op_list[0].op_name, False, _result.result_list,
                                       CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR, self.fusion_op_list[0].input_list,
                                       _result.input_result_list, _result.output_result_list, _result.is_ffts,
                                       self.op_name_origin_output_index_map, False)
        single_op_cmp_result.update_attr(result_info)
        return False, [single_op_cmp_result], CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR

    def _compare_for_l1_fusion(self: any, right_to_left_map: dict) -> (bool, list, int):
        timestamp_list = self.sort_l1_fusion_dump_file()
        error_msg = []
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager, right_to_left_map,
                                                            self.overflow_detection,
                                                            dump_is_cpu_or_gpu_data=[self.is_ground_truth_gpu_or_cpu,
                                                                                     self.is_my_dump_gpu_or_cpu])

        if not timestamp_list:
            return self._compare_for_l1_fusion_not_timestamp(fusion_op_result)

        # if the dump data is only input,
        # the dump file of min and max timestamp for input is the same
        # if the dump data is only output,
        # the dump file of max timestamp for output is complete
        # if the dump data contains input and output,
        # the dump file of max timestamp for output is complete
        # the dump file of min timestamp for input is complete
        l1_fusion_list = []
        if timestamp_list[-1][self.ORIGINAL_NAMES_INDEX] != timestamp_list[0][self.ORIGINAL_NAMES_INDEX]:
            l1_fusion_list.append(timestamp_list[0][ConstManager.FUSION_OP_INDEX])
        l1_fusion_list.append(timestamp_list[-1][ConstManager.FUSION_OP_INDEX])
        dump_match = False
        has_result_for_any_to_multi = False
        ret = CompareError.MSACCUCMP_NONE_ERROR
        cmp_result_list = []

        # compare each fusion op
        for fusion_op in l1_fusion_list:
            single_op_cmp_result = compare_result.SingleOpCmpResult()
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
            error_msg.clear()
            has_result_for_any_to_multi = True
            dump_match = True
            # 2. write compare result to file
            _result = fusion_op_result.get_result(fusion_op, result, error_msg)

            result_info = utils.ResultInfo(fusion_op.op_name, dump_match, _result.result_list, ret,
                fusion_op.input_list, _result.input_result_list, _result.output_result_list,
                _result.is_ffts, self.op_name_origin_output_index_map, False)

            single_op_cmp_result.update_attr(result_info)
            cmp_result_list.append(single_op_cmp_result)

        # no result for any to multi, write empty result to file
        if not has_result_for_any_to_multi:
            single_op_cmp_result = compare_result.SingleOpCmpResult()
            _result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)

            result_info = utils.ResultInfo(
                self.fusion_op_list[0].op_name, dump_match, _result.result_list, ret,
                self.fusion_op_list[0].input_list, _result.input_result_list, _result.output_result_list,
                _result.is_ffts, self.op_name_origin_output_index_map, False)

            single_op_cmp_result.update_attr(result_info)
            cmp_result_list.append(single_op_cmp_result)

        return dump_match, cmp_result_list, ret

    def _find_pre_op(self: any, fusion_op: FusionOp, index: int = 0) -> any:
        input_tensor = fusion_op.get_input_tensor(index)
        _, compare_fusion_op = \
            self.compare_rule.fusion_info.get_fusion_op_list(input_tensor.name)
        compare_index = input_tensor.index
        return compare_fusion_op, compare_index

    def _compare_by_one_tensor(self: any, fusion_op: FusionOp, index: int,
                               is_input: bool, tensor: any, tensor_id: str) -> (list, list, list):
        # 1. get ground truth dump data by original op name and index
        ground_truth_tensor = self.get_right_dump_data(fusion_op, index, is_input, tensor_id=tensor_id)
        if tensor.original_shape:
            ground_truth_tensor.shape = tensor.original_shape
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
             'shape_type': utils.get_shape_type(my_output_shape)},
            self.max_cmp_size)

        return result, compare_fail_message, [my_output_shape, my_output_dtype, ground_truth_dtype]

    def _compare_by_tensor(self: any, fusion_op: FusionOp, is_input: bool) -> (bool, list):
        tensor_list = self.left_dump_data.input_data if is_input else self.left_dump_data.output_data
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
            tensor_info = {
                "tensor_id": result.tensor_id,
                "shape": result.shape,
                "op_type": fusion_op.op_type,
                "my_output_dtype": result.my_output_dtype,
                "ground_truth_dtype": result.ground_truth_dtype,
                "my_output_address": result.my_output_address,
                "ground_truth_address": result.ground_truth_address
            }
            tensor_result = compare_result.TensorResult(tensor_info, [result.algorithm_result, overflow_result],
                                                        result.error_msg, tensor.is_ffts)
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
                fusion_op, index, is_input, tensor, tensor_id)
        except CompareError as compare_error:
            error_msg = self._handle_error_msg(compare_error, fusion_op, tensor_id)
            algorithm_result = self.algorithm_manager.make_nan_result()
            shape = tensor.shape
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

    def _get_my_output_dump_path(self: any, fusion_op: FusionOp, table_content: list) -> (bool, str):
        output_index = None
        left_dump_info = self.compare_data.left_dump_info
        if left_dump_info.type == DumpType.Quant:
            output_index = 0
        try:
            return True, self.compare_data.left_dump_info.get_op_dump_file(fusion_op.op_name, output_index)[0][-1]
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
                         table_content: list) -> (bool, DumpDataObj):
        try:
            return True, dump_utils.parse_dump_file(my_output_dump_path, self.compare_data.dump_version)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError,
                AttributeError, CompareError):
            log.print_error_log("{} file parse error".format(my_output_dump_path))
            table_content.append(self._make_mapping_table_content(
                ConstManager.NAN, my_output_dump_path, ConstManager.NAN, fusion_op))
        finally:
            pass
        return False, DumpDataObj()

    def _make_mapping_table_content(self: any, tensor_id: str, my_output_dump_path: str, ground_truth_dump_path: str,
                                    fusion_op: FusionOp) -> list:
        relation = fusion_rule_parser.get_relation_for_fusion(self.fusion_op_list)
        ground_truth_to_my_output_map = fusion_rule_parser.make_right_to_left_multi_map(self.fusion_op_list)
        if relation in (utils_type.FusionRelation.OneToOne, utils_type.FusionRelation.MultiToOne):
            ground_truth_op_str = utils.get_string_from_list(fusion_op.attr.original_op_names)
            if ground_truth_op_str == "":
                ground_truth_op_str = "*"
            my_output_op_str = fusion_op.op_name
        else:
            my_output_op_str, ground_truth_op_str = fusion_rule_parser.make_left_and_right_string(
                ground_truth_to_my_output_map)
        if fusion_op.op_type in [ConstManager.LEFT_TYPE, ConstManager.RIGHT_TYPE]:
            op_type = dump_utils.get_op_type_from_file_name(my_output_dump_path)
        else:
            op_type = fusion_op.op_type
        row_content = [
            str(fusion_op.op_id), op_type, my_output_op_str, ground_truth_op_str,
            tensor_id, my_output_dump_path, ground_truth_dump_path
        ]
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
