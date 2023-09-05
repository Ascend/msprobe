# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
NpuVsNpuComparison class. This class mainly involves the compare function.
"""

from cmp_utils import utils
from cmp_utils import common
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from dump_parse.dump import CompareData
from algorithm_manager.algorithm_manager import AlgorithmManager
from vector_cmp.fusion_manager.fusion_op import FusionOp
from vector_cmp.fusion_manager.fusion_op import Tensor
from vector_cmp.fusion_manager import compare_result
from cmp_utils.constant.compare_error import CompareError
from overflow.overflow_detection import OverflowDetection
from dump_parse.ffts_parser import FFTSParser
from dump_parse import dump_utils


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

    def compare(self: any) -> (int, bool, list):
        """
        Compare for npu vs npu by op_name
        :return ret: return code
        :return dump_match: True, at least one operator match;False, no operator match
        :return result: the compare result by the fusion op list
        """
        single_op_cmp_result = compare_result.SingleOpCmpResult()
        if len(self.fusion_op_list) == 1:
            return self._make_one_dump_file_result()
        # get my output and ground truth tensor
        error_msg = []
        try:
            my_output_dump_data = self._get_dump_data(
                self.fusion_op_list[0], self.compare_data.left_dump_info.path,
                self.compare_data.left_dump_info.op_name_to_task_mode_map, ConstManager.LEFT_TYPE)
        except CompareError as error:
            error_msg.append(error.message)
            fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager)
            _result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg)
            result_info = utils.ResultInfo(
                self.fusion_op_list[0].op_name, True, _result.result_list, error.code,
                [], _result.input_result_list, _result.output_result_list, _result.is_ffts,
                {}, True)
            single_op_cmp_result.update_attr(result_info)
            return error.code, True, [single_op_cmp_result]

        ground_truth_dump_data = self._get_dump_data(
            self.fusion_op_list[1], self.compare_data.right_dump_info.path,
            self.compare_data.right_dump_info.op_name_to_task_mode_map, ConstManager.RIGHT_TYPE)

        compare_vector_result = []
        # check npu input data valid
        input_ret, input_error_msg = self.check_tensor_valid(
            my_output_dump_data.data.input_data, ground_truth_dump_data.data.input_data, ConstManager.INPUT)
        if input_ret == CompareError.MSACCUCMP_NONE_ERROR:
            # compare input
            compare_vector_result += self._compare_by_tensor(my_output_dump_data, ground_truth_dump_data,
                                                             ConstManager.INPUT)

        # check npu output data valid
        output_ret, output_error_msg = self.check_tensor_valid(
            my_output_dump_data.data.output_data, ground_truth_dump_data.data.output_data, ConstManager.OUTPUT)

        if output_ret == CompareError.MSACCUCMP_NONE_ERROR:
            # compare output
            compare_vector_result += self._compare_by_tensor(my_output_dump_data, ground_truth_dump_data,
                                                             ConstManager.OUTPUT)

        if not my_output_dump_data.data.ffts_file_check:
            msg = "This is a FFTS+ mode dump data, The number of files does not match the number of thread"
            error_msg.append(msg)
        # if no input and output, result is NaN
        if input_ret != CompareError.MSACCUCMP_NONE_ERROR and output_ret != CompareError.MSACCUCMP_NONE_ERROR:
            error_msg.append(input_error_msg)
            error_msg.append(output_error_msg)
            compare_vector_result = None
        else:
            output_ret = CompareError.MSACCUCMP_NONE_ERROR
        fusion_op_result = compare_result.FusionOpComResult(self.algorithm_manager)
        _result = fusion_op_result.get_result(self.fusion_op_list[0], compare_vector_result, error_msg)

        result_info = utils.ResultInfo(
            my_output_dump_data.name, True, _result.result_list, output_ret,
            [], _result.input_result_list, _result.output_result_list, _result.is_ffts,
            {}, True)

        single_op_cmp_result.update_attr(result_info)

        return output_ret, True, [single_op_cmp_result]

    def _make_one_dump_file_result(self: any) -> (int, bool, list):
        error_msg = []
        # if only left or right has dump file, the result is NaN
        single_op_cmp_result = compare_result.SingleOpCmpResult()
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
        _result = fusion_op_result.get_result(self.fusion_op_list[0], None, error_msg, no_dump_file=True)

        result_info = utils.ResultInfo(
            self.fusion_op_list[0].op_name, False, _result.result_list,
            CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR,
            self.fusion_op_list[0].input_list, _result.input_result_list,
            _result.output_result_list, _result.is_ffts, {}, True)

        single_op_cmp_result.update_attr(result_info)

        return CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR, False, [single_op_cmp_result]

    def _get_dump_data(self: any, fusion_op: FusionOp, dump_path: str,
                       op_name_to_task_mode_map, dump_type: str) -> Tensor:
        """
        get dump data by fusion op output_desc
        """
        dump_file_list = fusion_op.output_desc
        if not dump_file_list:
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        dump_data_list = [dump_utils.parse_dump_file(dump_file_path, self.compare_data.dump_version)
                          for dump_file_path in dump_file_list]
        dump_mode = op_name_to_task_mode_map.get(self.op_name)
        if dump_mode == ConstManager.AUTOMATIC_MODE or dump_mode == ConstManager.MANUAL_MODE:
            ffts_parser = FFTSParser(dump_file_list, dump_data_list)
            dump_file_path, dump_data = ffts_parser.parse_ffts
            log.print_info_log(
                'The "%s" in the path "%s" is FFTS+ dump data. After process the output data, the file path is "%s".'
                % (fusion_op.op_name, dump_path, dump_file_path))
        else:
            dump_file_path = dump_file_list[-1]
            dump_data = dump_data_list[-1]
        log.print_info_log('[%s] [%s] %s' % (fusion_op.op_name, dump_type, dump_file_path))
        if dump_data.op_name and dump_data.attr:
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
            if my_output_tensor.tensor_format != ground_truth_tensor.tensor_format:
                message = log.print_not_match_error(
                    self.op_name, 'format',
                    common.get_format_string(my_output_tensor.tensor_format),
                    common.get_format_string(ground_truth_tensor.tensor_format), tensor_id)
                return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message

            # check the length of shape is the same
            if len(my_output_tensor.shape) != len(ground_truth_tensor.shape):
                message = log.print_not_match_error(
                    self.op_name, 'shape',
                    utils.convert_shape_to_string(my_output_tensor.shape),
                    utils.convert_shape_to_string(ground_truth_tensor.shape), tensor_id)
                return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message
            # check each dim in shape is the same
            for my_output_dim, ground_truth_dim in zip(my_output_tensor.shape, ground_truth_tensor.shape):
                if my_output_dim != ground_truth_dim:
                    message = log.print_not_match_error(
                        self.op_name, 'shape',
                        utils.convert_shape_to_string(my_output_tensor.shape),
                        utils.convert_shape_to_string(ground_truth_tensor.shape), tensor_id)
                    return CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message
        return CompareError.MSACCUCMP_NONE_ERROR, message

    def _compare_by_one_tensor(self: any, my_output_dump_data: Tensor, ground_truth_dump_data: Tensor,
                               my_output_tensor: any, ground_truth_tensor: any) -> (list, list):
        error_msg = []
        # 1. deserialize output data to array
        if my_output_tensor and ground_truth_tensor:
            my_output_data_array = my_output_tensor.data
            ground_truth_data_array = ground_truth_tensor.data
        else:
            return self.algorithm_manager.make_nan_result(), error_msg
        
        try:
            # 2. compare by support algorithm
            algorithm_result, error_msg = self.algorithm_manager.compare(
                my_output_data_array, ground_truth_data_array,
                {'my_output_dump_file': my_output_dump_data.path,
                 'ground_truth_dump_file': ground_truth_dump_data.path,
                 'shape_type': utils.get_shape_type(my_output_tensor.shape)})
        except CompareError as compare_error:
            if isinstance(compare_error, CompareError):
                error_msg.append(compare_error.message)
            algorithm_result = self.algorithm_manager.make_nan_result()

        return algorithm_result, error_msg

    def _compare_by_tensor(self: any, my_output_dump_data: Tensor, ground_truth_dump_data: Tensor,
                           tensor_type: str) -> list:
        tensor_result_list = []
        if tensor_type == ConstManager.INPUT:
            my_output_tensor_list = my_output_dump_data.data.input_data
            ground_truth_tensor_list = ground_truth_dump_data.data.input_data
            is_input = True
        else:
            my_output_tensor_list = my_output_dump_data.data.output_data
            ground_truth_tensor_list = ground_truth_dump_data.data.output_data
            is_input = False
        # compare each tensor
        for index, (my_output_tensor, ground_truth_tensor) in enumerate(
                zip(my_output_tensor_list, ground_truth_tensor_list)):
            tensor_id = '%s:%s:%d' % (my_output_dump_data.name, tensor_type, index)
            log.print_info_log('[%s] compare %s %s for %s.'
                               % (self.fusion_op_list[0].op_name,
                                  common.get_format_string(my_output_tensor.tensor_format),
                                  utils.convert_shape_to_string(my_output_tensor.shape),
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
            my_output_tensor_address = utils.get_address_from_tensor(my_output_tensor)
            ground_truth_tensor_address = utils.get_address_from_tensor(ground_truth_tensor)
            op_type = dump_utils.get_op_type_from_file_name(my_output_dump_data.path)

            # 3. merge result
            tensor_info = {
                "tensor_id": tensor_id,
                "shape": my_output_tensor.shape,
                "op_type": op_type,
                "my_output_dtype": my_output_tensor_dtype,
                "ground_truth_dtype": ground_truth_tensor_dtype,
                "my_output_address": my_output_tensor_address,
                "ground_truth_address": ground_truth_tensor_address
            }
            tensor_result_list.append(
                compare_result.TensorResult(
                    tensor_info, [algorithm_result, overflow_result], error_msg, my_output_tensor.is_ffts))
        return tensor_result_list
