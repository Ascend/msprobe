#!/usr/bin/env python
# coding=utf-8
"""
Function:
DumpDataParser class. This class mainly involves the parser_dump_data function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""
import os
import json
import struct
import numpy as np
import utils
import log
import common
from const_manager import ConstManager

from file_utils import FileUtils

from multi_convert_process import MultiConvertProcess
from compare_error import CompareError


class DumpDataParser:
    """
    The class for dump data parser
    """

    def __init__(self: any, arguments: any) -> None:
        self.path_str = arguments.dump_path
        self.input_path = []
        self.multi_process = None
        self.output_path = os.path.realpath(arguments.output_path)
        self.dump_version = arguments.dump_version
        self.output_file_type = arguments.output_file_type

    def check_arguments_valid(self: any) -> None:
        """
        Check arguments valid
        """
        self.input_path = utils.get_path_list_for_str(self.path_str)
        self.multi_process = MultiConvertProcess(self._parse_one_dump_file, self.input_path, self.output_path)
        ret = utils.check_output_path_valid(self.output_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

    def _save_tensor_to_file(self: any, dump_path: str, tensor_list: list, tensor_type: str, op_name: str) -> None:
        if len(tensor_list) == 0:
            log.print_warn_log('There is no %s in "%s".' % (tensor_type, dump_path))
            return
        name = os.path.basename(dump_path)
        for (index, tensor) in enumerate(tensor_list):
            log.print_info_log('Start to parse the data of %s:%d in "%s".' % (tensor_type, index, dump_path))
            try:
                array = utils.deserialize_dump_data_to_array(tensor)
            except CompareError:
                log.print_warn_log('Failed to parse the data of %s:%d in "%s".' % (tensor_type, index, dump_path))
                continue
            if self.output_file_type == 'npy':
                file_name = '%s.%s.%d.npy' % (name, tensor_type, index)
                file_name = FileUtils.handle_too_long_file_name(
                    file_name, '.npy', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
            elif self.output_file_type == 'msnpy':
                file_name = utils.make_msnpy_file_name(dump_path, op_name, tensor_type, index, tensor.format)
                file_name = FileUtils.handle_too_long_file_name(
                    file_name, '.npy', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
            else:
                file_name = '%s.%s.%d.%s.%s.%s.bin' \
                            % (name, tensor_type, index, utils.get_string_from_list(array.shape, '_'),
                               np.dtype(common.get_dtype_by_data_type(tensor.data_type)).name,
                               common.get_format_string(tensor.format))
                file_name = FileUtils.handle_too_long_file_name(
                    file_name, '.bin', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
            output_file_path = os.path.join(self.output_path, file_name)
            FileUtils.save_array_to_file(output_file_path, array, self.output_file_type != 'bin', tensor.shape.dim)
            log.print_info_log('The data of %s:%d has been parsed into "%s".'
                               % (tensor_type, index, output_file_path))

    def _save_buffer_to_file(self: any, dump_path: str, tensor_list: list) -> None:
        if len(tensor_list) == 0:
            log.print_warn_log('There is no buffer data in "%s".' % dump_path)
            return
        name = os.path.basename(dump_path)
        for (index, tensor) in enumerate(tensor_list):
            buffer_type = ConstManager.BUFFER_TYPE_MAP.get(tensor.buffer_type)
            log.print_info_log('Start to parse the data of %sbuffer:%d in "%s".'
                               % (buffer_type, index, dump_path))
            file_name = "%s.%sbuffer.%s.bin" % (name, buffer_type, index)
            file_name = FileUtils.handle_too_long_file_name(
                file_name, '.bin', os.path.join(self.output_path, ConstManager.MAPPING_FILE_NAME))
            output_dump_path = os.path.join(self.output_path, file_name)
            FileUtils.save_data_to_file(output_dump_path, tensor.data, 'wb', delete=True)
            log.print_info_log('The data of %sbuffer:%d has been parsed into "%s".'
                               % (buffer_type, index, output_dump_path))

    @staticmethod
    def _unpack_uint64_value(data: any, index: int) -> int:
        return struct.unpack(ConstManager.UINT64_FMT, data[index:index + ConstManager.UINT64_SIZE])[0]

    def _parser_overflow_info(self: any, data: any, start: int) -> dict:
        index = start
        model_id = self._unpack_uint64_value(data, index)
        index += ConstManager.UINT64_SIZE
        stream_id = self._unpack_uint64_value(data, index)
        index += ConstManager.UINT64_SIZE
        task_id = self._unpack_uint64_value(data, index)
        index += ConstManager.UINT64_SIZE
        task_type = self._unpack_uint64_value(data, index)
        index += ConstManager.UINT64_SIZE
        pc_start = self._unpack_uint64_value(data, index)
        index += ConstManager.UINT64_SIZE
        para_base = self._unpack_uint64_value(data, index)
        return {'model_id': model_id, 'stream_id': stream_id,
                'task_id': task_id, 'task_type': task_type,
                'pc_start': hex(pc_start), 'para_base': hex(para_base)}

    def _parser_ai_core_status(self: any, ai_core_info: dict, data: any, start: int) -> None:
        index = start
        kernel_code = self._unpack_uint64_value(data, index)
        ai_core_info['kernel_code'] = hex(kernel_code)
        index += ConstManager.UINT64_SIZE
        block_idx = self._unpack_uint64_value(data, index)
        ai_core_info['block_idx'] = block_idx
        index += ConstManager.UINT64_SIZE
        status = self._unpack_uint64_value(data, index)
        ai_core_info['status'] = status

    def _save_op_debug_to_file(self: any, dump_path: str, output: any) -> None:
        for idx, item in enumerate(output):
            if len(item.data) != ConstManager.OVERFLOW_CHECK_SIZE:
                log.print_error_log('The data size (%d) of output:%d is not equal to %d in %s. '
                                    'Please check the dump file.'
                                    % (len(item.data), idx, ConstManager.OVERFLOW_CHECK_SIZE, dump_path))
                raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
            # parser DHA Atomic Add info
            index = 0
            dha_atomic_add_info = self._parser_overflow_info(item.data, index)
            # parser L2 Atomic Add info
            index += ConstManager.DHA_ATOMIC_ADD_INFO_SIZE
            l2_atomic_add_info = self._parser_overflow_info(item.data, index)
            # parser AI Core info
            index += ConstManager.L2_ATOMIC_ADD_INFO_SIZE
            ai_core_info = self._parser_overflow_info(item.data, index)
            # parser DHA Atomic Add status
            index += ConstManager.AI_CORE_INFO_SIZE
            dha_atomic_add_status = self._unpack_uint64_value(item.data, index)
            dha_atomic_add_info['status'] = dha_atomic_add_status
            # parser L2 Atomic Add status
            index += ConstManager.DHA_ATOMIC_ADD_STATUS_SIZE
            l2_atomic_add_status = self._unpack_uint64_value(item.data, index)
            l2_atomic_add_info['status'] = l2_atomic_add_status
            # parser AI Core status
            index += ConstManager.L2_ATOMIC_ADD_STATUS_SIZE
            self._parser_ai_core_status(ai_core_info, item.data, index)

            data = {'DHA Atomic Add': dha_atomic_add_info,
                    'L2 Atomic Add': l2_atomic_add_info,
                    'AI Core': ai_core_info}
            json_path = os.path.join(self.output_path, "%s.output.%d.json" % (os.path.basename(dump_path), idx))
            FileUtils.save_data_to_file(json_path, json.dumps(data, sort_keys=False, indent=4), 'w+', delete=True)
            log.print_info_log('The data of output:%d has been parsed into "%s".' % (idx, json_path))

    def _parse_one_file_exec(self: any, dump_path: str) -> None:
        ret = utils.check_path_valid(dump_path, True, False, path_type=utils.PathType.File)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        dump_data = utils.parse_dump_file(dump_path, self.dump_version)
        if os.path.basename(dump_path).startswith('Opdebug.Node_OpDebug.'):
            self._save_op_debug_to_file(dump_path, dump_data.output)
        else:
            self._save_tensor_to_file(dump_path, dump_data.input, 'input', dump_data.op_name)
            self._save_tensor_to_file(dump_path, dump_data.output, 'output', dump_data.op_name)
            self._save_buffer_to_file(dump_path, dump_data.buffer)

    def _parse_one_dump_file(self: any, dump_path: str) -> (int, str):
        try:
            self._parse_one_file_exec(dump_path)
        except CompareError as error:
            return error.code, dump_path
        return CompareError.MSACCUCMP_NONE_ERROR, dump_path

    def parse_dump_data(self: any) -> int:
        """
        Convert dump data to numpy and bin file
        """
        # 1. check arguments valid
        self.check_arguments_valid()
        # 2. parse dump data
        if len(self.input_path) == 1 and os.path.isfile(self.input_path[0]):
            ret, _ = self._parse_one_dump_file(self.input_path[0])
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                log.print_error_log('Failed to parse dump file "%s".' % self.input_path[0])
            return ret
        return self.multi_process.process()
