
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the dump function.
"""
import re
import os
from enum import Enum
from enum import unique

from cmp_utils import utils, utils_type, path_check
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.reg_manager import RegManager
from cmp_utils.constant.compare_error import CompareError
from dump_parse.ffts_parser import FFTSParser
from dump_parse import dump_utils, mapping
from dump_parse.dump_data_object import DumpDataObj, DumpTensor


@unique
class DumpType(Enum):
    """
    The enum for dump type
    """
    Offline = 0
    Standard = 1
    Quant = 2
    Numpy = 3


class DumpInfo:
    """
    The class for dump info
    """

    def __init__(self: any, dump_path: str, dump_version: int, ffts: bool = False,
                 fusion_json_file_path: str = "") -> None:
        self.path = dump_path
        self.type = None
        self.op_name_to_file_map = {}
        self.op_name_to_task_mode_map = {}
        self.quant = False
        self.data_info = ''
        self.dump_version = dump_version
        self.hash_to_file_name_map = {}
        self.ffts = ffts
        self.fusion_json_file_path = fusion_json_file_path

    def check_arguments_valid(self: any) -> None:
        """
        Check arguments valid, if invalid, throw exception
        """
        ret = path_check.check_path_valid(self.path, True, False, path_check.PathType.Directory)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)
        self._make_op_name_to_file_map()

    def is_standard_quant(self: any) -> bool:
        """
        The dump data is standard quant
        :return: bool, true if dump data is standard quant
        """
        return self.type == DumpType.Quant or (self.type == DumpType.Numpy and self.quant)

    def is_standard_origin(self: any) -> bool:
        """
        The dump data is standard origin
        :return: bool, true if dump data is standard origin
        """
        return self.type == DumpType.Standard or (self.type == DumpType.Numpy and not self.quant)

    def get_op_dump_file(self: any, op_name: str, output_index: int = None, print_log: bool = True) -> tuple:
        """
        Get the dump file a by op name
        :param op_name: the op name
        :param output_index: the output index
        :param print_log: print log or not
        :return: the dump file path
        """
        original_op_name = op_name.replace('/', '_').replace('.', '_')
        if output_index is not None:
            original_op_name = '%s.%d' % (original_op_name, output_index)
        if original_op_name not in self.op_name_to_file_map:
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        dump_file_list = self.op_name_to_file_map.get(original_op_name)
        dump_mode = self.op_name_to_task_mode_map.get(original_op_name)
        if not dump_file_list:
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        if len(dump_file_list) > 1:
            dump_file_list = dump_utils.sort_dump_file_list(dump_mode, dump_file_list)
        if print_log:
            log.print_info_log('[%s] [%s] %s' % (op_name, str(self.type.name), dump_file_list))
        return dump_file_list, dump_mode

    def get_op_dump_data(self: any, op_name: str, output_index: int = None) -> (str, DumpDataObj):
        """
        Get the dump file and data by op name
        :param op_name: the op name
        :param output_index: the output index
        :return: the dump file path, dump data
        """
        if self.type == DumpType.Quant and output_index is None:
            output_index = 0
        dump_file_list, dump_mode = self.get_op_dump_file(op_name, output_index)
        dump_data_list = [dump_utils.parse_dump_file(dump_file_path,
                                                     self.dump_version) for dump_file_path in dump_file_list]
        if dump_mode == ConstManager.AUTOMATIC_MODE or dump_mode == ConstManager.MANUAL_MODE:
            ffts_parser = FFTSParser(dump_file_list, dump_data_list)
            dump_file_path, dump_data = ffts_parser.parse_ffts
        else:
            dump_file_path = dump_file_list[-1]
            dump_data = dump_data_list[-1]
        return dump_file_path, dump_data

    def get_data_info(self: any) -> str:
        """
        Get the dump data info
        :return: dump data info
        """
        if self.data_info == '':
            self.data_info = 'side is dump data of the '
            prefix = 'quantized' if self.quant else 'unquantized'
            if self.type == DumpType.Offline:
                self.data_info += prefix + ' model executed on the AI processor'
            elif self.type == DumpType.Standard:
                self.data_info += 'unquantized original model'
            elif self.type == DumpType.Quant:
                self.data_info += 'quantized original model'
        return self.data_info

    def _match_dump_pattern(self: any, pattern: str, item: str, info: str, expect: DumpType) -> (any, DumpType):
        _, match = RegManager.match_group(pattern, item)
        if match is None:
            if item in self.hash_to_file_name_map.values():
                msg = 'The file name \"{}\" in the file \"{}\" is invalid. It only supports {}' \
                    .format(item, os.path.join(self.path, ConstManager.MAPPING_FILE_NAME), info)
            else:
                msg = 'The file name \"{}\" in the path \"{}\"is invalid, only supports {}' \
                    .format(item, self.path, info)
            log.print_warn_log(msg)
            raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        return match.group(1), expect

    def _check_dump_file_is_quant(self: any, dump_type: DumpType, op_name: str) -> None:
        if dump_type in [DumpType.Offline, DumpType.Numpy]:
            suffix_list = ConstManager.QUANT_OP_NANE_SUFFIX_LIST + ConstManager.DEQUANT_OP_NANE_SUFFIX_LIST
            for suffix in suffix_list:
                if suffix in op_name:
                    self.quant = True

    def _check_file_match_pattern(self: any, item: str) -> (any, DumpType):
        if item.endswith(ConstManager.NUMPY_SUFFIX):
            return self._match_dump_pattern(
                RegManager.NUMPY_DUMP_PATTERN, item, ConstManager.NUMPY_FILE_NAME, DumpType.Numpy)
        if item.endswith(ConstManager.STANDARD_SUFFIX):
            return self._match_dump_pattern(
                RegManager.STANDARD_DUMP_PATTERN, item, ConstManager.STANDARD_FILE_NAME, DumpType.Standard)
        if item.endswith(ConstManager.QUANT_SUFFIX):
            return self._match_dump_pattern(
                RegManager.QUANT_DUMP_PATTERN, item, ConstManager.QUANT_FILE_NAME, DumpType.Quant)
        return self._match_dump_pattern(
            RegManager.OFFLINE_DUMP_PATTERN, item, ConstManager.OFFLINE_FILE_NAME, DumpType.Offline)

    def _handle_one_file(self: any, file_path: str) -> None:
        item = os.path.basename(file_path)
        item_name, item_extension = os.path.splitext(item)
        if item.isdigit() or (item_name.isdigit() and item_extension == ".npy"):
            mapping_file_path = os.path.join(self.path, ConstManager.MAPPING_FILE_NAME)
            if not os.path.exists(mapping_file_path):
                log.print_warn_log('The file name \"{}\" corresponding mapping file \"{}\" is not exist.'
                                   .format(item, mapping_file_path))
                return
            if self.hash_to_file_name_map and not self.hash_to_file_name_map.get(item):
                log.print_warn_log('The file name \"{}\" in the file \"{}\" is not exist.'
                                   .format(item, mapping_file_path))
                return
            item = self.hash_to_file_name_map.get(item)
        try:
            match, current_dump_type = self._check_file_match_pattern(item)
        except CompareError:
            return
        finally:
            pass
        if not self.ffts:
            op_name = handle_op_name(match, self.fusion_json_file_path)
        # if real op name contain '_lxslice' field, the op will not be added to map
            if ConstManager.FFTS_MANUAL_MODE_FIELD in op_name:
                return
            self._check_task_type(op_name, item)
        else:
            op_name = match

        self._check_task_type(op_name, item)
        self._check_dump_file_is_quant(current_dump_type, op_name)
        if self.type is None:
            self.type = current_dump_type
        else:
            if self.type != current_dump_type:
                log.print_error_log('Not all files in the path "%r" are of the same type, such as "%s".'
                                    % (self.path, item))
                raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        if op_name in self.op_name_to_file_map:
            self.op_name_to_file_map.get(op_name).append(file_path)
        else:
            self.op_name_to_file_map[op_name] = [file_path]

    def _make_op_name_to_file_map(self: any) -> None:
        mapping_file_path = os.path.join(self.path, ConstManager.MAPPING_FILE_NAME)
        self.hash_to_file_name_map = mapping.read_mapping_file(mapping_file_path)
        for item in os.listdir(self.path):
            if item in (ConstManager.MAPPING_FILE_NAME, ConstManager.CONVERT_FAILED_FILE_LIST_NAME):
                continue
            file_path = os.path.join(self.path, item)
            if os.path.isfile(file_path):
                self._handle_one_file(file_path)
        self._judge_dump_type()

    def _check_task_type(self: any, op_name: str, file_name: str) -> None:
        if file_name.endswith(ConstManager.NUMPY_SUFFIX):
            self.op_name_to_task_mode_map[op_name] = ConstManager.NORMAL_MODE
            return
        flied_list = file_name.split(".")
        if len(flied_list) <= ConstManager.OLD_FILE_FIELD_NUM + 1:
            self.op_name_to_task_mode_map[op_name] = ConstManager.NORMAL_MODE
            return
        if flied_list[-4] not in ConstManager.TASK_TYPE_MAP.values():
            raise CompareError(CompareError.MSACCUCMP_INVALID_TASK_TYPE)
        if flied_list[-4] == ConstManager.TASK_TYPE_MAP.get(ConstManager.FFTSPLUS):
            if ConstManager.FFTS_MANUAL_MODE_FIELD in file_name:
                self.op_name_to_task_mode_map[op_name] = ConstManager.MANUAL_MODE
            else:
                self.op_name_to_task_mode_map[op_name] = ConstManager.AUTOMATIC_MODE
            return
        self.op_name_to_task_mode_map[op_name] = ConstManager.NORMAL_MODE

    def _judge_dump_type(self: any) -> None:
        if self.type is None:
            log.print_error_log('No valid dump files in the path "%r".' % self.path)
            raise CompareError(CompareError.MSACCUCMP_DUMP_FILE_ERROR)
        if self.type == DumpType.Numpy:
            if self.quant:
                self.type = DumpType.Quant
            else:
                self.type = DumpType.Standard


class CompareData:
    """
    The class for compare data, left dump data and right dump data
    """

    def __init__(self: any, left_dump_path: str, right_dump_path: str, dump_version: int,
                 ffts: bool = False, fusion_json_file_path: str = "") -> None:
        self.left_dump_info = DumpInfo(left_dump_path, dump_version, ffts, fusion_json_file_path)
        self.right_dump_info = DumpInfo(right_dump_path, dump_version, ffts, fusion_json_file_path)
        self.dump_version = dump_version

    def check_arguments_valid(self: any, fusion_json_file_path: str,
                              quant_fusion_rule_file_path: str,
                              close_fusion_rule_file_path: str) -> None:
        """
        Check arguments valid, if invalid, throw exception
        :param fusion_json_file_path: the fusion json file path
        :param quant_fusion_rule_file_path: the quant fusion rule file path
        :param close_fusion_rule_file_path: the clsoe fusion json file path
        """
        self.left_dump_info.check_arguments_valid()
        self.right_dump_info.check_arguments_valid()
        left_type = self.left_dump_info.type
        info = 'When the left %s and the right %s,' \
               % (self.left_dump_info.get_data_info(), self.right_dump_info.get_data_info())
        if left_type == DumpType.Offline:
            self._check_left_type_offline_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                                close_fusion_rule_file_path)
        elif left_type == DumpType.Quant:
            self._check_left_type_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                              close_fusion_rule_file_path)
        elif left_type == DumpType.Standard:
            log.print_error_log('%s this scenario cannot be compared.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)

    def get_left_dump_data(self: any, op_name: str) -> (str, DumpDataObj):
        """
        Get the left dump file path and data by fusion op
        :param op_name: the op name
        :return: the left dump file path, the left dump data
        """
        return self.left_dump_info.get_op_dump_data(op_name)

    def get_right_dump_data(self: any, op_name: str, output_index: int = 0) -> (str, DumpDataObj):
        """
        Get the right dump file path and data by fusion op
        :param op_name: the op name
        :param output_index: the output index
        :return: the right dump file path, the right dump data
        """
        if self.right_dump_info.type == DumpType.Offline:
            return self.right_dump_info.get_op_dump_data(op_name)
        return self.right_dump_info.get_op_dump_data(op_name, output_index)

    def is_standard_quant_vs_origin(self: any) -> bool:
        """
        The scenario is quantized original vs unquantized original
        :return: bool, true if the scenario is quantized original vs unquantized original
        """
        return self.left_dump_info.is_standard_quant() and self.right_dump_info.is_standard_origin()

    def _check_offline_standard_valid(self: any, info: str, fusion_json_file_path: str,
                                      quant_fusion_rule_file_path: str, close_fusion_rule_file_path: str) -> None:
        if fusion_json_file_path == "":
            log.print_error_log('%s the -f parameter is required.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        if close_fusion_rule_file_path != '':
            log.print_error_log('%s there is no need to enter the -cf parameter.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        if self.left_dump_info.quant and quant_fusion_rule_file_path == '':
            log.print_error_log('%s the -q parameter is required.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        if not self.left_dump_info.quant and quant_fusion_rule_file_path != '':
            log.print_error_log('%s there is no need to enter the -q parameter.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def _check_offline_quant_valid(self: any, info: str, fusion_json_file_path: str,
                                   quant_fusion_rule_file_path: str, close_fusion_rule_file_path: str) -> None:
        if not self.left_dump_info.quant:
            log.print_error_log('%s this scenario cannot be compared.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)
        if fusion_json_file_path == "":
            log.print_error_log('%s the -f parameter is required.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        if quant_fusion_rule_file_path != '' or close_fusion_rule_file_path != '':
            log.print_error_log('%s there is no need to enter the -q or -cf parameter.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def _check_left_type_offline_valid(self: any, info: str, fusion_json_file_path: str,
                                       quant_fusion_rule_file_path: str, close_fusion_rule_file_path: str) -> None:
        right_type = self.right_dump_info.type
        if right_type == DumpType.Offline:
            if self.left_dump_info.quant != self.right_dump_info.quant:
                log.print_error_log('%s this scenario cannot be compared.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)
            if fusion_json_file_path != '' and close_fusion_rule_file_path == '':
                log.print_error_log('%s the -cf parameter is required.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            if fusion_json_file_path == '' and close_fusion_rule_file_path != '':
                log.print_error_log('%s the -f parameter is required.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
            if quant_fusion_rule_file_path != '':
                log.print_error_log('%s there is no need to enter the -q parameter.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)
        elif right_type == DumpType.Standard:
            self._check_offline_standard_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                               close_fusion_rule_file_path)
        elif right_type == DumpType.Quant:
            self._check_offline_quant_valid(info, fusion_json_file_path, quant_fusion_rule_file_path,
                                            close_fusion_rule_file_path)

    def _check_left_type_quant_valid(self: any, info: str, fusion_json_file_path: str,
                                     quant_fusion_rule_file_path: str, close_fusion_rule_file_path: str) -> None:
        right_type = self.right_dump_info.type
        if close_fusion_rule_file_path != "":
            log.print_error_log('%s there is no need to enter the -cf parameter.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)
        if right_type in (DumpType.Offline, DumpType.Quant):
            log.print_error_log('%s this scenario cannot be compared.' % info)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)
        elif right_type == DumpType.Standard:
            if quant_fusion_rule_file_path == '':
                log.print_error_log('%s the -q parameter is required.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)
            if fusion_json_file_path != "":
                log.print_error_log('%s there is no need to enter the -f parameter.' % info)
                raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_TYPE_ERROR)


def handle_op_name(file_op_name: str, fusion_json_file_path: str) -> str:
    if fusion_json_file_path and ConstManager.FFTS_MANUAL_MODE_FIELD in file_op_name:
        file_op_name = process_op_name(file_op_name)
        return file_op_name

    # filter field '_lxsliceX' and '_sgt_field'
    if ConstManager.FFTS_MANUAL_MODE_FIELD not in file_op_name \
            and ConstManager.SGT_FIELD not in file_op_name:
        return file_op_name

    # field '_lxsliceX' at the end of name
    if ConstManager.FFTS_MANUAL_MODE_FIELD in file_op_name:
        first_match = RegManager.get_matchs(
            RegManager.FFTS_MANUAL_FIELD_PATTERN, file_op_name)[0]
        file_op_name = \
            file_op_name[:first_match.start() - 1] if first_match.end() == first_match.endpos else file_op_name

    # filter field '_sgt_field'
    if ConstManager.SGT_FIELD in file_op_name:
        # field '_sgt_graph' in the name
        end_match = RegManager.get_matchs(
            RegManager.SGT_FLIED_PATTERN, file_op_name)[-1]
        file_op_name = file_op_name[end_match.end() + 1:] if end_match.end() != end_match.endpos else file_op_name
    return file_op_name


def process_op_name(name):
    re_pattern = re.compile(RegManager.LXSLICE_PATTERN)
    op_name = re_pattern.sub("", name)
    return op_name

