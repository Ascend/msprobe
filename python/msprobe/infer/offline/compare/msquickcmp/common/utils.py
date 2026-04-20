# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
Function:
This class mainly involves common function.
"""
import enum
import itertools
import os
import re
import shutil
import subprocess

import numpy as np

from msprobe.infer.offline.compare.msquickcmp.common.dynamic_argument_bean import DynamicArgumentEnum
from msprobe.core.common.log import logger

ACCURACY_COMPARISON_INVALID_PARAM_ERROR = 1
ACCURACY_COMPARISON_INVALID_DATA_ERROR = 2
ACCURACY_COMPARISON_INVALID_PATH_ERROR = 3
ACCURACY_COMPARISON_INVALID_COMMAND_ERROR = 4
ACCURACY_COMPARISON_PYTHON_VERSION_ERROR = 5
ACCURACY_COMPARISON_MODEL_TYPE_ERROR = 6
ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR = 7
ACCURACY_COMPARISON_WRITE_JSON_FILE_ERROR = 8
ACCURACY_COMPARISON_OPEN_FILE_ERROR = 9
ACCURACY_COMPARISON_BIN_FILE_ERROR = 10
ACCURACY_COMPARISON_INVALID_KEY_ERROR = 11
ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR = 12
ACCURACY_COMPARISON_TENSOR_TYPE_ERROR = 13
ACCURACY_COMPARISON_NO_DUMP_FILE_ERROR = 14
ACCURACY_COMPARISON_NOT_SUPPORT_ERROR = 15
ACCURACY_COMPARISON_NET_OUTPUT_ERROR = 16
ACCURACY_COMPARISON_INVALID_DEVICE_ERROR = 17
ACCURACY_COMPARISON_WRONG_AIPP_CONTENT = 18
ACCRACY_COMPARISON_EXTRACT_ERROR = 19
ACCRACY_COMPARISON_FETCH_DATA_ERROR = 20
ACCURACY_COMPARISON_ATC_RUN_ERROR = 21
ACCURACY_COMPARISON_INVALID_RIGHT_ERROR = 22
ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR = 23
ACCURACY_COMPARISON_EMPTY_CSV_ERROR = 24
MODEL_TYPE = ['.onnx', '.om']
DIM_PATTERN = r"^(-?[0-9]{1,100})(,-?[0-9]{1,100}){0,100}"
DYNAMIC_DIM_PATTERN = r"^([0-9-~]+)(,-?[0-9-~]+){0,3}"
MAX_DEVICE_ID = 255
SEMICOLON = ";"
COLON = ":"
EQUAL = "="
COMMA = ","
DOT = "."
ASCEND_BATCH_FIELD = "ascend_mbatch_batch_"
BATCH_SCENARIO_OP_NAME = "{0}_ascend_mbatch_batch_{1}"
INVALID_CHARS = ['|', ';', '&', '&&', '||', '>', '>>', '<', '`', '\\', '!', '\n']
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
DYM_SHAPE_END_MAX = 1000000
OPTYPE_WHITWLIST = ['Data', 'TransData', 'PartitionCall']


class AccuracyCompareException(Exception):
    """
    Class for Accuracy Compare Exception
    """

    def __init__(self, error_info):
        super(AccuracyCompareException, self).__init__()
        self.error_info = error_info


class InputShapeError(enum.Enum):
    """
    Class for Input Shape Error
    """

    FORMAT_NOT_MATCH = 0
    VALUE_TYPE_NOT_MATCH = 1
    NAME_NOT_MATCH = 2
    TOO_LONG_PARAMS = 3


def check_exec_cmd(command: str):
    if command.startswith("bash") or command.startswith("python"):
        cmds = command.split()
        if len(cmds) < 2:
            logger.error("Num of command elements is invalid.")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_COMMAND_ERROR)
        elif len(cmds) == 2:
            script_file = cmds[1]
            check_exec_script_file(script_file)
        else:
            script_file = cmds[1]
            check_exec_script_file(script_file)
            args = cmds[2:]
            check_input_args(args)
        return True

    else:
        logger.error("Command is not started with bash or python.")
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_COMMAND_ERROR)


def check_exec_script_file(script_path: str):
    if not os.path.exists(script_path):
        logger.error(f"File {script_path} is not exist.")
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    if not os.access(script_path, os.X_OK):
        logger.error(f"Script {script_path} don't has X authority.")
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_RIGHT_ERROR)


def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """

    if isdir:
        if not os.path.isdir(path):
            logger.error(f"The path {path} is not a directory. Please check the path")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        if not os.access(path, os.W_OK):
            logger.error(f"The path {path} does not have permission to write. Please check the path permission")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            logger.error(f"The path {path} is not a file.Please check the path")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        if not os.access(path, os.R_OK):
            logger.error(f"The path {path} does not have permission to read.Please check the path permission")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)


def check_input_bin_file_path(input_path):
    """
    Function Description:
        check the output bin file
    Parameter:
        input_path: input path directory
    """
    input_bin_files = input_path.split(',')
    bin_file_path_array = []
    for input_item in input_bin_files:
        input_item_path = os.path.realpath(input_item)
        if input_item_path.endswith('.bin'):
            check_file_or_directory_path(input_item_path)
            bin_file_path_array.append(input_item_path)
        else:
            check_file_or_directory_path(input_item_path, True)
            get_input_path(input_item_path, bin_file_path_array)
    return bin_file_path_array


def check_file_size_valid(file_path, size_max):
    if os.stat(file_path).st_size > size_max:
        logger.error(f'file_path={file_path} is too large, > {size_max}, not valid.')
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DATA_ERROR)


def check_input_args(args: list):
    for arg in args:
        if arg in INVALID_CHARS:
            logger.error("Args has invalid character.")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def check_convert_is_valid_used(dump, bin2npy, custom_op):
    """
    check dump is True while using convert
    """
    if not dump and (bin2npy or custom_op != ""):
        logger.error(
            "Convert option or custom_op is forbidden when dump is False!\
            Please keep dump True while using convert."
        )
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_COMMAND_ERROR)


def check_locat_is_valid(dump, locat):
    """
    Function:
        check locat args is completed
    Return:
        True or False
    """
    if locat and not dump:
        logger.error("Dump must be True when locat is used")
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_COMMAND_ERROR)


def check_device_param_valid(device):
    """
    check device param valid.
    """
    if not device.isdigit() or int(device) > MAX_DEVICE_ID:
        logger.error(
            "Please enter a valid number for device, the device id should be" " in [0, 255], now is %s." % device
        )
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DEVICE_ERROR)


def check_dynamic_shape(shape):
    """
    Function Description:
        check dynamic shpae
    Parameter:
        shape:shape
    Return Value:
        False or True
    """
    dynamic_shape = False
    for item in shape:
        if item is None or isinstance(item, str):
            dynamic_shape = True
            break
    return dynamic_shape


def _check_colon_exist(input_shape):
    if ":" not in input_shape:
        logger.error(get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, input_shape))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def _check_content_split_length(content_split):
    if not content_split[1]:
        logger.error(get_shape_not_match_message(InputShapeError.VALUE_TYPE_NOT_MATCH, content_split[1]))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def _check_shape_number(input_shape_value, pattern=DIM_PATTERN):
    dim_pattern = re.compile(pattern)
    match = dim_pattern.match(input_shape_value)
    if not match or match.group() is not input_shape_value:
        logger.error(get_shape_not_match_message(InputShapeError.VALUE_TYPE_NOT_MATCH, input_shape_value))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def check_input_name_in_model(tensor_name_list, input_name):
    """
    Function Description:
        check input name in model
    Parameter:
        tensor_name_list: the tensor name list
        input_name: the input name
    Exception Description:
        When input name not in tensor name list throw exception
    """
    if input_name not in tensor_name_list:
        logger.error(get_shape_not_match_message(InputShapeError.NAME_NOT_MATCH, input_name))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def check_max_size_param_valid(max_cmp_size):
    """
    check max_size param valid.
    """
    if max_cmp_size < 0:
        logger.error(
            "Please enter a valid number for max_cmp_size, the max_cmp_size should be"
            " in [0, ∞), now is %s." % max_cmp_size
        )
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DEVICE_ERROR)


def get_model_name_and_extension(offline_model_path):
    """
    Function Description:
        obtain the name and extension of the model file
    Parameter:
        offline_model_path: offline model path
    Return Value:
        model_name,extension
    Exception Description:
        when invalid data throw exception
    """
    file_name = os.path.basename(offline_model_path)
    model_name, extension = os.path.splitext(file_name)
    if extension not in MODEL_TYPE:
        logger.error(f"Model file {offline_model_path} suffix not valid, supported ones are {MODEL_TYPE}")
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
    return model_name, extension


def get_input_path(input_item_path, bin_file_path_array):
    for root, _, files in os.walk(input_item_path):
        for bin_file in files:
            if bin_file.endswith('.bin'):
                file_path = os.path.join(root, bin_file)
                bin_file_path_array.append(file_path)


def get_dump_data_path(dump_dir, is_net_output=False, model_name=None):
    """
    Function Description:
        traverse directories and obtain the absolute path of dump data
    Parameter:
        dump_dir: dump data directory
    Return Value:
        dump data path,file is exist or file is not exist
    """
    dump_data_path = None
    file_is_exist = False
    dump_data_dir_list = []
    for i in os.listdir(dump_dir):
        if not (os.path.isdir(os.path.join(dump_dir, i))):
            continue
        # net_output dump file directory, name is like 12_423_246_4352
        if is_net_output:
            if not i.isdigit():
                dump_data_dir = os.path.join(dump_dir, i)
                dump_data_dir_list.append(dump_data_dir)
                break
        # Contains the dump file directory, whose name is a pure digital timestamp
        elif i.isdigit():
            dump_data_dir = os.path.join(dump_dir, i)
            dump_data_dir_list.append(dump_data_dir)

    if not dump_data_dir_list:
        logger.error(f"The directory \"{dump_dir}\" does not contain dump data")
        raise AccuracyCompareException(ACCURACY_COMPARISON_NO_DUMP_FILE_ERROR)

    dump_data_path_list = []
    for candidate_dump_data_dir in dump_data_dir_list:
        for dir_path, _, files in os.walk(candidate_dump_data_dir):
            if files and not any(file.startswith("aclnn") for file in files):
                dump_data_path_list.append(dir_path)
                file_is_exist = True

    if len(dump_data_path_list) > 1:
        # find the model name directory
        dump_data_path = dump_data_path_list[0]
        for ii in dump_data_path_list:
            if model_name in ii:
                dump_data_path = ii
                break

        # move all dump files to single directory
        for ii in dump_data_path_list:
            if ii == dump_data_path:
                continue
            for file in os.listdir(ii):
                shutil.move(os.path.join(ii, file), dump_data_path)

    elif len(dump_data_path_list) == 1:
        dump_data_path = dump_data_path_list[0]
    else:
        dump_data_path = None

    return dump_data_path, file_is_exist


def get_shape_to_directory_name(input_shape):
    shape_info = re.sub(r"[:;]", "-", input_shape)
    shape_info = re.sub(r",", "_", shape_info)
    return shape_info


def get_shape_not_match_message(shape_error_type, value):
    """
    Function Description:
        get shape not match message
    Parameter:
        input:the value
        shape_error_type: the shape error type
    Return Value:
        not match message
    """
    message = ""
    if shape_error_type == InputShapeError.FORMAT_NOT_MATCH:
        message = (
            "Input shape \"{}\" format mismatch,the format like: "
            "input_name1:1,224,224,3;input_name2:3,300".format(value)
        )
    if shape_error_type == InputShapeError.VALUE_TYPE_NOT_MATCH:
        message = "Input shape \"{}\" value not number".format(value)
    if shape_error_type == InputShapeError.NAME_NOT_MATCH:
        message = "Input tensor name \"{}\" not in model".format(value)
    if shape_error_type == InputShapeError.TOO_LONG_PARAMS:
        message = "Input \"{}\" value too long".format(value)
    return message


def get_batch_index(dump_data_path):
    for _, _, files in os.walk(dump_data_path):
        for file_name in files:
            if ASCEND_BATCH_FIELD in file_name:
                return get_batch_index_from_name(file_name)
    return ""


def get_mbatch_op_name(om_parser, op_name, npu_dump_data_path):
    _, scenario = om_parser.get_dynamic_scenario_info()
    if scenario in [DynamicArgumentEnum.DYM_BATCH, DynamicArgumentEnum.DYM_DIMS]:
        batch_index = get_batch_index(npu_dump_data_path)
        current_op_name = BATCH_SCENARIO_OP_NAME.format(op_name, batch_index)
    else:
        return op_name
    return current_op_name


def get_batch_index_from_name(name):
    batch_index = ""
    last_batch_field_index = name.rfind(ASCEND_BATCH_FIELD)
    pos = last_batch_field_index + len(ASCEND_BATCH_FIELD)
    while pos < len(name) and name[pos].isdigit():
        batch_index += name[pos]
        pos += 1
    return batch_index


def get_data_len_by_shape(shape):
    data_len = 1
    for item in shape:
        if item == -1:
            logger.warning("please check your input shape, one dim in shape is -1.")
            return -1
        data_len = data_len * item
    return data_len


def parse_input_shape(input_shape):
    """
    Function Description:
        parse input shape
    Parameter:
        input_shape:the input shape,this format like:tensor_name1:dim1,dim2;tensor_name2:dim1,dim2
    Return Value:
        the map type of input_shapes
    """
    input_shapes = {}
    if input_shape == '':
        return input_shapes
    _check_colon_exist(input_shape)
    tensor_list = input_shape.split(';')
    for tensor in tensor_list:
        _check_colon_exist(input_shape)
        tensor_shape_list = tensor.rsplit(':', maxsplit=1)
        if len(tensor_shape_list) == 2:
            shape = tensor_shape_list[1]
            input_shapes[tensor_shape_list[0]] = shape.split(',')
            _check_shape_number(shape)
        else:
            logger.error(get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, input_shape))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return input_shapes


def parse_input_shape_to_list(input_shape):
    """
    Function Description:
        parse input shape, and get a list only contains inputs shape and a list contains inputs tensor name
    Parameter:
        input_shape:the input shape,this format like:tensor_name1:dim1,dim2;tensor_name2:dim1,dim2.
    Return Value:
        a list only contains inputs shape, this format like [[dim1,dim2],[dim1,dim2]] , and a list contains tensor name
    """
    input_shape_list = []
    name_list = []
    if not input_shape:
        return input_shape_list
    _check_colon_exist(input_shape)
    tensor_list = input_shape.split(';')

    for tensor in tensor_list:
        tensor_shape_list = tensor.rsplit(':', maxsplit=1)
        if len(tensor_shape_list) == 2:
            shape_list_int = []
            for dim in tensor_shape_list[1].split(','):
                if dim.isdigit():
                    shape_list_int.append(int(dim))
                else:
                    raise ValueError("The input of --input_shape parameter is unreasonable, " \
                                     "because the tensor shape is not digit.")
            for dim_int in shape_list_int:
                if dim_int < 0:
                    raise ValueError("The input of --input_shape parameter is unreasonable, " \
                                     "possibly because the upper bound is smaller than 0.")
                prompt = "The --input_shape %r is larger than expected. " \
                         "Attempting to input such a shape could potentially impact system performance.\n" \
                         "Please confirm your awareness of the risks associated with this action ([y]/n): " % tensor
                if dim_int > DYM_SHAPE_END_MAX and not dym_shape_range_interaction(prompt):
                    raise ValueError("The dim of --input_shape %r is too large." % (str(dim_int)))

            input_shape_list.append(shape_list_int)
            name_list.append(tensor_shape_list[0])
        else:
            logger.error(get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, input_shape))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return input_shape_list, name_list


def dym_shape_range_interaction(prompt):
    confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)

    try:
        user_action = input(prompt)
    except Exception:
        return False

    return bool(confirm_pattern.match(user_action))


def parse_dym_shape_range(dym_shape_range):
    """
    Function Description:
        parse dynamic input shape
    Parameter:
        dym_shape_range:the input shape,this format like:tensor_name1:dim1,dim2-dim3;tensor_name2:dim1,dim2~dim3.
         - means the both dim2 and dim3 value, ~ means the range of [dim2:dim3]
    Return Value:
        a list only contains inputs shape, this format like [[dim1,dim2],[dim1,dim2]]
    """
    _check_colon_exist(dym_shape_range)
    input_shapes = {}
    tensor_list = dym_shape_range.split(";")
    info_list = []

    for tensor in tensor_list:
        _check_colon_exist(dym_shape_range)
        shapes = []
        name, shapestr = tensor.split(":")
        if len(shapestr) < 50:
            _check_shape_number(shapestr, DYNAMIC_DIM_PATTERN)
        else:
            logger.error(get_shape_not_match_message(InputShapeError.TOO_LONG_PARAMS, input_shapes))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        for content in shapestr.split(","):
            if "~" in content:
                content_split = content.split("~")
                _check_content_split_length(content_split)
                start_str = content_split[0]
                end_str = content_split[1]
                step_str = content_split[2] if len(content_split) == 3 else "1"
                if not start_str.isdigit() or not end_str.isdigit() or not step_str.isdigit():
                    raise ValueError(f"--dym-shape parameter should be digit.")
                start = int(start_str)
                end = int(end_str)
                step = int(step_str)
                if start > end or start < 0:
                    raise ValueError("The input of --dym-shape parameter is unreasonable, " \
                                     "possibly because the upper bound of the shape is greater than the lower bound" \
                                     "or the upper bound is smaller than 0.")
                if step <= 0:
                    raise ValueError(f"Step in --dym-shape parameter should be greater than 0, now is {step}.")
                prompt = "The --dym-shape-range %r is larger than expected. " \
                         "Attempting to input such a shape could potentially impact system performance.\n" \
                         "Please confirm your awareness of the risks associated with this action ([y]/n): " % content
                if (end - start) / step > DYM_SHAPE_END_MAX and not dym_shape_range_interaction(prompt):
                    raise ValueError("--dym-shape-range is too large, start: %r, end: %r, step: %r" % (str(start), \
                                                                                                       str(end),
                                                                                                       str(step)))
                ranges = [str(i) for i in range(start, end + 1, step)]
            elif "-" in content:
                ranges = content.split("-")
            else:
                start = int(content)
                ranges = [str(start)]
            shapes.append(ranges)
        shape_list = [",".join(s) for s in list(itertools.product(*shapes))]
        info = ["{}:{}".format(name, s) for s in shape_list]
        info_list.append(info)
    res = [";".join(s) for s in list(itertools.product(*info_list))]
    logger.info("shape_list:" + str(res))
    return res


def parse_arg_value(values):
    """
    parse dynamic arg value of atc cmdline
    """
    value_list = []
    for item in values.split(SEMICOLON):
        value_list.append(parse_value_by_comma(item))
    return value_list


def parse_value_by_comma(value):
    """
    parse value by comma, like '1,2,4,8'
    """
    value_list = []
    value_str_list = value.split(COMMA)
    for value_str in value_str_list:
        value_str = value_str.strip()
        if value_str.isdigit() or value_str == '-1':
            value_list.append(int(value_str))
        else:
            logger.error("please check your input shape.")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return value_list


def execute_command(cmd, info_need=True):
    """
    Function Description:
        run the following command
    Parameter:
        cmd: command
    Return Value:
        command output result
    Exception Description:
        when invalid command throw exception
    """
    if info_need:
        logger.info('Execute command:%s' % " ".join(cmd))
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ais_bench_logs = ""
    try:
        while process.poll() is None:
            line = process.stdout.readline()
            if line:  # 检查line是否为空，避免解码和追加是的潜在错误
                ais_bench_logs += line.decode()
    finally:
        process.stdout.close()
    if process.returncode != 0:
        logger.error('Failed to execute command:%s' % " ".join(cmd))
        logger.error(f'\nerror log:\n {ais_bench_logs}')
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DATA_ERROR)


def handle_ground_truth_files(om_parser, npu_dump_data_path, golden_dump_data_path):
    _, scenario = om_parser.get_dynamic_scenario_info()
    if scenario in [DynamicArgumentEnum.DYM_BATCH, DynamicArgumentEnum.DYM_DIMS]:
        batch_index = get_batch_index(npu_dump_data_path)
        for root, _, files in os.walk(golden_dump_data_path):
            for file_name in files:
                first_dot_index = file_name.find(DOT)
                if first_dot_index == -1:
                    logger.warning("file name in golden dump data path found it does not contain '.', skip copy.")
                    continue
                current_op_name = BATCH_SCENARIO_OP_NAME.format(file_name[:first_dot_index], batch_index)
                dst_file_name = current_op_name + file_name[first_dot_index:]
                shutil.copy(os.path.join(root, file_name), os.path.join(root, dst_file_name))


def load_npy_from_buffer(raw_data, dtype, shape):
    no_dump_data = None
    try:
        return np.frombuffer(raw_data, dtype=dtype).reshape(shape)
    except Exception as e:
        return no_dump_data


def find_om_files(dir_path):
    om_files = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.om'):
            om_files.append(os.path.join(dir_path, filename))
    return om_files
