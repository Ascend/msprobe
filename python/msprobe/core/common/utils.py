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

import collections
import functools
import inspect
import os
import re
import threading
import time
import importlib.util
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import (FileOpen, check_file_or_directory_path, load_json, load_construct_json)
from msprobe.core.common.log import logger

_has_torchrec = False
try:
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
    _has_torchrec = True
except Exception as e:
    logger.warning(
        f"torchrec is not available or incompatible: {e}, skip dump JaggedTensor, KeyedJaggedTensor, KeyedTensor."
    )
    _has_torchrec = False

device = collections.namedtuple('device', ['type', 'index'])
prefixes = ['api_stack', 'list', 'range', 'acl']
file_suffix_to_file_type = {
    "dump.json": Const.DUMP_JSON_FILE,
    "debug.json": Const.DEBUG_JSON_FILE,
}


class MsprobeBaseException(Exception):
    """
    Base class for all custom exceptions.
    """
    # 所有的错误代码
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    OPEN_FILE_ERROR = 2
    CLOSE_FILE_ERROR = 3
    READ_FILE_ERROR = 4
    WRITE_FILE_ERROR = 5
    INVALID_FILE_ERROR = 6
    PERMISSION_ERROR = 7
    INDEX_OUT_OF_BOUNDS_ERROR = 8
    NO_DUMP_FILE_ERROR = 9
    INVALID_DATA_ERROR = 10
    INVALID_PARAM_ERROR = 11
    INVALID_DUMP_RATIO = 12
    INVALID_DUMP_FILE = 13
    UNKNOWN_ERROR = 14
    INVALID_DUMP_MODE = 15
    PARSE_FILE_ERROR = 16
    INVALID_COMPARE_MODE = 17
    OVER_SIZE_FILE_ERROR = 18
    INVALID_SUMMARY_MODE = 19
    INVALID_TASK_ERROR = 20
    DETACH_ERROR = 21
    INVALID_OBJECT_TYPE_ERROR = 22
    INVALID_CHAR_ERROR = 23
    RECURSION_LIMIT_ERROR = 24
    INVALID_ATTRIBUTE_ERROR = 25
    OUTPUT_HOOK_ERROR = 26
    INPUT_HOOK_ERROR = 27
    FUNCTION_CALL_ERROR = 28
    FORWARD_DATA_COLLECTION_ERROR = 29
    BACKWARD_DATA_COLLECTION_ERROR = 30
    INVALID_KEY_ERROR = 31
    MISSING_HEADER_ERROR = 32
    MERGE_COMPARE_RESULT_ERROR = 33
    NAMES_STRUCTS_MATCH_ERROR = 34
    INVALID_STATE_ERROR = 35
    INVALID_API_NAME_ERROR = 36
    CROSS_FRAME_ERROR = 37
    MISSING_THRESHOLD_ERROR = 38
    WRONG_THRESHOLD_ERROR = 39
    MULTIPROCESS_ERROR = 40

    def __init__(self, code, error_info: str = ""):
        super(MsprobeBaseException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class CompareException(MsprobeBaseException):
    """
    Class for Accuracy Compare Exception
    """

    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__(code, error_info)


class DumpException(MsprobeBaseException):
    """
    Class for Dump Exception
    """

    def __init__(self, code, error_info: str = ""):
        super(DumpException, self).__init__(code, error_info)

    def __str__(self):
        return f"Dump Error Code {self.code}: {self.error_info}"


class ThreadSafe:
    """
    线程安全控制工具类，提供三种使用方式：
    1.上下文管理器：with ThreadSafe()
    2.主动加锁与释放锁：ThreadSafe.acquire()/ThreadSafe.release()
    3.方法装饰器：@ThreadSafe.synchronized
    """
    _lock = threading.RLock()

    def __enter__(self):
        self.__class__._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._lock.release()

    @classmethod
    def acquire(cls):
        cls._lock.acquire()

    @classmethod
    def release(cls):
        cls._lock.release()

    @classmethod
    def synchronized(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cls._lock:
                return func(*args, **kwargs)

        return wrapper


class ModuleQueue:
    def __init__(self):
        self.queue = OrderedDict()

    def add_name(self, name):
        self.queue[name] = True

    def remove_name(self, name):
        if name in self.queue:
            del self.queue[name]

    def find_last(self, name):
        """
        在队列中找到当前 Module/Cell 的父节点名称并返回，若找不到则返回None

        Args:
            name: 需要寻找父节点的 Module/Cell 的名称

        Returns:
            返回父节点名称，找不到则返回None

        Examples:
            父节点名称格式: Module.module1.module1.forward.0
            子节点名称格式: Module.module1.module2.Module2.forward.0
            匹配关系: Module/Cell 的名称总能被点(.)分割符分成5个部分及以上，子节点截断后4个点和父节点截断后3个点的前缀名称是匹配的
        """
        child_parts = name.split('.')
        if len(child_parts) < 5:
            return None
        child_name_prefix = '.'.join(child_parts[:-4])
        if child_name_prefix in Const.MODULE_PREFIX:
            return None

        for parent_name in reversed(self.queue):
            parent_parts = parent_name.split('.')
            if len(parent_parts) < 5:
                return None
            parent_name_prefix = '.'.join(parent_parts[:-3])
            if parent_name_prefix == child_name_prefix:
                return parent_name
        return None


def is_json_file(file_path):
    if isinstance(file_path, str) and file_path.lower().endswith('.json'):
        return True
    else:
        return False


def check_compare_param(input_param, output_path, dump_mode, stack_mode):
    if not isinstance(input_param, dict):
        logger.error(f"Invalid input parameter 'input_param', the expected type dict but got {type(input_param)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(output_path, str):
        logger.error(f"Invalid input parameter 'output_path', the expected type str but got {type(output_path)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)

    def check_json_path(json_path_str):
        json_path = input_param.get(json_path_str)
        check_file_or_directory_path(json_path, False)
        json_type_check = is_json_file(json_path)
        if not json_type_check:
            logger.error(f"Invalid {json_path_str}: {json_path}, please check!")
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    check_json_path("npu_path")
    check_json_path("bench_path")
    if stack_mode:
        check_json_path("stack_path")

    if dump_mode == Const.ALL:
        check_file_or_directory_path(input_param.get("npu_dump_data_dir"), True)
        check_file_or_directory_path(input_param.get("bench_dump_data_dir"), True)
    check_file_or_directory_path(output_path, True)

    with FileOpen(input_param.get("npu_path"), "r") as npu_json, \
            FileOpen(input_param.get("bench_path"), "r") as bench_json:
        _check_json(npu_json, input_param.get("npu_path"))
        _check_json(bench_json, input_param.get("bench_path"))
    if stack_mode:
        with FileOpen(input_param.get("stack_path"), "r") as stack_json:
            _check_json(stack_json, input_param.get("stack_path"))


def _check_json(json_file_handle, file_name):
    tensor_line = json_file_handle.readline()
    if not tensor_line:
        logger.error("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    json_file_handle.seek(0, 0)


def check_json_file(input_param, npu_json, bench_json, stack_json):
    _check_json(npu_json, input_param.get("npu_path"))
    _check_json(bench_json, input_param.get("bench_path"))
    _check_json(stack_json, input_param.get("stack_path"))


def check_regex_prefix_format_valid(prefix):
    """
        validate the format of the regex prefix

    Args:
        prefix (str): The prefix string to validate.

    Returns:
        no returns

    Raises:
        ValueError: if the prefix length exceeds Const.REGEX_PREFIX_MAX_LENGTH characters or the prefix do not match
        the given pattern Const.REGEX_PREFIX_PATTERN
    """
    if len(prefix) > Const.REGEX_PREFIX_MAX_LENGTH:
        raise ValueError(f"Maximum length of prefix is {Const.REGEX_PREFIX_MAX_LENGTH}, while current length "
                         f"is {len(prefix)}")
    if not re.match(Const.REGEX_PREFIX_PATTERN, prefix):
        raise ValueError(f"prefix contains invalid characters, prefix pattern {Const.REGEX_PREFIX_PATTERN}")


def add_time_as_suffix(name):
    return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_csv(name):
    return add_time_as_suffix(name)


def add_time_with_xlsx(name):
    return '{}_{}.xlsx'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_json(name):
    return '{}_{}.json'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_yaml(name):
    return '{}_{}.yaml'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def get_time():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_value(value):
    return float('{:.12f}'.format(value))


@recursion_depth_decorator('msprobe.core.common.utils.md5_find', max_depth=Const.DUMP_MAX_DEPTH)
def md5_find(data, json_type=Const.DUMP_JSON_FILE):
    if json_type == Const.DUMP_JSON_FILE:
        for key_op in data:
            for api_info in data[key_op]:
                if isinstance(data[key_op][api_info], list):
                    for data_detail in data[key_op][api_info]:
                        if data_detail and Const.MD5 in data_detail:
                            return True
                if isinstance(data[key_op][api_info], bool):
                    continue
                elif data[key_op][api_info] and Const.MD5 in data[key_op][api_info]:
                    return True
    elif json_type == Const.DEBUG_JSON_FILE:
        if isinstance(data, dict):
            if Const.MD5 in data:
                return True
            else:
                for _, data_info in data.items():
                    if md5_find(data_info, Const.DEBUG_JSON_FILE):
                        return True
        elif isinstance(data, list):
            for data_info in data:
                if md5_find(data_info, Const.DEBUG_JSON_FILE):
                    return True
        else:
            return False
    return False


def detect_framework_by_dump_json(file_path):
    json_data = load_json(file_path)
    framework = json_data.get("framework", None)
    if framework in [Const.PT_FRAMEWORK, Const.MS_FRAMEWORK]:
        return framework
    pattern_ms = r'"type":\s*"mindspore'
    pattern_pt = r'"type":\s*"torch'
    with FileOpen(file_path, 'r') as file:
        for line in file:
            if re.search(pattern_ms, line):
                return Const.MS_FRAMEWORK
            if re.search(pattern_pt, line):
                return Const.PT_FRAMEWORK
    logger.error(f"{file_path} must be based on the MindSpore or PyTorch framework.")
    raise CompareException(CompareException.INVALID_PARAM_ERROR)


def get_stack_construct_by_dump_json_path(dump_json_path):
    if not dump_json_path:
        logger.error("The path is empty. Please enter a valid path.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    directory = os.path.dirname(dump_json_path)
    check_file_or_directory_path(directory, True)
    stack_json = os.path.join(directory, "stack.json")
    construct_json = os.path.join(directory, "construct.json")

    stack_json_exist = os.path.exists(stack_json)
    construct_json_exist = os.path.exists(construct_json)

    if not stack_json_exist and not construct_json_exist:
        logger.info("stack.json and construct.json not found")
        return {}, {}
    if not stack_json_exist or not construct_json_exist:
        logger.error("stack.json or construct.json not found, please check.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    stack = load_json(stack_json)
    construct, _ = load_construct_json(construct_json)
    return stack, construct


def set_dump_path(input_param):
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    dump_json_path_valid = npu_path is not None and npu_path.endswith("dump.json") and \
                           bench_path is not None and bench_path.endswith("dump.json")
    debug_json_path_valid = npu_path is not None and npu_path.endswith("debug.json") and \
                            bench_path is not None and bench_path.endswith("debug.json")
    if not dump_json_path_valid and not debug_json_path_valid:
        logger.error(f"Please check the json path is valid and ensure that neither npu_path nor bench_path is None.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    input_param[CompareConst.NPU_DUMP_DATA_DIR] = os.path.join(os.path.dirname(npu_path), Const.DUMP_TENSOR_DATA)
    input_param[CompareConst.BENCH_DUMP_DATA_DIR] = os.path.join(os.path.dirname(bench_path), Const.DUMP_TENSOR_DATA)


def get_file_type(file_path):
    if not isinstance(file_path, str):
        logger.error("get_file_type failed, check the type of file_path.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    file_type = file_suffix_to_file_type.get(file_path.split(Const.SCOPE_SEPARATOR)[-1])
    if file_type is None:
        logger.error("get_file_type failed, file_path is neither dump.json nor debug.json.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    return file_type


def check_dump_json_key(json_data, device_type):
    task = json_data.get('task', None)
    if not task:
        logger.error(f"Task for {device_type} is empty, please check.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)
    if 'data' not in json_data:
        logger.error(f"Missing 'data' in dump.json, please check dump.json of {device_type}.")
        raise CompareException(CompareException.INVALID_DATA_ERROR)
    api_data = json_data.get('data')
    if not isinstance(api_data, dict):
        logger.error(f"Invalid type for 'data': expected a dict. Please check dump.json of {device_type}.")
        raise CompareException(CompareException.INVALID_DATA_ERROR)
    return task, api_data


def get_dump_mode(input_param):
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    npu_json_data = load_json(npu_path)
    bench_json_data = load_json(bench_path)
    json_type = get_file_type(file_path=npu_path)

    npu_task, npu_api_data = check_dump_json_key(npu_json_data, 'npu')
    bench_task, bench_api_data = check_dump_json_key(bench_json_data, 'bench')

    if npu_task != bench_task:
        logger.error(f"Please check the dump task is consistent.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)

    if npu_task == Const.TENSOR:
        return Const.ALL

    if npu_task == Const.STRUCTURE:
        return Const.STRUCTURE

    if npu_task == Const.STATISTICS:
        npu_md5_compare = md5_find(npu_api_data, json_type)
        bench_md5_compare = md5_find(bench_api_data, json_type)
        if npu_md5_compare == bench_md5_compare:
            return Const.MD5 if npu_md5_compare else Const.SUMMARY
        else:
            logger.error(f"Please check the dump task is consistent, "
                         f"dump mode of npu and bench should both be statistics or md5.")
            raise CompareException(CompareException.INVALID_TASK_ERROR)

    logger.error(f"Compare applies only to task is tensor or statistics")
    raise CompareException(CompareException.INVALID_TASK_ERROR)


def get_header_index(header_name, dump_mode):
    header = CompareConst.HEAD_OF_COMPARE_MODE.get(dump_mode)
    if not header:
        logger.error(f"{dump_mode} not in {CompareConst.HEAD_OF_COMPARE_MODE}")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if header_name not in header:
        logger.error(f"{header_name} not in data name")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return header.index(header_name)


def convert_tuple(data):
    return data if isinstance(data, tuple) else (data,)


def check_op_str_pattern_valid(string, op_name=None, stack=False):
    if isinstance(string, str) and is_invalid_pattern(string):
        if stack:
            message = f"stack info of {op_name} contains special characters, please check!"
        elif not op_name:
            message = f"api name contains special characters, please check!"
        else:
            message = f"data info of {op_name} contains special characters, please check!"
        logger.error(message)
        raise CompareException(CompareException.INVALID_CHAR_ERROR)


def is_invalid_pattern(string):
    pattern = Const.STRING_BLACKLIST
    return re.search(pattern, string)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def print_tools_ends_info():
    total_len = len(Const.TOOL_ENDS_SUCCESSFULLY) + Const.FILL_CHAR_NUMS
    logger.info('*' * total_len)
    logger.info(f"*{Const.TOOL_ENDS_SUCCESSFULLY.center(total_len - 2)}*")
    logger.info('*' * total_len)


def get_step_or_rank_from_string(step_or_rank, obj):
    splited = step_or_rank.split(Const.HYPHEN)
    if len(splited) == 2:
        try:
            borderlines = int(splited[0]), int(splited[1])
        except (ValueError, IndexError) as e:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   "The hyphen(-) must start and end with decimal numbers.") from e
    else:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f'The string parameter for {obj} only supports formats like "3-5". '
                               f'Now string parameter for {obj} is "{step_or_rank}".')
    if all(Const.STEP_RANK_MINIMUM_VALUE <= b <= Const.STEP_RANK_MAXIMUM_VALUE for b in borderlines):
        if borderlines[0] <= borderlines[1]:
            continual_step_or_rank = list(range(borderlines[0], borderlines[1] + 1))
        else:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f'For the hyphen(-) in {obj}, the left boundary ({borderlines[0]}) cannot be '
                                   f'greater than the right boundary ({borderlines[1]}).')
    else:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"The boundaries must fall within the range of "
                               f"[{Const.STEP_RANK_MINIMUM_VALUE}, {Const.STEP_RANK_MAXIMUM_VALUE}].")
    return continual_step_or_rank


def get_real_step_or_rank(step_or_rank_input, obj):
    if obj not in [Const.STEP, Const.RANK]:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"Only support parsing {[Const.STEP, Const.RANK]}, the current parsing object is {obj}.")
    if step_or_rank_input is None:
        return []
    if not isinstance(step_or_rank_input, list):
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, f"{obj} is invalid, it should be a list")
    if len(step_or_rank_input) > Const.STEP_RANK_MAXIMUM_VALUE:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{obj} is invalid, its length cannot exceed {Const.STEP_RANK_MAXIMUM_VALUE}")

    real_step_or_rank = []
    for element in step_or_rank_input:
        if not is_int(element) and not isinstance(element, str):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"{obj} element {element} must be an integer or string.")
        if is_int(element):
            if not Const.STEP_RANK_MINIMUM_VALUE <= element <= Const.STEP_RANK_MAXIMUM_VALUE:
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    f"Each element of {obj} must be between {Const.STEP_RANK_MINIMUM_VALUE} and "
                    f"{Const.STEP_RANK_MAXIMUM_VALUE}, currently it is {element}."
                )
            real_step_or_rank.append(element)
            continue
        continual_step_or_rank = get_step_or_rank_from_string(element, obj)
        real_step_or_rank.extend(continual_step_or_rank)
    real_step_or_rank = list(set(real_step_or_rank))
    real_step_or_rank.sort()
    return real_step_or_rank


def check_init_step(step):
    if not is_int(step):
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{step} must be an integer")
    if not step >= 0:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{step} must be greater than or equal to 0")


def check_token_range(token_range):
    if token_range is None:
        return
    if not isinstance(token_range, (list, tuple)):
        logger.error("Token_range must be a list or tuple.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if len(token_range) != 2:
        logger.error("Token_range must contains exactly 2 elements.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    start, end = token_range
    if not is_int(start) or not is_int(end):
        logger.error("Start and end in token_range must be integer.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if start > end:
        logger.error("Start in token_range must less than the end.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if start < 0:
        logger.error("Start in token_range must >= 0.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)


def check_seed_all(seed, mode, rm_dropout):
    if is_int(seed):
        if seed < 0 or seed > Const.MAX_SEED_VALUE:
            logger.error(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    else:
        logger.error("Seed must be integer.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        logger.error("seed_all mode must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(rm_dropout, bool):
        logger.error("The rm_dropout parameter must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)


def safe_get_value(container, index, container_name, key=None):
    try:
        # 处理字典情况
        if isinstance(container, dict):
            return container.get(key)[index]
        # 处理列表、元组、numpy情况
        elif isinstance(container, (list, tuple, np.ndarray)):
            return container[index]
        else:
            err_msg = f"Unsupported container type for '{container_name}': {type(container)}"
            logger.error(err_msg)
            raise MsprobeBaseException(MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)
    except IndexError as e:
        err_msg = "index out of bounds error occurs, please check!\n" \
                  f"{container_name} is {container}\n" \
                  f"index is {index}"
        logger.error(err_msg)
        raise MsprobeBaseException(MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR) from e
    except TypeError as e:
        err_msg = "wrong type, please check!\n" \
                  f"{container_name} is {container}\n" \
                  f"index is {index}\n" \
                  f"key is {key}"
        logger.error(err_msg)
        raise MsprobeBaseException(MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR) from e


def check_str_param(param):
    if not re.match(Const.REGEX_PREFIX_PATTERN, param):
        logger.error('The parameter {} contains special characters.'.format(param))
        raise MsprobeBaseException(MsprobeBaseException.INVALID_CHAR_ERROR)


class DumpPathAggregation:
    dump_file_path = None
    stack_file_path = None
    construct_file_path = None
    dump_tensor_data_dir = None
    debug_file_path = None


def is_save_variable_valid(variable, valid_special_types, depth=0):
    if depth > Const.DUMP_MAX_DEPTH:
        return False
    if isinstance(variable, valid_special_types):
        return True
    elif isinstance(variable, (list, tuple)):
        return all(is_save_variable_valid(item, valid_special_types, depth + 1) for item in variable)
    elif isinstance(variable, dict):
        return all(isinstance(key, str) and is_save_variable_valid(value, valid_special_types, depth + 1)
                   for key, value in variable.items())
    else:
        return False


def replace_last_occurrence(text, old, new):
    if text is None:
        return text
    index = text.rfind(old)
    if index != -1:
        return text[:index] + text[index:].replace(old, new, 1)
    return text


def load_stack_json(stack_path):
    stack_dict = load_json(stack_path)

    if not isinstance(stack_dict, dict):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            "The format of the stack.json is incorrect, the outermost layer of stack.json should be a dict type."
        )

    if not stack_dict.get(Const.NEW_STACK_FLAG):
        return stack_dict

    new_stack_dict = {}
    for stack_info in stack_dict.values():
        if not isinstance(stack_info, list) or len(stack_info) != 2:
            continue

        api_list, stack_str = stack_info
        if not isinstance(api_list, list):
            continue

        for api_name in api_list:
            new_stack_dict.update({api_name: stack_str})
    return new_stack_dict


def analyze_api_call_stack(name):
    try:
        api_stack = inspect.stack()[2:]
    except Exception as e:
        logger.warning(f"The call stack of {name} failed to retrieve, {e}.")
        api_stack = None
    stack_str = []
    if api_stack:
        for (_, path, line, func, code, _) in api_stack:
            if not code:
                continue
            stack_line = f"File {path}, line {str(line)}, in {func}, \n {code[0].strip()} \n"
            stack_str.append(stack_line)
    else:
        stack_str.append(Const.WITHOUT_CALL_STACK)
    return "".join(stack_str)


def check_extern_input_list(input_list):
    if not isinstance(input_list, list):
        raise Exception("input is not a list")
    if len(input_list) > Const.EXTERN_INPUT_LIST_MAX_LEN:
        raise Exception(f"input list exceed max length {Const.EXTERN_INPUT_LIST_MAX_LEN}")


def check_process_num(process_num):
    if not is_int(process_num) or process_num <= 0:
        raise ValueError(f"process_num({process_num}) is not a positive integer")
    if process_num > Const.MAX_PROCESS_NUM:
        raise ValueError(f"The maximum supported process_num is {Const.MAX_PROCESS_NUM}, current value: {process_num}.")


def is_module_available(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def check_rank_id(rank_id):
    if rank_id is None:
        return
    if not is_int(rank_id):
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{rank_id} must be an integer.")
    if not rank_id >= 0:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{rank_id} must be greater than or equal to 0.")


def is_np2():
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    return np_version >= (2, 0)

def is_torchrec_available():
    return _has_torchrec

def is_jagged_tensor(data):
    return is_torchrec_available() and isinstance(data, JaggedTensor)

def is_keyed_jagged_tensor(data):
    return is_torchrec_available() and isinstance(data, KeyedJaggedTensor)

def is_keyed_tensor(data):
    return is_torchrec_available() and isinstance(data, KeyedTensor)
