
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves file-related function.
"""
import csv
import json
import os
import re
import uuid
import numpy as np

from cmp_utils import log
from cmp_utils import utils
from cmp_utils import common
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError
from dump_parse import dump_utils

OP_TYPE = 1
OP_NAME = 2
TASK_ID = 3
STREAM_ID = 4
TIMESTAMP = 4
CONTEXT_ID = 6
THREAD_ID = 7


class FileUtils:
    """
    The class for file utils
    """
    CSV_SUFFIX = '.csv'

    @staticmethod
    def read_csv(path: str) -> list:
        """
        Read csv file to list.
        :path: csv file path
        """
        content = []
        path = os.path.realpath(path)  # 标准化文件路径
        if not str(path).endswith(FileUtils.CSV_SUFFIX):
            log.print_warn_log('read csv failed, file path'
                               ' [{}] is invalid'.format(path))
            return content
        utils.check_file_size(path, ConstManager.ONE_HUNDRED_MB)
        try:
            with open(path, 'r') as f:
                csv_handle = csv.reader(f)
                for row in csv_handle:
                    content.append(row)
            return content
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as io_error:
            log.print_open_file_error(path, io_error)
            raise CompareError(CompareError.MSACCUCMP_OPEN_FILE_ERROR) from io_error
        finally:
            pass

    @staticmethod
    def save_file(file_path: str, content: any) -> None:
        """
        save txt format result to file.
        :file_path: file path
        :content: txt content
        """
        if os.path.islink(file_path):
            os.unlink(file_path)
        try:
            with os.fdopen(os.open(file_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           'w+') as output_file:
                output_file.write(content)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            log.print_error_log('Failed to open "%r". %s ' % (file_path, str(error)))
            raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from error
        finally:
            pass

    @staticmethod
    def delete_file(path: str) -> None:
        '''Delete file if it exists and user has permission'''
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as error:
            raise CompareError(CompareError.MSACCUCMP_DELETE_FILE_ERROR) from error

    @staticmethod
    def save_data_to_file(path: str, data: any, flag: str, delete: bool) -> None:
        """
        Save data to file.
        :param path: the saved file path
        :param data: the data to save
        :param flag: the write flag
        :param delete: delete the path or not
        """
        if delete:
            FileUtils.delete_file(path)

        if flag == "r":
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

        try:
            with os.fdopen(os.open(path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES), flag) as output_file:
                output_file.write(data)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            log.print_error_log('Failed to write data to "%r". %s ' % (path, str(error)))
            raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from error
        finally:
            pass

    @staticmethod
    def save_array_to_file(path: str, array: np.ndarray, np_save: bool, shape: any = None) -> None:
        """
        Save numpy array to file.
        :param path: the saved file path
        :param array: the numpy array
        :param np_save: save or not
        :param shape: the array shape
        """
        try:
            FileUtils._save_array_to_file(path, array, np_save, shape)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, MemoryError) as error:
            log.print_error_log('Failed to write data to "%r". %s' % (path, str(error)))
        finally:
            pass

    @staticmethod
    def handle_too_long_file_name(old_file_name: str, suffix: str, mapping_file: str) -> str:
        """
        Handle too log file name and save {new,old} map to file
        :param old_file_name: the old file name
        :param suffix: the file suffix
        :param mapping_file: the mapping file
        """
        if len(old_file_name) >= ConstManager.LINUX_FILE_NAME_MAX_LEN:
            value = ''.join(str(uuid.uuid3(uuid.NAMESPACE_DNS, old_file_name)).split('-'))
            new_file_name = "%s%s" % (value, suffix)
            FileUtils.save_data_to_file(mapping_file, "%s,%s\n" % (new_file_name, old_file_name), 'a+', delete=False)
            return new_file_name
        return old_file_name

    @staticmethod
    def _save_array_to_file(path: str, array: np.ndarray, np_save: bool, shape: any = None) -> None:
        '''
        Save numpy array to file.
        :param path: the saved file path
        :param array: the numpy array
        :param np_save: save or not
        :param shape: the array shape
        '''
        if shape:
            array = array.reshape(shape)

        if os.path.islink(path):
            os.unlink(path)
        if os.path.exists(path):
            os.remove(path)

        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT, ConstManager.WRITE_MODES), 'wb') as f:
            if np_save:
                np.save(f, array)
            else:
                array.tofile(f)
        os.chmod(path, ConstManager.WRITE_MODES)

    @classmethod
    def load_json_file(cls: any, json_file: str) -> any:
        """
        load json file
        :json_file: the json file
        """
        utils.check_file_size(json_file, ConstManager.ONE_HUNDRED_MB)
        try:
            with open(json_file, 'r') as input_file:
                return json.load(input_file)
        except Exception as error:
            log.print_error_log(
                'Failed to load json object. The content of the json file "%r" is invalid.' % json_file)
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR) from error


class OverflowFileUtils(FileUtils):
    """
    The class for the overflow files
    """
    # file name patterns
    # example:
    # dump_file: Add.partition0_rank1_new_sub_graph1_sgt_graph_0_Add.0.3.1673383728093446.4.1.0.0
    # opdeug_file: Opdebug.Node_OpDebug.0.3.1673383728094391.4.1.0.0
    # parsed_debug_file: Opdebug.Node_OpDebug.0.3.1673383728094391.4.1.0.0.output.0.json
    DUMP_FILE_PATTERN = r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)" \
                        r"\.([0-9]+)\.?([0-9]+)?\.([0-9]{1,255})" \
                        r"\.?([0-9]+)?\.?([0-9]+)?\.?([0-9]+)?\.?([0-9]+)?"
    PARSED_DUMP_FILE_PATTERN = \
        r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)\.?([0-9]+)?" \
        r"\.([0-9]{1,255})\.([a-z]+)\.([0-9]{1,255})\.npy$"
    DEBUG_FILE_PATTERN = r"(Opdebug)\.(Node_OpDebug)\.([0-9]+)\.?([0-9]+)?\.([0-9]{1,255})" \
                         r"\.?([0-9]+)?\.?([0-9]+)?\.?([0-9]+)?\.?([0-9]+)?"
    PARSED_DEBUG_FILE_PATTERN = r"Opdebug\.Node_OpDebug\.([0-9]+)\.?([0-9]+)?" \
                                r"\.([0-9]{1,255})(\.[0-9])?\.?([0-9]+)?\.?([0-9]+)?" \
                                r"\.?([0-9]+)?\.([a-z]+)\.([0-9]{1,255})\.json"
    MAX_DEPTH = 30

    @staticmethod
    def _list_file_with_pattern(path: str, pattern: str,
                                extern_pattern: any, gen_info_func: any) -> dict:
        """
        find all files for a specific pattern
        :path: the path of the dir
        :pattern: the pattern
        :extern_pattern: the extern_pattern
        :gen_info_func: a function to gen the file obj
        """
        if not path or not os.path.exists(path):
            raise CompareError(
                CompareError.MSACCUCMP_INVALID_PATH_ERROR)

        matched_files = {}
        re_pattern = re.compile(pattern)
        for dir_path, _, files in os.walk(path):
            current_depth = dir_path.count(os.sep) - path.count(os.sep)
            if current_depth > OverflowFileUtils.MAX_DEPTH:
                log.print_warn_log(
                    "The current file depth has exceeded the maximum depth limit, "
                    f"which is set to {OverflowFileUtils.MAX_DEPTH}.")
                break
            for file in files:
                match = re_pattern.match(file)
                if match is None:
                    continue
                if extern_pattern != '' and not re.match(extern_pattern, file):
                    continue
                if gen_info_func:
                    matched_files[file] = gen_info_func(file, dir_path, match)
        return matched_files

    @staticmethod
    def _gen_dump_file_info(name: str, dir_path: str, match: any) -> any:
        """
        generate the dump file object
        :name: file name
        :dir_path: file path
        :match: the result of re.match() match with file name
        """

        if len(match.groups()) > 4:
            file_desc = {
                "file_path": os.path.join(dir_path, name),
                "timestamp": int(match.groups()[TIMESTAMP])
            }
            dump_attr = {
                "op_name": match.group(OP_NAME),
                "op_type": match.group(OP_TYPE),
                "task_id": int(match.group(TASK_ID)),
                "stream_id": match.group(STREAM_ID)
            }
        if len(match.groups()) > 7 and match.groups()[CONTEXT_ID] and match.groups()[THREAD_ID]:
            dump_attr["context_id"] = int(match.groups()[CONTEXT_ID])
            dump_attr["thread_id"] = int(match.groups()[THREAD_ID])

        return DumpFileDesc(file_desc, dump_attr)

    @staticmethod
    def _gen_one_tensor_desc(output_path: str, dump_file_desc: any,
                             tensor_list: any, tensor_type: str = '') -> dict:
        """
        generate the ParsedDumpFileDesc of the tensor list
        :output_path: the output path
        :dump_file_desc: the DumpFileDesc for the debug file
        :tensor_list: tensor list of one dump file
        :tensor_type: tensor type, such as "input"/"output"
        """
        if len(tensor_list) == 0:
            return {}

        dump_file_name = os.path.basename(dump_file_desc.file_path)
        parsed_dump_files = {}
        for (index, tensor) in enumerate(tensor_list):
            parsed_dump_file_name = "{}.{}.{}.npy" \
                .format(dump_file_name, tensor_type, index)
            file_desc = {
                "file_path": os.path.join(output_path, parsed_dump_file_name),
                "timestamp": dump_file_desc.timestamp
            }
            dump_attr = {
                "op_name": dump_file_desc.op_name,
                "op_type": dump_file_desc.op_type,
                "task_id": dump_file_desc.task_id,
                "stream_id": dump_file_desc.stream_id
            }
            anchor = {
                "anchor_type": tensor_type,
                "anchor_idx": index,
                "format": common.get_format_string(tensor.tensor_format)
            }

            parsed_dump_files[parsed_dump_file_name] \
                = ParsedDumpFileDesc(file_desc, dump_attr, anchor)
        return parsed_dump_files

    @staticmethod
    def _gen_parsed_debug_file_info(name: str, dir_path: str, match: any) -> object:
        """
        generate the objects of the parsed debug_file.
        :name: parsed file name
        :dir_path: the path of parsed debug file
        :match: the result of re.match() match with \
               the name of parsed debug file
        """
        file_desc = {
            "file_path": os.path.join(dir_path, name),
            "timestamp": int(match.groups()[2])
        }
        dump_attr = {
            "op_name": 'Node_OpDebug',
            "op_type": 'Opdebug',
            "task_id": int(match.group(1)),
            "stream_id": match.group(2)
        }
        if len(match.groups()) > 5 and match.groups()[4] and match.groups()[5]:
            dump_attr["context_id"] = int(match.groups()[4])
            dump_attr["thread_id"] = int(match.groups()[5])
        anchor = {
            "anchor_type": match.groups()[-2],
            "anchor_idx": int(match.groups()[-1])
        }
        return ParsedDumpFileDesc(file_desc, dump_attr, anchor)

    def parse_mapping_csv(self: any, path: str, pattern: str, extern_pattern: any = '') -> dict:
        """
        parse files in mapping.csv
        :path: the path of the dir
        :pattern: the pattern
        """
        hash_index = 0
        file_name_index = 1
        matched_files = {}
        re_pattern = re.compile(pattern)
        if ConstManager.MAPPING_FILE_NAME not in os.listdir(path):
            return matched_files
        mapping = self.read_csv(os.path.join(path, ConstManager.MAPPING_FILE_NAME))
        for item in mapping:
            src_file = os.path.realpath(os.path.join(path, item[hash_index]))
            if not os.path.isfile(src_file):
                log.print_warn_log("the file %r in mapping.csv is not exist, dir: %r."
                                   % (item[hash_index], path))
                continue
            match = re_pattern.match(item[file_name_index])
            if match is None:
                continue
            if extern_pattern != '' and not re.match(extern_pattern,
                                                     item[file_name_index]):
                continue

            matched_files[item[hash_index]] = self._gen_dump_file_info(
                item[hash_index], path, match)
        return matched_files

    def list_dump_files(self: any, dump_path: str, pattern: str, extern_pattern: any = '') -> any:
        """
        parse all the dump files
        :dump_path: the dump file path
        :pattern: the special matching mode of dump files
        :extern_pattern: the extern_pattern
        """
        npu_dump_files = self._list_file_with_pattern(
            dump_path, pattern, extern_pattern, self._gen_dump_file_info)
        npu_dump_files.update(
            self.parse_mapping_csv(dump_path, pattern, extern_pattern))
        return list(npu_dump_files.values())

    def list_parsed_dump_files(self: any, output_path: str, dump_file_desc: any) -> dict:
        """
        search all the parsed npu dump files
        :output_path: the output path
        :dump_file_desc: the DumpFileDesc for the debug file
        """
        # only support the default dump version ‘2’
        dump_data = dump_utils.parse_dump_file(dump_file_desc.file_path, ConstManager.BINARY_DUMP_TYPE)
        dump_file_desc.set_op_name(dump_data.op_name)
        parsed_dump_files = self._gen_one_tensor_desc(output_path, dump_file_desc,
                                                      dump_data.input_data, 'input')
        parsed_dump_files.update(self._gen_one_tensor_desc(output_path, dump_file_desc,
                                                           dump_data.output_data, 'output'))
        parsed_dump_files.update(self._gen_one_tensor_desc(output_path, dump_file_desc,
                                                           dump_data.buffer))
        return parsed_dump_files

    def list_parsed_debug_files(self: any, dir_path: str, extern_pattern: str = '') -> dict:
        """
        search for all parsed debug files in the path.
        :dir_path: the path
        :extern_pattern: the extern_pattern
        """
        return self._list_file_with_pattern(
            dir_path, self.PARSED_DEBUG_FILE_PATTERN, extern_pattern,
            self._gen_parsed_debug_file_info)


class FileDesc:
    """
    The class for file description
    """

    def __init__(self: any, file_desc: dict) -> None:
        self.file_path = file_desc.get("file_path")
        self.timestamp = file_desc.get("timestamp")
        if not self.timestamp:
            self.timestamp = os.path.getmtime(self.file_path)

    def get_file_path(self: any) -> str:
        """
        get file name
        """
        return self.file_path

    def get_file_time(self: any) -> float:
        """
        get file path
        """
        return self.timestamp


class DumpFileDesc(FileDesc):
    """
    The class for file dump file description
    """

    def __init__(self: any, file_desc: dict, dump_attr: dict) -> None:
        super(DumpFileDesc, self).__init__(file_desc)
        self.op_name = dump_attr.get("op_name")
        self.op_type = dump_attr.get("op_type")
        self.task_id = dump_attr.get("task_id")
        self.stream_id = int(dump_attr.setdefault("stream_id", '0'))
        self.context_id = dump_attr.setdefault("context_id", None)
        self.thread_id = dump_attr.setdefault("thread_id", None)

    def set_op_name(self: any, op_name: str) -> None:
        """
        update op name.
        :op_name: the new op_name
        """
        self.op_name = op_name


class ParsedDumpFileDesc(DumpFileDesc):
    """
    The class for file decoded dump file description
    """

    def __init__(self: any, file_desc: dict, dump_attr: dict, anchor: dict) -> None:
        super(ParsedDumpFileDesc, self).__init__(file_desc, dump_attr)
        self.type = anchor.get("anchor_type")
        self.idx = anchor.get("anchor_idx")
        self.format = anchor.get("format")


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)
