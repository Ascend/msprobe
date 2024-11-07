
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This file mainly involves the overflow analyse function.
"""
import argparse
import os
import time

import numpy as np

from cmp_utils import log, utils_type, path_check
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from dump_parse.dump_data_parser import DumpDataParser
from cmp_utils.file_utils import OverflowFileUtils, DumpFileDesc


class OverflowAnalyse:
    """
    The class for parse the overflow info
    """

    OVER_FLOW_TYPE = [
        'AI Core',
        'DHA Atomic Add',
        'L2 Atomic Add'
    ]

    def __init__(self: any, args: any = None) -> None:
        self.dump_path = os.path.realpath(args.dump_path)
        if os.path.islink(os.path.abspath(args.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % args.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        self.output_path = os.path.realpath(args.output_path)
        self.top_n = args.top_num
        self.debug_files = None
        self.overflow_file_utils = OverflowFileUtils()

    @staticmethod
    def check_argument(arguments: any = None) -> int:
        """
        check the arguments of overflow
        :arguments: input args
        """
        ret = path_check.check_path_valid(arguments.dump_path, exist=True,
                                     path_type=path_check.PathType.Directory)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            log.print_error_log('[Overflow] the -d parameter: "%r"'
                                ' is invalid!' % arguments.dump_path)
        else:
            ret = path_check.check_output_path_valid(arguments.output_path, exist=True)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                log.print_error_log('[Overflow] the -out parameter: "%r"'
                                    ' is invalid!' % arguments.output_path)
        return ret

    @staticmethod
    def npy_data_summary(source_data: any) -> str:
        """
        Get npy information
        :source_data: np data
        """
        if isinstance(source_data, str):
            if not str(source_data).endswith('.npy'):
                raise CompareError(
                    CompareError.MSACCUCMP_INVALID_TYPE_ERROR)
            data = np.load(source_data)
        elif isinstance(source_data, np.ndarray):
            data = source_data
        else:
            raise CompareError(
                CompareError.MSACCUCMP_INVALID_TYPE_ERROR)
        if np.size(data) == 0:
            raise CompareError(
                CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)
        return '[Shape: %s] [Dtype: %s] [Max: %s] [Min: %s] [Mean: %s]' \
               % (data.shape, data.dtype, np.max(data), np.min(data), data.mean())

    @staticmethod
    def _get_overflow_info_new_version(res: list, json_txt: any) -> any:
        acc_type = json_txt.get('acc_list', {}).get('acc_type')
        if acc_type in ConstManager.ACC_TYPE.values():
            overflow_type = acc_type
            detail = json_txt.get('acc_list', {}).get('data', {})
            return OverflowAnalyse._gen_overflow_info(res, overflow_type, detail)
        log.print_error_log("[Overflow] Invalid overflow type, type is {}".format(acc_type))
        raise CompareError(CompareError.MSACCUCMP_INVALID_OVERFLOW_TYPE_ERROR)

    @staticmethod
    def _get_overflow_info_old_version(res: list, json_txt: any) -> any:
        overflow = False
        id_info = ()
        for overflow_type, detail in json_txt.items():
            if detail.get('status'):
                overflow = True
                id_info = OverflowAnalyse._gen_overflow_info(res, overflow_type, detail)

        if not overflow:
            log.print_error_log("[Overflow] Invalid json_txt, overflow type invalid or status is zero!")
            raise CompareError(CompareError.MSACCUCMP_INVALID_OVERFLOW_TYPE_ERROR)
        return id_info

    @staticmethod
    def _gen_overflow_info(res: list, overflow_type: str, detail: any) -> any:
        status = detail.get('status')
        if status and status != 0:
            overflow_info = ' [%s][TaskId:%s][StreamId:%s][Status:%s]' \
                            % (overflow_type, detail.get('task_id'),
                               detail.get('stream_id'), status)
            res.append(overflow_info)
            task_info = (
                detail.get('task_id'),
                detail.get('stream_id'),
                detail.setdefault('context_id', ConstManager.INVALID_ID),
                detail.setdefault('thread_id', ConstManager.INVALID_ID)
            )
            return task_info
        log.print_error_log("[Overflow] The OpDebug file exists, but the value of status is {}!".format(status))
        raise CompareError(CompareError.MSACCUCMP_INVALID_OVERFLOW_STATUS_ERROR)

    @staticmethod
    def _insert_delimiter(res: list, over_index: int) -> list:
        """
        insert delimiter for every overflow result
        :over_index: the overflow index
        """
        res.insert(0, '=================================================[%d]'
                      '==================================================' % over_index)
        return res

    @staticmethod
    def _parse_overflow_file(file_path: str, output_path: str) -> any:
        """
        parse debug file or dump/debug file
        :file_path: the path of dump/debug file
        :output_path: store the result file
        """
        args = argparse.Namespace(dump_path=file_path,
                                  output_path=output_path,
                                  dump_version=2,
                                  output_file_type='npy')
        return DumpDataParser(args).parse_dump_data()

    @staticmethod
    def _is_dump_file_match(dump_file_desc: DumpFileDesc, task_info: any):
        task_id = task_info[0]
        stream_id = task_info[1]
        context_id = task_info[2] if task_info[2] != ConstManager.INVALID_ID else None
        thread_id = task_info[3] if task_info[3] != ConstManager.INVALID_ID else None
        if dump_file_desc.task_id != task_id or dump_file_desc.stream_id != stream_id:
            return False
        if context_id and dump_file_desc.context_id != context_id:
            return False
        if thread_id and dump_file_desc.thread_id != thread_id:
            return False
        return True

    def analyse(self: any) -> int:
        """
        analyse overflow info
        """
        if self._find_all_debug_files():
            max_num = len(self.debug_files)
            if self.top_n < len(self.debug_files):
                max_num = self.top_n
            log.print_info_log('[Overflow] Find [{}] overflow ops. Will show the top {}.'
                               .format(len(self.debug_files), max_num))

            overflow_result = ''
            for i, debug_file in enumerate(self.debug_files):
                if i >= max_num:
                    break
                parsed_debug_file = self._get_parsed_debug_file(debug_file)
                overflow_json = self.overflow_file_utils.load_json_file(parsed_debug_file.file_path)
                if not isinstance(overflow_json, dict):
                    log.print_warn_log("[Overflow] overflow summary file {} contents is not dict"
                                       .format(parsed_debug_file.file_path))
                    continue
                overflow_result = '{}{}\n'.format(overflow_result,
                                                  self._json_summary(i + 1, overflow_json, debug_file))

            log.print_info_log("[Overflow] The overflow analyse result:\n %s" % overflow_result)
            result_file_name = 'overflow_summary_%s.txt' \
                               % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            self.overflow_file_utils.save_file(os.path.join(self.output_path, result_file_name),
                                               overflow_result)

        return CompareError.MSACCUCMP_NONE_ERROR

    def _find_all_debug_files(self: any) -> bool:
        """
        find debug_files and sort it
        """
        debug_files = self.overflow_file_utils.list_dump_files(self.dump_path,
                                                               self.overflow_file_utils.DEBUG_FILE_PATTERN)
        # sort by timestamp
        self.debug_files = sorted(debug_files, key=lambda x: x.timestamp)
        if len(self.debug_files) == 0:
            log.print_warn_log("[Overflow] Find [0] overflow node!")
            return False
        return True

    def _json_summary(self: any, overflow_index: int, json_txt: any, debug_file: any) -> any:
        """
        get the summary info about overflow op
        :overflow_index: index of overflow op
        :json_txt: the parsed info from debug file
        :debug_file: the desc of debug file
        """
        res = []
        if ConstManager.MAGIC_KEY_WORD in json_txt.keys():
            task_info = self._get_overflow_info_new_version(res, json_txt)
        else:
            task_info = self._get_overflow_info_old_version(res, json_txt)
        res.append(' [timestamp:%s]' % debug_file.timestamp)
        try:
            dump_file_desc = self._find_dump_files_by_task_id(os.path.dirname(debug_file.file_path),
                                                              task_info)
        except CompareError:
            log.print_warn_log("[Overflow] Can't find the dump file corresponding to the"
                               " debug file: %s" % os.path.basename(debug_file.file_path))
            res = self._insert_delimiter(res, overflow_index)
            return '\n'.join(res)
        finally:
            pass

        parsed_dump_files = self._get_parsed_dump_file(dump_file_desc)
        # sort input/output & index
        for anchor_type in ['input', 'output']:
            for parsed_dump_file in parsed_dump_files:
                if parsed_dump_file.type != anchor_type:
                    continue
                res.append(' %s' % os.path.basename(parsed_dump_file.file_path))
                res.append(' -[Format: %s] %s' % (parsed_dump_file.format,
                                                  self.npy_data_summary(parsed_dump_file.file_path)))

        res.insert(0, '[%s] %s' % (dump_file_desc.op_type, dump_file_desc.op_name))
        res = self._insert_delimiter(res, overflow_index)
        return "\n".join(res)

    def _get_parsed_debug_file(self: any, debug_file_desc: any) -> any:
        """
        get parsed debug file
        :debug_file_desc: the DumpFileDesc of the debug file
        """
        file_name = os.path.basename(debug_file_desc.file_path)
        parsed_debug_files = self.overflow_file_utils.list_parsed_debug_files(self.output_path, file_name)
        if not parsed_debug_files:
            self._parse_overflow_file(debug_file_desc.file_path, self.output_path)
            parsed_debug_files = self.overflow_file_utils.list_parsed_debug_files(self.output_path, file_name)
            if not parsed_debug_files:
                log.print_warn_log("[Overflow] Parsed overflow debug file: %s failed." % file_name)
                raise CompareError(CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR)
        # only one json file when parse one debug file
        if len(parsed_debug_files) != 1:
            raise CompareError(CompareError.MSACCUCMP_MATCH_MORE_FILE_ERROR)
        return list(parsed_debug_files.values())[0]

    def _get_parsed_dump_file(self: any, dump_file_desc: any) -> list:
        """
        get parsed dump file
        :dump_file_desc: the DumpFileDesc of the dump file
        """
        self._parse_overflow_file(dump_file_desc.file_path, self.output_path)
        parsed_dump_files = self.overflow_file_utils.list_parsed_dump_files(self.output_path, dump_file_desc)
        if len(parsed_dump_files) == 0:
            log.print_warn_log("[Overflow] Parsed overflow dump file: %s failed."
                               % os.path.basename(dump_file_desc.file_path))
            raise CompareError(CompareError.MSACCUCMP_UNKNOWN_ERROR)
        return sorted(parsed_dump_files.values(), key=lambda x: x.idx)

    def _find_dump_files_by_task_id(self: any, dump_path: str, task_info: any) -> any:
        """
        find dump file by the task id and stream id
        :dump_path：the dump file path
        :task_id: the task id for the dump file
        :stream_id: the steam id for the dump file
        """
        dump_files = self.overflow_file_utils.list_dump_files(dump_path, self.overflow_file_utils.DUMP_FILE_PATTERN)
        dump_file_list = []
        for item in dump_files:
            if item.op_type != 'Opdebug' \
                    and self._is_dump_file_match(item, task_info):
                dump_file_list.append(item)
        if dump_file_list:
            dump_file_list.sort(key=lambda x: x.timestamp)
            return dump_file_list[0]
        raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
