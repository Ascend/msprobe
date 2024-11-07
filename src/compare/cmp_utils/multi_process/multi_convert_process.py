
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
MultiConvertProcess class. This class mainly involves the process function.
"""
import os
import multiprocessing
import psutil
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.file_utils import FileUtils

from cmp_utils.multi_process.progress import Progress
from cmp_utils.constant.compare_error import CompareError


class MultiConvertProcess:
    """
    The class for multi process for convert
    """
    MAX_MULTI = 4
    MULTI_THREAD_RESULT_COUNT = 2
    MULTI_THREAD_RETURN_CODE_INDEX = 0
    MULTI_THREAD_ERROR_FILE_INDEX = 1

    def __init__(self: any, process_func: any, input_path: list, output_path: str) -> None:
        self._process_func = process_func
        self._output_path = output_path
        self._input_path = input_path
        self._progress = None

    def process(self: any) -> int:
        """
        Process by multi process
        """
        return_code = CompareError.MSACCUCMP_NONE_ERROR

        # split big file and common file
        multi_process_file_list, big_file_list = self._split_big_file()

        if len(multi_process_file_list) > 0:
            ret = self._do_multi_process(multi_process_file_list)
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                return_code = ret
        # big file do not multi process
        for big_file in big_file_list:
            ret, _ = self._process_func(big_file)
            self._handle_result_callback([ret, big_file])
            if ret != CompareError.MSACCUCMP_NONE_ERROR:
                return_code = ret
        if return_code != CompareError.MSACCUCMP_NONE_ERROR:
            error_file_path = os.path.join(self._output_path, ConstManager.CONVERT_FAILED_FILE_LIST_NAME)
            if os.path.exists(error_file_path):
                log.print_info_log('The list of files that failed to be converted has been written to "%r".'
                                   % error_file_path)

        return return_code

    def get_max_file_size(self: any) -> int:
        """
        Get max file size
        :return int
        """
        mem = psutil.virtual_memory()
        available = mem.available
        cpu_count = int((multiprocessing.cpu_count() + 1) / 2)
        if cpu_count != 0:
            return available / cpu_count / self.MAX_MULTI
        else:
            return 0

    def _handle_result_callback(self: any, result: list) -> None:
        self._progress.update_progress()
        self._progress.update_and_print_progress()
        if len(result) == self.MULTI_THREAD_RESULT_COUNT:
            cur_ret = result[self.MULTI_THREAD_RETURN_CODE_INDEX]
            if cur_ret != CompareError.MSACCUCMP_NONE_ERROR:
                error_file_path = os.path.join(self._output_path, ConstManager.CONVERT_FAILED_FILE_LIST_NAME)
                FileUtils.save_data_to_file(error_file_path, "%s\n" % result[self.MULTI_THREAD_ERROR_FILE_INDEX],
                                            'a+', delete=False)

    def _do_multi_process(self: any, file_list: list) -> int:
        cpu_count = int((multiprocessing.cpu_count() + 1) / 2)
        pool = multiprocessing.Pool(cpu_count)
        all_task = []
        for cur_path in file_list:
            task = pool.apply_async(self._process_func,
                                    args=(cur_path,),
                                    callback=self._handle_result_callback)
            all_task.append(task)
        pool.close()
        pool.join()
        for task in all_task:
            result = task.get()
            if len(result) != self.MULTI_THREAD_RESULT_COUNT:
                continue
            cur_ret = result[self.MULTI_THREAD_RETURN_CODE_INDEX]
            if cur_ret != CompareError.MSACCUCMP_NONE_ERROR:
                return cur_ret
        return CompareError.MSACCUCMP_NONE_ERROR

    def _split_big_file(self: any) -> (list, list):
        if len(self._input_path) == 1:
            files = os.listdir(self._input_path[0])
        else:
            files = self._input_path
        self._progress = Progress(len(files))
        multi_process_file_list = []
        big_file_list = []
        max_file_size = self.get_max_file_size()
        for cur_file in files:
            if cur_file == ConstManager.MAPPING_FILE_NAME:
                continue
            if len(self._input_path) == 1:
                cur_path = os.path.join(self._input_path[0], cur_file)
            else:
                cur_path = cur_file
            if os.path.isfile(cur_path):
                # skip big file
                if os.path.getsize(cur_path) > max_file_size:
                    big_file_list.append(cur_path)
                else:
                    multi_process_file_list.append(cur_path)
        return multi_process_file_list, big_file_list
