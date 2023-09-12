# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the common function.
"""
import itertools
import multiprocessing
import os
import uuid

import numpy as np

from cmp_utils import log, path_check
from dump_parse import dump_utils, mapping
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.constant.const_manager import ConstManager
from vector_cmp.compare_detail.detail import DetailInfo
from cmp_utils.file_utils import FileUtils
from cmp_utils.multi_process.progress import Progress


class MinMaxValue:
    """
    The class for absolute and relative value min max value recorder
    """

    def __init__(self: any) -> None:
        self._min_absolute_error = np.nan
        self._max_absolute_error = np.nan
        self._min_relative_error = np.nan
        self._max_relative_error = np.nan

    def get_min_and_max_value(self: any) -> str:
        """
        Get min/max absolute error and min/max relative error
        """
        return "MinAbsoluteError:%.6f\nMaxAbsoluteError:%.6f\nMinRelativeError:%.6f\nMaxRelativeError:%.6f" \
               % (self._min_absolute_error, self._max_absolute_error,
                  self._min_relative_error, self._max_relative_error)

    def set_min_absolute_error(self: any, value: any) -> None:
        """
        Set min absolute error
        """
        self._min_absolute_error = value

    def set_max_absolute_error(self: any, value: any) -> None:
        """
        Set max absolute error
        """
        self._max_absolute_error = value

    def set_min_relative_error(self: any, value: any) -> None:
        """
        Set min relative error
        """
        self._min_relative_error = value

    def set_max_relative_error(self: any, value: any) -> None:
        """
        Set max relative error
        """
        self._max_relative_error = value


class TopN:
    """
    The class for recording the top n absolute/relative error values
    """

    def __init__(self: any, is_bool: bool = False) -> None:
        self._absolute_error_top_n_list = []
        self._relative_error_top_n_list = []
        self.is_bool = is_bool

    def set_absolute_top_n_list(self: any, index_list: list, *data_lists: any) -> None:
        """
        set and store the top n absolute error values
        """
        self._absolute_error_top_n_list = self._make_top_n_list(index_list, *data_lists)

    def set_relative_top_n_list(self: any, index_list: list, *data_lists: any) -> None:
        """
        set and store the top n relative error values
        """
        self._relative_error_top_n_list = self._make_top_n_list(index_list, *data_lists)

    def get_absolute_error_top_n_list(self: any) -> list:
        """
        get top n absolute error list
        """
        return self._absolute_error_top_n_list

    def get_relative_error_top_n_list(self: any) -> list:
        """
        get top n relative error list
        """
        return self._relative_error_top_n_list

    def _make_top_n_list(self: any, index_list: list, *data_lists: any) -> list:
        if len(data_lists) < 6:
            raise RuntimeError('The number of arguments is not correct.')
        dim_list = data_lists[0]
        left_list = data_lists[1]
        right_list = data_lists[2]
        absolute_error_list = data_lists[3]
        relative_error_list = data_lists[4]
        error_type = data_lists[5]
        top_n_original_list = []
        for i, index in enumerate(index_list):
            # variable row value,for example,[17866338,"22 16 32 98",-0.866699,-0.000017,0.866682,51349.906250]
            row = [
                index, " ".join((str(x) for x in dim_list[i])), left_list[i], right_list[i],
                absolute_error_list[i], relative_error_list[i]
            ]
            top_n_original_list.append(row)
        if error_type == "absolute":
            # 4 is AbsoluteError column subscript.
            # 5 is RelativeError column subscript.
            # 0 is Index column subscript.
            return self._get_top_n_result(top_n_original_list, 4, 0)
        return self._get_top_n_result(top_n_original_list, 5, 0)

    def _get_top_n_result(self: any, original_list: list, first_key: int, second_key: int) -> list:
        top_n_list = []
        sort_list = sorted(original_list, key=lambda x: (x[first_key], x[second_key]), reverse=True)
        if not self.is_bool:
            item_template = '%d,%s,%.6f\t,%.6f\t,%.6f\t,%.6f\t\n'
        else:
            item_template = '%d,%s,%s\t,%s\t,%.6f\t,%.6f\t\n'
        for item in sort_list:
            if self.is_bool:
                item[2] = str(item[2])
                item[3] = str(item[3])
            line = item_template % tuple(item)
            line = line.replace("nan", "-").replace("inf", "-")
            top_n_list.append(line)
        return top_n_list


class DetailWriter:
    """
    The class for detail writer
    """

    def __init__(self: any, output_path: str, detail_info: DetailInfo) -> None:
        self.output_path = output_path
        self.progress = None
        self.detail_info = detail_info
        self.min_max_value = MinMaxValue()
        self.top_n = TopN()
        self._total_num = 0
        self.dump_file_name = ""

    @staticmethod
    def _padding_shape_to_4d(dim: tuple) -> list:
        left_count = max(0, ConstManager.FOUR_DIMS_LENGTH) - len(dim)
        return list(dim) + left_count * [1]

    @staticmethod
    def _write_one_detail(index: int, file_stream: any, *data_values: any, is_bool: bool = False) -> None:
        cur_dim = data_values[0]
        my_output_value = data_values[1]
        ground_truth_value = data_values[2]
        absolute_error = data_values[3]
        relative_error = data_values[4]
        # add index, n,c,h,w or id, my output value, ground truth value, absolute error, relative error to line
        if not is_bool:
            if np.isnan(relative_error) or np.isinf(relative_error):
                line = '%d,%s,%.6f\t,%.6f\t,%.6f\t,-\n' \
                       % (index, " ".join((str(x) for x in cur_dim)), my_output_value,
                          ground_truth_value, absolute_error)
            else:
                line = '%d,%s,%.6f\t,%.6f\t,%.6f\t,%.6f\t\n' \
                       % (index, " ".join((str(x) for x in cur_dim)), my_output_value,
                          ground_truth_value, absolute_error, relative_error)
        else:
            line = '%d,%s,%s\t,%s\t,-\t,-\n' \
                   % (index, " ".join((str(x) for x in cur_dim)), my_output_value,
                      ground_truth_value)
        # write line to file
        file_stream.write(line)

    @staticmethod
    def _replace_inf_and_nan(nd_array: any) -> any:
        if np.isinf(nd_array).any():
            inf_index = np.isinf(nd_array)
            nd_array[inf_index] = np.nan
            return np.nan_to_num(nd_array)
        return np.nan_to_num(nd_array)

    @staticmethod
    def _calculate_str_length(headers: list, data: list) -> list:
        max_header_list = (len(header.strip()) for header in headers)
        max_column_list = list(max_header_list)
        for _data in data:
            max_data_list = (len(str(_element.strip())) for _element in _data)
            new_max_column_list = []
            for _element in zip(max_data_list, max_column_list):
                new_max_column_list.append(max(_element))
            max_column_list = new_max_column_list
        return max_column_list

    @staticmethod
    def _show_top_n_data(data: list, max_column_list: list) -> None:
        for _data in data:
            for index, _element in enumerate(_data):
                end = '\n' if index == len(_data) - 1 else '\t'
                print(str(_element.strip()).ljust(max_column_list[index], ' '), end=end)

    @staticmethod
    def _show_top_n_header(headers: list, max_column_list: list) -> None:
        for index, header in enumerate(headers):
            end = '\n' if index == len(headers) - 1 else '\t'
            print(str(header.strip()).ljust(max_column_list[index], ' '), end=end)

    def delete_old_detail_result_files(self: any) -> None:
        """
        Delete old detail result files
        """
        if not os.path.exists(self.output_path):
            return
        # delete old result
        for item in os.listdir(self.output_path):
            if item == ConstManager.SIMPLE_OP_MAPPING_FILE_NAME:
                continue
            if self._match_detail_file_name(item):
                path = os.path.join(self.output_path, item)
                if os.path.exists(path):
                    os.remove(path)
        self._delete_too_long_file()

    def write(self: any, dim: tuple, my_output_data: any, ground_truth_data: any, dump_file_name: str = "") -> None:
        """
        Write detail info file
        :param dim: the shape
        :param my_output_data: my output data
        :param ground_truth_data: ground truth data
        :param dump_file_name: dump file name
        """
        self.progress = Progress(len(my_output_data))
        self.dump_file_name = dump_file_name
        old_file_name = '%s.npy' % (self.detail_info.tensor_id.get_file_prefix())
        new_file_name = self._handle_too_long_file_name(old_file_name, ConstManager.NPY_SUFFIX)
        new_file_path = os.path.join(self.output_path, new_file_name)
        FileUtils.save_array_to_file(new_file_path, my_output_data.reshape(dim), np_save=True)
        log.print_write_result_info("%s after shape conversion" % self.detail_info.tensor_id.get_tensor_id(),
                                    new_file_name)
        # check whether the types of my_output_data and ground_truth_data are Boolean.
        self._check_dtype_is_bool(my_output_data, ground_truth_data)
        # transform dim shape to list
        dim_list = self._transform_dim_list(dim)
        # use numpy to calculate the summaries
        res_generator = self._array_calculate(my_output_data, ground_truth_data, dim_list, self.top_n.is_bool)
        # write summary file
        self._write_detail_summary_file(self.top_n.is_bool)
        # write each value to file
        self._write_detail_result_multi_proc(res_generator)
        # finish and print progress
        self.progress.update_and_print_progress(ConstManager.MAX_PROGRESS)
        # write and print top_n
        self._handle_top_n()
        log.print_write_result_info('details result', self.output_path)

    def _write_detail_result_multi_proc(self: any, res_generator: any) -> None:
        if self.detail_info.ignore_result:
            return
        # prepare task pool
        cpu_count = int((multiprocessing.cpu_count() + 1) / 2)
        pool = multiprocessing.Pool(cpu_count)

        # allocate all tasks evenly by multi-processing number
        task_list = []
        for _ in range(cpu_count):
            task_list.append([])
        for i, group in enumerate(res_generator):
            task_list[i % cpu_count].append(group)

        # make a listen process to listen the print progress
        comm_queue = multiprocessing.Manager().Queue()
        listen_proc = multiprocessing.Process(target=self._listen_and_print,
                                              args=(comm_queue,))
        listen_proc.start()

        # apply tasks
        for task in task_list:
            pool.apply_async(self._multi_process_group_write,
                             args=(task, comm_queue,))
        # wait till done
        pool.close()
        pool.join()
        listen_proc.terminate()

    def _listen_and_print(self: any, queue: multiprocessing.Queue) -> None:
        while True:
            add_num = queue.get(True)
            self.progress.update_progress(add_num)
            if not self.progress.is_done():
                self.progress.print_progress()

    def _separate_group_generator(self: any, dim_list: list, data_group: tuple, is_bool: bool = False) -> tuple:
        # generate the slice data of input lists
        i = 0
        length = len(dim_list)
        my_output_data = data_group[0]
        ground_truth_data = data_group[1]
        absolute_error = data_group[2]
        relative_error = data_group[3]
        max_line = self.detail_info.max_line
        while (i + 1) * max_line < length:
            left = i * max_line
            right = (i + 1) * max_line
            s = slice(left, right)
            yield i, dim_list[s], my_output_data[s], ground_truth_data[s], absolute_error[s], relative_error[s], is_bool
            i = i + 1

        s = slice(i * max_line, length)
        yield i, dim_list[s], my_output_data[s], ground_truth_data[s], absolute_error[s], relative_error[s], is_bool

    def _group_write_exec(self: any, task_group: list, queue: multiprocessing.Queue) -> None:
        for arg_group in task_group:
            task_index = arg_group[0]
            dim_list = arg_group[1]
            my_output_data = arg_group[2]
            ground_truth_data = arg_group[3]
            absolute_error = arg_group[4]
            relative_error = arg_group[5]
            # make header, should do once
            file_path = self._make_detail_output_file(task_index * self.detail_info.max_line)
            # create file IO pointer
            detail_output_file = os.fdopen(os.open(file_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                                           'w')
            detail_output_file.write(self.detail_info.make_detail_header())
        
            # write detail results one by one
            for offset, dim in enumerate(dim_list):
                index = task_index * self.detail_info.max_line + offset
                self._write_one_detail(index, detail_output_file, dim, my_output_data[offset],
                                       ground_truth_data[offset], absolute_error[offset], relative_error[offset],
                                       is_bool=arg_group[6])
        
            detail_output_file.close()
        
            # add the write count to the shared queue
            queue.put(len(dim_list))

    def _multi_process_group_write(self: any, task_group: list, queue: multiprocessing.Queue) -> None:
        try:
            self._group_write_exec(task_group, queue)
        except Exception as error:
            log.print_error_log('Failed to write detail info in subprocess. %s' % error)

    def _transform_dim_list(self: any, dim: tuple) -> list:
        if self.detail_info.detail_format != '()':
            # padding shape to 4d
            dims = self._padding_shape_to_4d(dim)
            args = (range(x) for x in dims)
            # write each detail value
            cur_dim_list = list(itertools.product(*args))
        else:
            cur_dim_list = []
            for cur_dim0 in range(dim[0]):
                cur_dim_list.append([cur_dim0])

        return cur_dim_list

    def _match_detail_file_name(self: any, file_name: str) -> bool:
        name = self.detail_info.tensor_id.get_file_prefix()
        return file_name == ("%s_summary.txt" % name) or (file_name.startswith("%s_" % name) and (
                file_name.endswith(ConstManager.CSV_SUFFIX) or file_name.endswith(ConstManager.NPY_SUFFIX)))

    def _make_detail_output_file(self: any, file_index: int = 0) -> str:
        # make file name based on the index，then return the path
        old_file_name = "%s_%d.csv" % (self.detail_info.tensor_id.get_file_prefix(), file_index)
        new_file_name = self._handle_too_long_file_name(old_file_name, ConstManager.CSV_SUFFIX, file_index)
        file_path = os.path.join(self.output_path, new_file_name)

        return file_path

    def _array_calculate(self: any, my_output_data: any, ground_truth_data: any, dim_list: list,
                         is_bool: bool = False) -> any:
        self._total_num = len(my_output_data)
        absolute_error, relative_error = self._cal_err(my_output_data, ground_truth_data, is_bool)
        # find the index of top n error values
        top_n = self.detail_info.top_n
        if len(absolute_error) < top_n:
            top_n = len(absolute_error) - 1
        absolute_top_n_index, relative_top_n_index = self._get_error_top_n_index(absolute_error, relative_error, top_n,
                                                                                 is_bool)
        # record top n lines in calculator
        self._set_top_n_list("absolute", absolute_top_n_index, dim_list,
                             (my_output_data[absolute_top_n_index], ground_truth_data[absolute_top_n_index],
                              absolute_error[absolute_top_n_index], relative_error[absolute_top_n_index]))
        self._set_top_n_list("relative", relative_top_n_index, dim_list,
                             (my_output_data[relative_top_n_index], ground_truth_data[relative_top_n_index],
                              absolute_error[relative_top_n_index], relative_error[relative_top_n_index]))

        # return the generator of compute result, to save memory
        return self._separate_group_generator(dim_list,
                                              (my_output_data, ground_truth_data, absolute_error, relative_error),
                                              is_bool)

    def _cal_err(self: any, my_output_data: any, ground_truth_data: any, is_bool: bool) -> tuple:
        if not is_bool:
            # use numpy to calculate summary data quickly
            absolute_error = np.abs(my_output_data - ground_truth_data)
            abs_ground_data = np.abs(ground_truth_data)
            relative_error = np.true_divide(absolute_error, abs_ground_data)
        else:
            absolute_error = np.array([np.nan] * self._total_num)
            relative_error = np.array([np.nan] * self._total_num)
        return absolute_error, relative_error

    def _get_error_top_n_index(self: any, absolute_error: any, relative_error: any, top_n: int, is_bool: bool) -> tuple:
        if not is_bool:
            absolute_new_error = self._replace_inf_and_nan(absolute_error)
            relative_new_error = self._replace_inf_and_nan(relative_error)
            absolute_top_n_index = np.argpartition(absolute_new_error, -top_n)[-top_n:].tolist()
            relative_top_n_index = np.argpartition(relative_new_error, -top_n)[-top_n:].tolist()
            if len(absolute_new_error) != 0:
                self.min_max_value.set_min_absolute_error(np.nanmin(absolute_new_error))
                self.min_max_value.set_max_absolute_error(np.nanmax(absolute_new_error))
            if len(relative_new_error) != 0:
                self.min_max_value.set_min_relative_error(np.nanmin(relative_new_error))
                self.min_max_value.set_max_relative_error(np.nanmax(relative_new_error))
        else:
            absolute_top_n_index = list(range(top_n))
            relative_top_n_index = list(range(top_n))

            self.min_max_value.set_min_absolute_error(np.nan)
            self.min_max_value.set_max_absolute_error(np.nan)
            self.min_max_value.set_min_relative_error(np.nan)
            self.min_max_value.set_max_relative_error(np.nan)
        return absolute_top_n_index, relative_top_n_index

    def _set_top_n_list(self: any, error_type: str, index_list: list, dim_list: list, top_n_data_group: tuple) -> None:
        top_n_dims = []
        for i in index_list:
            top_n_dims.append(dim_list[i])
        if error_type == "absolute":
            self.top_n.set_absolute_top_n_list(index_list, top_n_dims, top_n_data_group[0], top_n_data_group[1],
                                               top_n_data_group[2], top_n_data_group[3], error_type)
        else:
            self.top_n.set_relative_top_n_list(index_list, top_n_dims, top_n_data_group[0], top_n_data_group[1],
                                               top_n_data_group[2], top_n_data_group[3], error_type)

    def _check_dtype_is_bool(self: any, my_output_data: any, ground_truth_data: any) -> None:
        is_bool = my_output_data.dtype == np.bool_ and ground_truth_data.dtype == np.bool_
        self.top_n.is_bool = is_bool

    def _write_top_n_file(self: any, top_n_type: str, top_n_list: list) -> None:
        old_file_name = "%s_%s_error_topn.csv" % (self.detail_info.tensor_id.get_file_prefix(), top_n_type)
        top_n_type_suffix = "_%s_error_topn.csv" % top_n_type
        new_file_name = self._handle_too_long_file_name(old_file_name, top_n_type_suffix)
        top_n_path = os.path.join(self.output_path, new_file_name)
        path_check.check_write_path_secure(top_n_path)
        try:
            with os.fdopen(os.open(top_n_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES), 'w') as top_n_file:
                top_n_file.write(self.detail_info.make_detail_header())
                for line in top_n_list:
                    top_n_file.write(line)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as error:
            log.print_error_log('Failed to write top n file. %s' % error)
            raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from error

    def _make_show_data(self: any, top_n_list: list) -> (list, list):
        header = self.detail_info.make_detail_header().split(',')
        data = []
        for item in top_n_list:
            data.append(item.split(','))
        return header, data

    def _show_top_n(self: any, top_n_type: str, top_n_list: list) -> None:
        headers, data = self._make_show_data(top_n_list)
        max_column_list = self._calculate_str_length(headers, data)
        log.print_info_log("%s TopN" % top_n_type)
        self._show_top_n_header(headers, max_column_list)
        self._show_top_n_data(data, max_column_list)

    def _handle_top_n(self: any) -> None:
        print("\n")
        absolute_top_n = self.top_n.get_absolute_error_top_n_list()
        self._write_top_n_file('absolute', absolute_top_n)
        self._show_top_n("Absolute Error", absolute_top_n)
        print("\n")
        relative_top_n = self.top_n.get_relative_error_top_n_list()
        self._write_top_n_file('relative', relative_top_n)
        self._show_top_n("Relative Error", relative_top_n)

    def _write_detail_summary_file(self: any, is_bool: bool = False) -> None:
        content = "TotalCount:%d\n%s%s\n" \
                  % (self._total_num, self.detail_info.get_detail_info(), self.min_max_value.get_min_and_max_value())
        if is_bool:
            log.print_warn_log('Boolean data does not support calculate Relative Error or Absolute Error!')
            content = content.replace('nan', '-')
        old_file_name = "%s_summary.txt" % (self.detail_info.tensor_id.get_file_prefix())
        new_file_name = self._handle_too_long_file_name(old_file_name, ConstManager.SUMMARY_TXT_SUFFIX)
        summary_file_path = os.path.join(self.output_path, new_file_name)
        path_check.check_write_path_secure(summary_file_path)
        try:
            with os.fdopen(os.open(summary_file_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           'w') as summary_file:
                summary_file.write(content)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as error:
            log.print_error_log('Failed to write detail summary file. %s' % error)
            raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from error

    def _handle_too_long_file_name(self: any, file_name: str, suffix: str, file_index: int = 0) -> str:
        if len(file_name) >= ConstManager.LINUX_FILE_NAME_MAX_LEN:
            tensor_type_index = self.detail_info.tensor_id.get_tensor_type_index()
            new_file_name = "%s%s%s" % (self.dump_file_name.replace('/', '_').replace('.', '_'),
                                        tensor_type_index, suffix)
            if ConstManager.CSV_SUFFIX == suffix:
                new_file_name = "%s%s_%s%s" % (self.dump_file_name.replace('/', '_').replace('.', '_'),
                                               tensor_type_index, file_index, suffix)
            if len(new_file_name) >= ConstManager.LINUX_FILE_NAME_MAX_LEN:
                value = ''.join(str(uuid.uuid3(uuid.NAMESPACE_DNS, file_name)).split('-'))
                new_file_name = "%s%s" % (value, suffix)
            self._save_op_mapping_file(file_name, new_file_name)
            return new_file_name
        return file_name

    def _save_op_mapping_file(self: any, old_file_name: str, new_file_name: str) -> None:
        single_op_mapping_path = os.path.join(self.output_path, ConstManager.SIMPLE_OP_MAPPING_FILE_NAME)
        try:
            with os.fdopen(os.open(single_op_mapping_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           'a+') as mapping_file:
                content = "%s,%s\n" % (new_file_name, old_file_name)
                mapping_file.write(content)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as error:
            log.print_error_log('Failed to save mapping file. %s' % error)
            raise CompareError(CompareError.MSACCUCMP_WRITE_FILE_ERROR) from error

    def _delete_file_by_mapping(self: any, hash_to_file_name_map: dict) -> list:
        delete_keys = []
        # delete old result
        for key, value in hash_to_file_name_map.items():
            if self._match_detail_file_name(value):
                path = os.path.join(self.output_path, key)
                if os.path.exists(path):
                    os.remove(path)
                delete_keys.append(key)
        return delete_keys

    def _delete_too_long_file(self: any) -> None:
        mapping_file_path = os.path.join(self.output_path, ConstManager.SIMPLE_OP_MAPPING_FILE_NAME)
        if not os.path.exists(mapping_file_path):
            return
        hash_to_file_name_map = mapping.read_mapping_file(mapping_file_path)
        delete_keys = self._delete_file_by_mapping(hash_to_file_name_map)
        for key in delete_keys:
            hash_to_file_name_map.pop(key)
        if os.path.exists(mapping_file_path):
            os.remove(mapping_file_path)
        # save other compare result records.
        if hash_to_file_name_map:
            with os.fdopen(os.open(mapping_file_path, ConstManager.WRITE_FLAGS, ConstManager.WRITE_MODES),
                           "w") as output_file:
                for key, value in hash_to_file_name_map.items():
                    content = "%s,%s\n" % (key, value)
                    output_file.write(content)
