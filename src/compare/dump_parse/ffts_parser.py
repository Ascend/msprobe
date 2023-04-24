
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
import numpy as np

from cmp_utils import log
from dump_parse.dump_data_object import DumpTensor, DumpDataObj
from cmp_utils.constant.compare_error import CompareError


class FFTSParser:
    """
    The class for FFTS mode type parser
    """
    def __init__(self, dump_file_list, dump_data_list):
        self.dump_file_list = dump_file_list
        self.dump_data_list = dump_data_list

    @property
    def parse_ffts(self: any) -> tuple:
        """
        parse the ffts mode dump data and merge data
        @return: file path, dump data
        """
        dump_base = self.dump_data_list[0]
        thread_num = dump_base.get_thread_num
        if self.check_file_missing(thread_num):
            dump_base.ffts_file_check = False
            log.print_warn_log(
                f"This is a FFTS+ mode dump data {dump_base.op_name},"
                f" The number of files does not match the number of thread (instance slice num).")
        if dump_base.get_ffts_mode:
            cut_axis = dump_base.get_cut_axis_auto
        else:
            cut_axis = dump_base.get_cut_axis_manual
        if not cut_axis or self.check_invalid_cut_axis(cut_axis):
            msg = "The cut axis of Dump data is invalid. The files can not be merged. " \
                  "Please check the files {}".format(",".join(self.dump_file_list))
            log.print_warn_log(msg)
            raise CompareError(CompareError.MSACCUCMP_INVALID_SLICE_DATA, msg)
        else:
            output_num = len(dump_base.output_data)
            output_data_list = [dump_data.get_output_data for dump_data in self.dump_data_list]

            dump_data_output_list = []
            for i in range(output_num):
                output_index = []
                for output in output_data_list:
                    output_index.append(output[i])
                dump_data_output_list.append(output_index)

            merge_output = self.merge_data(dump_data_output_list, cut_axis)
            dump_data = self.create_merge_dump_data(dump_base, merge_output)
            file_path = '.'.join(self.dump_file_list[0].split(".")[:4] + ['*'])
            log.print_info_log(f"This is a FFTS+ mode dump data {dump_base.op_name}, "
                               f"output data has been merged, new file path is {file_path}")
        return file_path, dump_data

    @staticmethod
    def check_invalid_cut_axis(cut_axis: list) -> bool:
        """
        check if the cut axis is valid
        @param cut_axis: cut axis
        @return: True or False
        """
        return all(dim == [] for dim in cut_axis)

    @staticmethod
    def merge_data(output_list: list, cut_axis: list) -> list:
        merge_output = []
        for index, dim in enumerate(cut_axis):
            if not dim:
                merge_output.append(output_list[index][0])
            else:
                axis = cut_axis[index][0]
                merge_output.append(np.concatenate(output_list[index], axis))
        return merge_output

    @staticmethod
    def create_merge_dump_data(dump_base: DumpDataObj, merge_output: list) -> DumpDataObj:
        dump_data = DumpDataObj()
        dump_data.set_op_attr(dump_base.op_name, dump_base.ffts_file_check)
        for index, data in enumerate(merge_output):
            shape = list(data.shape)
            common_attr = dump_base.output_data[index].get_common_attr
            dump_tensor = DumpTensor(index=index, data=data.reshape(-1), shape=shape,
                                     data_type=common_attr[0], tensor_format=common_attr[1],
                                     address=common_attr[2], original_shape=common_attr[3])
            dump_data.output_data.append(dump_tensor)
        return dump_data

    def check_file_missing(self, thread_num: int) -> bool:
        """
        check if file number match thread number
        @param thread_num: thread number
        @return: True of False
        """
        return len(self.dump_data_list) != thread_num
