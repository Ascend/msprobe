
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
import os
import numpy as np

from cmp_utils import log
from dump_parse.dump_data_object import DumpTensor, DumpDataObj
from dump_parse import dump


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
        if len(self.dump_file_list) == 1:
            dump_data = self.dump_data_list[-1]
            file_path = self.dump_file_list[-1]
            dump_data.set_op_attr(dump_data.op_name, True)
            return file_path, dump_data
        dump_base = self.dump_data_list[0]
        thread_num = dump_base.get_thread_num
        if self.check_file_missing(thread_num):
            dump_base.ffts_file_check = False
            log.print_warn_log(
                f"This is a FFTS+ mode dump data {dump_base.op_name},"
                f" The number of files does not match the number of thread (instance slice num).")

        cut_axis = dump_base.get_cut_axis_auto if dump_base.get_ffts_mode else dump_base.get_cut_axis_manual

        if not cut_axis or self.check_invalid_cut_axis(cut_axis):
            dump_data_to_file = list(zip(self.dump_data_list, self.dump_file_list))
            dump_data_to_file.sort(key=lambda x: os.path.basename(x[1]).split(".")[4])
            file_path = dump_data_to_file[-1][1]
            dump_data = dump_data_to_file[-1][0]
            log.print_warn_log("The cut axis of Dump data is invalid. The current compare dump file is {}. "
                               "All dump files are {}".format(dump_data_to_file[-1][1], ",".join(self.dump_file_list)))
        else:
            output_num = len(dump_base.output_data)
            if dump_base.get_ffts_mode:
                output_data_list = []
                for index, dump_file in enumerate(self.dump_data_list):
                    output_shape = dump_file.calculate_auto_mode_shape(index, "output")
                    output_data_list.append(dump_file.get_auto_output_data(output_shape))
            else:
                output_data_list = [dump_data.get_output_data() for dump_data in self.dump_data_list]

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
        op_name = dump.process_op_name(dump_base.op_name) if dump_base.op_name else ""
        dump_data.set_op_attr(op_name, dump_base.ffts_file_check)
        for index, data in enumerate(merge_output):
            shape = list(data.shape)
            common_attr = dump_base.output_data[index].get_common_attr
            dump_tensor = DumpTensor(index=index, data=data.reshape(-1), shape=shape,
                                     data_type=common_attr[0], tensor_format=common_attr[1],
                                     address=common_attr[2], original_shape=common_attr[3], is_ffts=True)
            dump_data.output_data.append(dump_tensor)
        dump_data.input_data = dump_base.input_data
        return dump_data

    def check_file_missing(self, thread_num: int) -> bool:
        """
        check if file number match thread number
        @param thread_num: thread number
        @return: True of False
        """
        return len(self.dump_data_list) != thread_num
