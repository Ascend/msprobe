# -*- coding: UTF-8 -*-
import struct
import os
import xlsxwriter
import xlrd

"""
@function:
    analysis training trace.
@author: 
    w00282991, j00445069
@email:
    wangbei5@huawei.com
"""


def get_timestamp(check_id_bin, bin_file, parsed_file):
    with open(parsed_file, 'a+') as output_file:
        while check_id_bin:
            check_id = struct.unpack("=Q", check_id_bin)[0]
            if check_id < 255 or check_id == 256 or check_id == 257:
                if check_id == 1:
                    output_file.write("cp_add.FP_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.FP_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.FP_start =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                elif check_id == 2:
                    output_file.write("cp_add.BP_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.BP_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.BP_end =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                elif check_id % 2 == 1 and check_id != 257:
                    output_file.write("cp_reduceadd = cp_add.all_reduces.add()")
                    output_file.write("cp_reduceadd.start_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_reduceadd.start_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_reduceadd.start =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                elif check_id % 2 == 0 and check_id != 256:
                    output_file.write("cp_reduceadd.end_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_reduceadd.end_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_reduceadd.end =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                elif check_id == 256:
                    output_file.write("cp_add.FP_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.FP_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.FP_end =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                elif check_id == 257:
                    output_file.write("cp_add.BP_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.BP_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.BP_start =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                check_id_bin = bin_file.read(struct.calcsize("=Q"))
            elif check_id == 255:
                output_file.write("cp_add.iter_stream =" + str(
                    struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                output_file.write("cp_add.iter_task =" + str(
                    struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                output_file.write("cp_add.iteration_end =" + str(
                    struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                break
            else:
                break


def parse_trace_data(trace_file, parsed_file):
    iteration = 1
    with open(trace_file, 'rb') as bin_file:
        while True:
            job_id_bin = bin_file.read(struct.calcsize("=Q"))
            if job_id_bin:
                job_id = struct.unpack("=Q", job_id_bin)[0]
                if job_id > 255:
                    _ = struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]
                    _ = struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]
                    _ = struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]
                    check_id_bin = bin_file.read(struct.calcsize("=Q"))
                    get_timestamp(check_id_bin, bin_file, parsed_file)
                    iteration += 1
                else:
                    _ = bin_file.read(struct.calcsize("=HHQ"))
                    continue
            else:
                break


def gen_timestamp(combine_file, timestamp_file, device_id, reduce_nums):
    excel_file = xlsxwriter.Workbook(timestamp_file)
    excel_sheet = excel_file.add_worksheet('device %s' % device_id)
    first_row = ['FP Start', 'BP End']
    for i in range(1, reduce_nums + 1):
        first_row.append('Reduceadd%s Start' % i)
        first_row.append('Reduceadd%s End' % i)
    first_row.append('Iteration End')
    first_row_style = excel_file.add_format({
        'font_name': 'Times New Roman',
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#92D050'
    })
    other_row_style = excel_file.add_format({
        'font_name': 'Times New Roman',
        'bold': False
    })
    for i in range(len(first_row)):
        excel_sheet.write(0, i, first_row[i], first_row_style)
        excel_sheet.set_column(i, i, len(first_row[i]))
    with open(combine_file, 'r') as pf:
        row_num = 1
        # distinguish reduceadd1 and reduceadd2, flag == 0 means reduceadd1, flag == 1 means reduceadd2
        ra_start_flag = 0
        ra_end_flag = 0
        reduce_start_col = 2
        reduce_end_col = 3
        end_col = len(first_row) - 1
        for line in pf.readlines():
            if 'FP_start' in line:
                fp_start_value = line.split('=')[-1].strip()
                excel_sheet.write(row_num, 0, float(fp_start_value), other_row_style)
            elif 'BP_end' in line:
                bp_end_value = line.split('=')[-1].strip()
                excel_sheet.write(row_num, 1, float(bp_end_value), other_row_style)
            elif 'iteration_end' in line:
                ie_end_value = line.split('=')[-1].strip()
                excel_sheet.write(row_num, end_col, float(ie_end_value), other_row_style)
                row_num += 1
            elif 'cp_reduceadd.start' in line and 'task' not in line and 'stream' not in line:
                ra_start_value = line.split('=')[-1].strip()
                if ra_start_flag < reduce_nums - 1:
                    excel_sheet.write(row_num, reduce_start_col, float(ra_start_value), other_row_style)
                    ra_start_flag += 1
                    reduce_start_col += 2
                else:
                    if reduce_nums > 0:
                        excel_sheet.write(row_num, reduce_start_col, float(ra_start_value), other_row_style)
                        ra_start_flag = 0
                        reduce_start_col = 2
            elif 'cp_reduceadd.end' in line and 'task' not in line and 'stream' not in line:
                ra_end_value = line.split('=')[-1].strip()
                if ra_end_flag < reduce_nums - 1:
                    excel_sheet.write(row_num, reduce_end_col, float(ra_end_value), other_row_style)
                    ra_end_flag += 1
                    reduce_end_col += 2
                else:
                    if reduce_nums > 0:
                        excel_sheet.write(row_num, reduce_end_col, float(ra_end_value), other_row_style)
                        ra_end_flag = 0
                        reduce_end_col = 3
    excel_file.close()


def iteration_total_time(booksheet, reduce_nums):
    total_time_list = list()
    end_col = reduce_nums * 2 + 2
    for i in range(2, booksheet.nrows):
        try:
            every_total_cycle = float(booksheet.cell(i, end_col).value) - float(booksheet.cell(i - 1, end_col).value)
        except ValueError:
            return None
        total_time_list.append(every_total_cycle / 10**5)
    return total_time_list


def iteration_interval_time(booksheet, reduce_nums):
    interval_time_list = list()
    end_col = reduce_nums * 2 + 2
    for i in range(2, booksheet.nrows):
        try:
            every_interval_cycle = float(booksheet.cell(i, 0).value) - float(booksheet.cell(i - 1, end_col).value)
        except ValueError:
            return None
        interval_time_list.append(every_interval_cycle / 10**5)
    return interval_time_list


def bp_fp_time(booksheet):
    bp_fp_time_list = list()
    for i in range(2, booksheet.nrows):
        try:
            every_bp_fp_cycle = float(booksheet.cell(i, 1).value) - float(booksheet.cell(i, 0).value)
        except ValueError:
            return None
        bp_fp_time_list.append(every_bp_fp_cycle / 10**5)
    return bp_fp_time_list


def bpend_to_iter_time(booksheet, reduce_nums):
    bpend_to_iter_time_list = list()
    end_col = reduce_nums * 2 + 2
    for i in range(2, booksheet.nrows):
        try:
            every_bpend_to_iter_cycle = float(booksheet.cell(i, end_col).value) - float(booksheet.cell(i, 1).value)
        except ValueError:
            return None
        bpend_to_iter_time_list.append(every_bpend_to_iter_cycle / 10**5)
    return bpend_to_iter_time_list


def allreduce1_start_time(booksheet):
    allreduce1_start_time_list = list()
    for i in range(2, booksheet.nrows):
        try:
            every_allreduce1_start_cycle = float(booksheet.cell(i, 2).value) - float(booksheet.cell(i, 0).value)
        except ValueError:
            return None
        allreduce1_start_time_list.append(every_allreduce1_start_cycle / 10**5)
    return allreduce1_start_time_list


def reduceadd_time(booksheet, reduce_id):
    reduceadd_time_list = list()
    start_col = reduce_id * 2
    end_col = reduce_id * 2 + 1
    for i in range(2, booksheet.nrows):
        try:
            every_reduceadd1_cycle = float(booksheet.cell(i, end_col).value) - float(booksheet.cell(i, start_col).value)
        except ValueError:
            return None, None
        reduceadd_time_list.append(every_reduceadd1_cycle / 10**5)
    return reduceadd_time_list


def reduce1end_to_bpend_time(booksheet):
    reduce1end_to_bpend_list = list()
    for i in range(2, booksheet.nrows):
        try:
            every_reduce1end_to_bpend_cycle = float(booksheet.cell(i, 1).value) - float(booksheet.cell(i, 3).value)
        except ValueError:
            return None
        reduce1end_to_bpend_list.append(every_reduce1end_to_bpend_cycle / 10**5)
    return reduce1end_to_bpend_list


def bpend_to_lastreduce_time(booksheet, reduce_nums):
    bpend_to_lastreduce_list = list()
    start = reduce_nums * 2 + 1
    for i in range(2, booksheet.nrows):
        try:
            every_bpend_to_lastreduce_cycle = float(booksheet.cell(i, start).value) - float(booksheet.cell(i, 1).value)
        except ValueError:
            return None
        bpend_to_lastreduce_list.append(every_bpend_to_lastreduce_cycle / 10**5)
    return bpend_to_lastreduce_list


def parse_trace(training_trace_list, temp_trace_path, device_id, reduce_nums, result_file):
    slice_num = 0
    parsed_list = list()
    for trace_file in training_trace_list:
        parsed_file = os.path.join(temp_trace_path, 'parsed_trace.%s.slice_%s.log' % (device_id, slice_num))
        parse_trace_data(trace_file, parsed_file)
        parsed_list.append(parsed_file)
        slice_num += 1
    combine_file = os.path.join(temp_trace_path, 'parsed_trace_combine.%s.log' % device_id)
    with open(combine_file, 'wb') as out_file:
        for parsed_file in parsed_list:
            with open(parsed_file, 'rb') as in_file:
                infile_data = in_file.read()
                out_file.write(infile_data)
    timestamp_file = os.path.join(temp_trace_path, 'parsed_trace_timestamp.%s.xlsx' % device_id)
    gen_timestamp(combine_file, timestamp_file, device_id, reduce_nums)
    workbook = xlrd.open_workbook(filename=timestamp_file)
    booksheet = workbook.sheet_by_index(0)
    iteration_list = iteration_total_time(booksheet, reduce_nums)
    interval_list = iteration_interval_time(booksheet, reduce_nums)
    bp_fp_list = bp_fp_time(booksheet)
    bpend_to_iter_list = bpend_to_iter_time(booksheet, reduce_nums)
    reduce_dict = dict()
    allreduce1_start_list = list()
    reduce1end_to_bpend_list = list()
    bpend_to_lastreduce_list = list()
    if reduce_nums > 0:
        for i in range(1, reduce_nums + 1):
            reduce_dict.update({i: reduceadd_time(booksheet, i)})
        allreduce1_start_list = allreduce1_start_time(booksheet)
        reduce1end_to_bpend_list = reduce1end_to_bpend_time(booksheet)
        bpend_to_lastreduce_list = bpend_to_lastreduce_time(booksheet, reduce_nums)
    excel_file = xlsxwriter.Workbook(result_file)
    excel_sheet = excel_file.add_worksheet('iteration dashboard')
    first_row = ['Serial Number', 'Iteration Time', 'Interval Time', 'FPStart to BPEnd Time', 'BPEnd to IterEnd Time']
    for i in range(1, reduce_nums + 1):
        first_row.append('Reduceadd%s Time' % i)
    if reduce_nums != 0:
        first_row.append('FPStart to Reduce1 Start Time')
        first_row.append('Reduce1End to BPEnd Time')
        first_row.append('BPEnd to LastReduceStart Time')
    first_row_style = excel_file.add_format({
        'font_name': 'Times New Roman',
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#92D050'
    })
    other_row_style = excel_file.add_format({
        'font_name': 'Times New Roman',
        'bold': False
    })
    for i in range(len(first_row)):
        excel_sheet.write(0, i, first_row[i], first_row_style)
        excel_sheet.set_column(i, i, len(first_row[i]))
    serial_list = list(range(1, len(bpend_to_iter_list) + 1))
    excel_sheet.write_column(1, 0, serial_list, other_row_style)
    excel_sheet.write_column(1, 1, iteration_list, other_row_style)
    excel_sheet.write_column(1, 2, interval_list, other_row_style)
    excel_sheet.write_column(1, 3, bp_fp_list, other_row_style)
    excel_sheet.write_column(1, 4, bpend_to_iter_list, other_row_style)
    if reduce_nums != 0:
        for i in range(1, reduce_nums + 1):
            excel_sheet.write_column(1, 4 + i, reduce_dict[i], other_row_style)
        excel_sheet.write_column(1, 5 + reduce_nums, allreduce1_start_list, other_row_style)
        excel_sheet.write_column(1, 6 + reduce_nums, reduce1end_to_bpend_list, other_row_style)
        excel_sheet.write_column(1, 7 + reduce_nums, bpend_to_lastreduce_list, other_row_style)
    excel_file.close()
