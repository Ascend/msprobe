# -*- coding: UTF-8 -*-
import xlsxwriter
import xlrd
from string import digits
import os

"""
@function:
    parse aicore op.
@author: 
    w00282991
@email:
    wangbei5@huawei.com
"""
import ascend_function


def get_op_info(kernel_name, file_name):
    with open(file_name, 'r') as info_file:
        for line in info_file.readlines():
            if kernel_name in line:
                line_list = line.split('|')
                result = [line_list[1], line_list[2], line_list[3], line_list[4],
                          line_list[5], line_list[6], line_list[7]]
                return result


def op_type_judge(cube_time, vector_time, scalar_time, mte_time):
    max_value = cube_time
    for i in (vector_time, scalar_time, mte_time):
        if max_value < i:
            max_value = i
    if max_value == cube_time:
        op_type = 'cube bound'
    elif max_value == vector_time:
        op_type = 'vector bound'
    elif max_value == scalar_time:
        op_type = 'scalar bound'
    else:
        op_type = 'mte bound'
    return op_type


def match_kernel_name(runtime_file):
    iteration = 0
    stream_id_in_rt = 7
    task_id_in_rt = 8
    kernel_name_in_rt = 11
    kernel_dict = dict()
    with open(runtime_file) as f_rt:
        for rt_line in f_rt.readlines():
            if iteration < 3:
                iteration += 1
            else:
                iteration += 1
                rt_line_2_list = rt_line.split()
                try:
                    kernel_dict.update({
                        str(rt_line_2_list[task_id_in_rt]) + str(rt_line_2_list[stream_id_in_rt]):
                            rt_line_2_list[kernel_name_in_rt]})
                except IndexError:
                    pass
    return kernel_dict


def match_hwts_time(hwts_file):
    iteration = 0
    start_dict = dict()
    hwts_time_list = list()
    with open(hwts_file) as f_hwts:
        for hwts_line in f_hwts.readlines():
            if iteration < 3:
                iteration += 1
            else:
                iteration += 1
                hwts_list = hwts_line.split()
                len_hwts_line = len(hwts_list)
                if 'Start of task' in hwts_line:
                    start_dict.update({
                        str(hwts_list[len_hwts_line - 3]) + str(hwts_list[-1]): hwts_list[len_hwts_line - 2]
                    })
                if 'End of task' in hwts_line:
                    end_task_id = str(hwts_list[len_hwts_line - 3]) + str(hwts_list[-1])
                    end_time = hwts_list[len_hwts_line - 2]
                    end_stream_id = hwts_list[len_hwts_line - 1]
                    if end_task_id in start_dict.keys():
                        start_time = start_dict[end_task_id]
                        cost_time = (float(end_time) - float(start_time)) / 10 ** 5
                        hwts_time_list.append([end_task_id, start_time, end_time, cost_time, end_stream_id])
                        start_dict.pop(end_task_id)
    return hwts_time_list


def match_aicore_time(aicore_file):
    iteration = 0
    task_id_position = 2
    total_cycle_position = 3
    stream_id_position = 5
    pmu_events_position = 6
    aicore_time_list = list()
    with open(aicore_file) as f_aicore:
        for aicore_line in f_aicore.readlines():
            if iteration < 3:
                iteration += 1
            else:
                iteration += 1
                aicore_list = aicore_line.split()
                task_id = str(aicore_list[task_id_position]) + str(aicore_list[stream_id_position])
                total_cycle = aicore_list[total_cycle_position]
                pmu_events = aicore_list[pmu_events_position:]
                aicore_time_list.append([task_id, total_cycle, pmu_events])
    return aicore_time_list


def get_benchmark_op_time(kernel_name, benchmark_op_file):
    if os.path.isfile(benchmark_op_file):
        with open(benchmark_op_file) as bf:
            for line in bf.readlines():
                if kernel_name in line:
                    return float(line.split(',')[-1])
    else:
        return 0.0


def gen_op_excel(hwts_file, runtime_file, aicore_file, end_op_name, enable_pmu,
                 enable_op_info, op_info_file, result_file, benchmark_op_file):
    kernel_name_dict = match_kernel_name(runtime_file)
    hwts_time_list = match_hwts_time(hwts_file)
    aicore_time_list = list()
    excel_file = xlsxwriter.Workbook(result_file)
    excel_sheet = excel_file.add_worksheet('aicore op dashboard')
    first_row = ['serial_number', 'task_id', 'stream_id', 'kernel_name', 'task_time',
                 'task_start', 'task_stop', 'task_wait', 'benchmark_time', 'difference']
    if enable_op_info:
        first_row += ['block_dim', 'input_shape', 'input_dtype',
                      'input_format', 'output_shape', 'output_dtype', 'output_format']
    if enable_pmu:
        first_row += ['PMU_events', 'op_type', 'cube_cycle', 'cube_percent', 'vector_cycle',
                      'vector_percent', 'scalar_cycle', 'scalar_percent', 'mte1_cycle', 'mte2_cycle',
                      'mte3_cycle', 'mte_percent', 'l2read_cycle', 'l2read_percent', 'l2write_cycle',
                      'l2write_percent', 'total_cycle']
        aicore_time_list = match_aicore_time(aicore_file)
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
    # get iter2 op information.
    iter2_flag = 0
    iteration = 0
    for hwts_info in hwts_time_list:
        task_id = hwts_info[0]
        if task_id in kernel_name_dict.keys():
            if iter2_flag == 1:
                task_start = hwts_info[1]
                task_end = hwts_info[2]
                task_time = hwts_info[3]
                stream_id = hwts_info[4]
                iteration += 1
                excel_sheet.write(iteration, 0, iteration, other_row_style)
                excel_sheet.write(iteration, 1, task_id, other_row_style)
                excel_sheet.write(iteration, 2, stream_id, other_row_style)
                if kernel_name_dict[task_id].split('_', 1)[0].isdigit():
                    temp_kernel_name = kernel_name_dict[task_id].split('_', 1)[1] + '_tvmbin'
                    if temp_kernel_name.split('_', 1)[0].isdigit():
                        temp_kernel_name = temp_kernel_name.split('_', 1)[1]
                else:
                    temp_kernel_name = kernel_name_dict[task_id] + '_tvmbin'
                # avoid the mistake in ge topo for layernorm op name.
                if temp_kernel_name.split('/')[-1] == '_tvmbin':
                    temp_kernel_name = temp_kernel_name .split('_tvmbin')[0] + 'Layernorm_tvmbin'
                excel_sheet.write(iteration, 3, temp_kernel_name, other_row_style)
                excel_sheet.write(iteration, 4, task_time, other_row_style)
                excel_sheet.write(iteration, 5, task_start, other_row_style)
                excel_sheet.write(iteration, 6, task_end, other_row_style)
                excel_formula = '=(F%s-G%s)/100000' % (iteration + 1, iteration)
                if iteration > 1:
                    excel_sheet.write_formula(iteration, 7, excel_formula, other_row_style)
                else:
                    excel_sheet.write(iteration, 7, 0, other_row_style)
                benchmark_op_time = get_benchmark_op_time(temp_kernel_name, benchmark_op_file)
                excel_sheet.write(iteration, 8, benchmark_op_time, other_row_style)
                excel_sheet.write(iteration, 9, task_time - benchmark_op_time, other_row_style)
                opt_fun_col = 9
                if enable_op_info:
                    op_info = get_op_info(temp_kernel_name, op_info_file)
                    excel_sheet.write(iteration, opt_fun_col + 1, op_info[0], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 2, op_info[1], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 3, op_info[2], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 4, op_info[3], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 5, op_info[4], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 6, op_info[5], other_row_style)
                    excel_sheet.write(iteration, opt_fun_col + 7, op_info[6], other_row_style)
                    opt_fun_col = opt_fun_col + 7
                if enable_pmu:
                    aicore_iter2_flag = 0
                    for aicore_info in aicore_time_list:
                        if aicore_info[0] == task_id and aicore_iter2_flag == 0:
                            aicore_iter2_flag += 1
                        elif aicore_info[0] == task_id and aicore_iter2_flag == 1:
                            total_cycle = int(aicore_info[1])
                            pmu_str = "".join(aicore_info[2])
                            excel_sheet.write(iteration, opt_fun_col + 1, pmu_str, other_row_style)
                            pmu_list = pmu_str.split('(')[-1].split(')')[0].split(',')
                            [cube_cycle, vector_cycle, scalar_cycle, mte1_cycle, mte2_cycle, mte3_cycle,
                             l2read_cycle, l2write_cycle] = map(lambda x: int(x), pmu_list)
                            max_mte_cycle = max(mte1_cycle, mte2_cycle, mte3_cycle)
                            op_type = op_type_judge(cube_cycle, vector_cycle, scalar_cycle, max_mte_cycle)
                            excel_sheet.write(iteration, opt_fun_col + 2, op_type, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 3, cube_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 4, cube_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 5, vector_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 6, vector_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 7, scalar_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 8, scalar_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 9, mte1_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 10, mte2_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 11, mte3_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 12, max_mte_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 13, l2read_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 14, l2read_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 15, l2write_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 16, l2write_cycle / total_cycle, other_row_style)
                            excel_sheet.write(iteration, opt_fun_col + 17, total_cycle, other_row_style)
                            aicore_iter2_flag += 1
                        elif aicore_info[0] == task_id and aicore_iter2_flag == 2:
                            break
            if end_op_name in kernel_name_dict[task_id] and iter2_flag == 0:
                iter2_flag += 1
            elif end_op_name in kernel_name_dict[task_id] and iter2_flag == 1:
                iter2_flag += 1
    excel_file.close()


def gen_statistics_excel(result_file, statistics_file):
    op_file = xlrd.open_workbook(result_file)
    op_sheet = op_file.sheet_by_name('aicore op dashboard')
    kernel_name = op_sheet.col_values(3)
    task_time = op_sheet.col_values(4)
    excel_file = xlsxwriter.Workbook(statistics_file)
    excel_sheet = excel_file.add_worksheet('aicore op statistics dashboard')
    first_row = ['serial_number', 'kernel_name', 'occurrences', 'total_time', 'max_time', 'min_time', 'mean_time']
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
    num_row_style = excel_file.add_format({
        'font_name': 'Times New Roman',
        'bold': False
    })
    num_row_style.set_num_format(9)
    count = 0
    exist_kernel_names = list()
    for i in range(len(first_row)):
        excel_sheet.write(0, i, first_row[i], first_row_style)
        excel_sheet.set_column(i, i, len(first_row[i]))
    count += 1
    for i in range(1, op_sheet.nrows):
        kernel_name_i = kernel_name[i].split('/')[-1]
        name_list_i = kernel_name_i.split('_')
        for x in range(len(name_list_i)):
            if name_list_i[x].isdigit():
                del name_list_i[x]
                break
        kernel_name_i = '_'.join(name_list_i).translate(str.maketrans('', '', digits))
        if kernel_name_i not in exist_kernel_names:
            exist_kernel_names.append(kernel_name_i)
            total_task_time = 0
            kernel_count = 0
            tmp_max_task_time = 0
            tmp_min_task_time = 100000
            for j in range(i, op_sheet.nrows):
                name_list_j = kernel_name[j].split('_')
                for y in range(len(name_list_j)):
                    if name_list_j[y].isdigit():
                        del name_list_j[y]
                        break
                kernel_name_j = '_'.join(name_list_j).translate(str.maketrans('', '', digits)).split('/')[-1]
                if kernel_name_i == kernel_name_j:
                    total_task_time += task_time[j]
                    tmp_max_task_time = tmp_max_task_time if tmp_max_task_time > task_time[j] else task_time[j]
                    tmp_min_task_time = tmp_min_task_time if tmp_min_task_time < task_time[j] else task_time[j]
                    kernel_count += 1
            excel_sheet.write(count, 0, count, other_row_style)
            excel_sheet.write(count, 1, kernel_name_i, other_row_style)
            excel_sheet.write(count, 2, kernel_count, other_row_style)
            excel_sheet.write(count, 3, total_task_time, other_row_style)
            excel_sheet.write(count, 4, tmp_max_task_time, other_row_style)
            excel_sheet.write(count, 5, tmp_min_task_time, other_row_style)
            excel_sheet.write(count, 6, total_task_time / kernel_count, other_row_style)
            count += 1
    excel_file.close()


def parse_op(hwts_file, runtime_file, aicore_file, end_op_name, enable_pmu,
             enable_statistics, enable_op_info, op_info_file, result_file, statistics_file, benchmark_op_file):
    gen_op_excel(hwts_file, runtime_file, aicore_file, end_op_name, enable_pmu, enable_op_info, op_info_file,
                 result_file, benchmark_op_file)
    if enable_statistics:
        gen_statistics_excel(result_file, statistics_file)
