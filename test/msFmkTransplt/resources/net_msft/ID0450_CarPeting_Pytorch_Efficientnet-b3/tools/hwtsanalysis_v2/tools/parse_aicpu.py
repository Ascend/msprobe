# -*- coding: UTF-8 -*-
import xlsxwriter

"""
@function:
    parse aicpu op.
@author: 
    w00282991, w00451502, j00445069
@email:
    wangbei5@huawei.com
"""
import ascend_function


def parse_aicpu(aicpu_list, result_file, start_aicpu_name):
    prefix = '_'.join(aicpu_list[0].split('_')[:-1])
    arranged_list = list()
    for i in range(len(aicpu_list)):
        arranged_list.append(prefix + '_' + str(i))
    aicpu_lines = list()
    for aicpu_file in arranged_list:
        with open(aicpu_file, 'rb') as aicpu_data:
            aicpu_lines += str(aicpu_data.read().replace(b'\n\x00', b' ___ ')
                               .replace(b'\x00', b' ___ '))[2:-1].split(" ___ ")
    iteration = 0
    excel_file = xlsxwriter.Workbook(result_file)
    excel_sheet = excel_file.add_worksheet('aicpu op dashboard')
    first_row = ['serial_number',
                 'kernel_name',
                 'dispatch_time(us)',
                 'total_time(us)',
                 'run_v2_start',
                 'compute_start',
                 'memcpy_start',
                 'memcpy_end',
                 'run_v2_end']
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
    iteration += 1
    creat_session_cnt = 0
    for line in aicpu_lines:
        if "Create session start" in line:
            creat_session_cnt += 1
    start_idx = creat_session_cnt * 2
    aicpu_lines = aicpu_lines[start_idx:-3]
    thread_list = list()
    node_list = list()
    iter2_flag = 0
    for line in aicpu_lines:
        if "Thread" in line:
            total_time = line.split(',')[-1].split('=')[-1].split()[0]
            dispatch_time = line.split(',')[-2].split('=')[-1].split()[0]
            thread_list.append([total_time, dispatch_time])
        elif "Node" in line:
            node_name = line.split(',')[0].split(':')[-1]
            run_v2_start = line.split(',')[1].split(':')[-1]
            compute_start = line.split(',')[2].split(':')[-1]
            memcpy_start = line.split(',')[3].split(':')[-1]
            memcpy_end = line.split(',')[4].split(':')[-1]
            run_v2_end = line.split(',')[5].split(':')[-1]
            node_data = [node_name, run_v2_start, compute_start, memcpy_start, memcpy_end, run_v2_end]
            node_list.append(node_data)
    row_num = 1
    for iteration in range(0, len(thread_list)):
        node_data = node_list[iteration]
        thread_data = thread_list[iteration]
        if start_aicpu_name in node_data[0] and iter2_flag <= 3:
            iter2_flag += 1
        if iter2_flag == 3:
            excel_sheet.write(row_num, 0, iteration, other_row_style)
            excel_sheet.write(row_num, 1, node_data[0], other_row_style)
            excel_sheet.write(row_num, 2, thread_data[1], other_row_style)
            excel_sheet.write(row_num, 3, thread_data[0], other_row_style)
            excel_sheet.write(row_num, 4, node_data[1], other_row_style)
            excel_sheet.write(row_num, 5, node_data[2], other_row_style)
            excel_sheet.write(row_num, 6, node_data[3], other_row_style)
            excel_sheet.write(row_num, 7, node_data[4], other_row_style)
            excel_sheet.write(row_num, 8, node_data[5], other_row_style)
            row_num += 1
    excel_file.close()
