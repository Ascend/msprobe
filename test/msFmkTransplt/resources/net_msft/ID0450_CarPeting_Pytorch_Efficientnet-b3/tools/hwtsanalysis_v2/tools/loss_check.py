# -*- coding: UTF-8 -*-
import os
import numpy as np
import xlsxwriter

"""
@function:
    check loss fit.
@author: 
    w00282991
@email:
    wangbei5@huawei.com
"""
import ascend_function


def cosine_similarity(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    cos = float(vector_a * vector_b.T) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    # normalize cosine from [-1 1] to [0 1]
    cos = 0.5 + 0.5 * cos
    return cos


def draw_loss_file(npu_loss, gpu_loss, excel_file):
    file_name = xlsxwriter.Workbook(excel_file)
    file_sheet = file_name.add_worksheet('LossGraph')
    first_row = ['Serial Number', 'NPU Loss List', 'GPU Loss List', 'NPU - GPU List']
    first_row_style = file_name.add_format({
        'font_name': 'Times New Roman',
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#92D050'
    })
    other_row_style = file_name.add_format({
        'font_name': 'Times New Roman',
        'bold': False
    })
    for i in range(len(first_row)):
        file_sheet.write(0, i, first_row[i], first_row_style)
        file_sheet.set_column(i, i, len(first_row[i]))
    color_list = ['green', 'blue', 'red', 'orange']
    serial_list = list(range(1, len(npu_loss) + 1))
    file_sheet.write_column(1, 0, serial_list, other_row_style)
    file_sheet.write_column(1, 1, npu_loss, other_row_style)
    file_sheet.write_column(1, 2, gpu_loss, other_row_style)
    file_sheet.write_column(1, 3, npu_loss - gpu_loss, other_row_style)
    chart_line = file_name.add_chart({'type': 'line'})
    for col in range(1, 4):
        chart_line.add_series({
            'name': ['LossGraph', 0, col],
            'categories': ['LossGraph', 1, 0, len(npu_loss), 0],
            'values': ['LossGraph', 1, col, len(npu_loss), col],
            'line': {'color': color_list[col], 'width': 1.5}
        })
    chart_line.set_title({'name': 'Loss Trend'})
    chart_line.set_x_axis({'name': "Serial Number"})
    chart_line.set_y_axis({'name': 'Value'})
    chart_line.set_size({'width': 1000, 'height': 600})
    file_sheet.insert_chart('D1', chart_line, {'x_offset': 25, 'y_offset': 0})
    file_name.close()


def loss_check(benchmark_loss_file, loss_file, loss_format, threshold, result_file):
    benchmark_loss = np.load(benchmark_loss_file)
    keyword = loss_format['keyword']
    position = loss_format['position']
    loss_list = list()
    if not os.path.exists(loss_file):
        return 1
    with open(loss_file) as train_log:
        for loss_line in train_log.readlines():
            if keyword in loss_line:
                loss_list.append(loss_line.split()[position])
    mini_num = len(loss_list) if len(loss_list) < len(benchmark_loss) else len(benchmark_loss)
    npu_loss_list = np.asanyarray(loss_list[0:mini_num], dtype=float)
    gpu_loss_list = np.asanyarray(benchmark_loss[0:mini_num], dtype=float)
    draw_loss_file(npu_loss_list, gpu_loss_list, result_file)
    try:
        cos_sim = cosine_similarity(npu_loss_list, gpu_loss_list)
        if cos_sim > threshold:
            return 0
        else:
            return 3
    except ValueError:
        return 2
