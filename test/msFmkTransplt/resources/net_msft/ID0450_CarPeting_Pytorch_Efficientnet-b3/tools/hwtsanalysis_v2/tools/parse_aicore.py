# -*- coding: UTF-8 -*-
import os
import struct
from tabulate import tabulate

"""
@function:
    parse aicore log.
@author: 
    w00282991, j00445069
@email:
    wangbei5@huawei.com
"""
import ascend_function


def get_43_ai_core_data(combine_file):
    result_data = list()
    with open(combine_file, 'rb') as ai_core_file:
        while True:
            _line = ai_core_file.read(128)
            if _line:
                if not _line.strip():
                    continue
            else:
                break
            _format = "BBHHHIIqqqqqqqqqqIIIIIIII"
            _result = [hex(i) for i in struct.unpack(_format, _line)]
            _byte01 = bin(int(_result[0].replace('0x', ''), 16)).replace('0b', '').zfill(8)
            ov = _byte01[-4]
            cnt = _byte01[0:4]
            task_id = int(_result[4].replace('0x', ''), 16)
            total_cyc = int(_result[7].replace('0x', ''), 16)
            ov_cyc = int(_result[8].replace('0x', ''), 16)
            pmu_cnt = tuple(int(i.replace('0x', ''), 16) for i in _result[9:17])
            stream_id = int(_result[17].replace('0x', ''), 16)
            result_data.append((ov, cnt, task_id, total_cyc, ov_cyc, stream_id, pmu_cnt))
    return result_data


def write_format(save_file_name, data_source):
    with open(save_file_name, 'a+') as f:
        f.write(data_source)
        f.write("\n")


def parse_aicore(aicore_list, temp_aicore_path, device_id, aicore_file):
    prefix = '_'.join(aicore_list[0].split('_')[:-1])
    arranged_list = list()
    for i in range(len(aicore_list)):
        arranged_list.append(prefix + '_' + str(i))
    combine_file = os.path.join(temp_aicore_path, 'aicore_combine.%s.log' % device_id)
    with open(combine_file, 'wb') as out_file:
        for bin_file in arranged_list:
            with open(bin_file, 'rb') as in_file:
                infile_data = in_file.read()
                out_file.write(infile_data)
    ai_core_data = get_43_ai_core_data(combine_file)
    write_format(aicore_file, data_source='============================43 AI core data =========================')
    write_format(aicore_file, data_source=tabulate(ai_core_data,
                                                   ['Overflow', 'cnt', 'Task id', 'Total cycles', 'overflowed cycles',
                                                    'Stream ID', 'PMU events'], tablefmt='simple'))
