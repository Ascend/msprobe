# -*- coding: UTF-8 -*-
import os
import struct
from tabulate import tabulate

"""
@function:
    parse hwts log.
@author: 
    w00282991, j00445069
@email:
    wangbei5@huawei.com
"""


def get_45_hwts_log(combine_file):
    _format = ['QIIIIIIIIIIII', 'QIIQIIIIIIII', 'IIIIQIIIIIIII']
    log_type = ['Start of task', 'End of task', 'Start of block', 'End of block', 'Block PMU']
    _type_1 = list()
    with open(combine_file, 'rb') as hwts_data:
        while True:
            _line = hwts_data.read(64)
            if _line:
                if not _line.strip():
                    continue
            else:
                break
            byte_first_four = struct.unpack('BBHHH', _line[0:8])
            byte_first = bin(byte_first_four[0]).replace('0b', '').zfill(8)
            _type = byte_first[-3:]
            is_warn_res0_ov = byte_first[4]
            cnt = int(byte_first[0:4], 2)
            core_id = byte_first_four[1]
            blk_id, task_id = byte_first_four[3], byte_first_four[4]
            if _type in ['000', '001', '010']:
                _result = struct.unpack(_format[0], _line[8:])
                syscnt = _result[0]
                stream_id = _result[1]
                _type_1.append((log_type[int(_type, 2)], cnt, core_id, blk_id, task_id, syscnt, stream_id))
            elif _type == '011':
                _result = struct.unpack(_format[1], _line[8:])
                syscnt = _result[0]
                stream_id = _result[1]
                _type_1.append((log_type[int(_type, 2)], cnt, core_id, blk_id, task_id, syscnt, stream_id))
            elif _type == '100':
                _result = struct.unpack(_format[2], _line[8:])
                stream_id = _result[2]
                if is_warn_res0_ov == '0':
                    total_cyc = _result[4]
                else:
                    total_cyc = None
                _type_1.append((log_type[int(_type, 2)], cnt, core_id, blk_id, task_id, total_cyc, stream_id))
    return _type_1


def write_format(save_file_name, data_source):
    with open(save_file_name, 'a+') as f:
        f.write(data_source)
        f.write("\n")


def parse_hwts(hwts_list, temp_hwts_path, device_id, hwts_file):
    prefix = '_'.join(hwts_list[0].split('_')[:-1])
    arranged_list = list()
    for i in range(len(hwts_list)):
        arranged_list.append(prefix + '_' + str(i))
    combine_file = os.path.join(temp_hwts_path, 'hwts_combine.%s.log' % device_id)
    with open(combine_file, 'wb') as out_file:
        for bin_file in arranged_list:
            with open(bin_file, 'rb') as in_file:
                infile_data = in_file.read()
                out_file.write(infile_data)
    hwts_data = get_45_hwts_log(combine_file)
    write_format(hwts_file, data_source='============================45 HWTS data ============================')
    new_data = list()
    for i in hwts_data:
        if i[4] in [8969, 8974]:
            if i[0] == 'Start of task' or i[0] == 'End of task':
                new_data.append((i[0], i[4], i[5]))
    write_format(hwts_file, data_source=tabulate(hwts_data,
                                                 ['Type', 'cnt', 'Core ID', 'Block ID', 'Task ID',
                                                  'Cycle counter', 'Stream ID'],
                                                 tablefmt='simple'))
