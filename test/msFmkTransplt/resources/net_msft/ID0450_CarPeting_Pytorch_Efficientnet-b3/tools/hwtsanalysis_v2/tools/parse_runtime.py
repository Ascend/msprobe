# -*- coding: UTF-8 -*-
import os
import struct
from tabulate import tabulate

"""
@function:
    parse runtime log.
@author: 
    w00282991, j00445069
@email:
    wangbei5@huawei.com
"""
import ascend_function


def get_first_runtime_task_trace(combine_file):
    result_data = list()
    task_kernel = list()
    _format = "BBHIQHHHHII"
    _format_last = "B"
    with open(combine_file, 'rb') as bin_data:
        while True:
            _line = bin_data.read(288)
            if _line:
                if not _line.strip():
                    continue
            else:
                break
            if len(_line) == 288:
                _unpack_tuple = struct.unpack(_format, _line[0:32])
                _char_string = _line[32:287].decode().strip(b'\x00'.decode())
                _result_last = [hex(i) for i in struct.unpack(_format_last, _line[287:288])]
                _byte01 = bin(int(_result_last[0].replace('0x', ''), 16)).replace('0b', '').zfill(8)
                persistent_1bit = _byte01[-1]
                reserved_7bit = _byte01[0:7]
                kernel_name = _char_string
                if kernel_name:
                    result_data.append((_unpack_tuple[0], _unpack_tuple[1], _unpack_tuple[2], _unpack_tuple[3],
                                        _unpack_tuple[4], _unpack_tuple[5], _unpack_tuple[6], _unpack_tuple[7],
                                        _unpack_tuple[8], _unpack_tuple[9], _unpack_tuple[10],
                                        kernel_name, persistent_1bit, reserved_7bit))
                    task_kernel.append((_unpack_tuple[8], kernel_name))
    return result_data, task_kernel


def write_format(save_file_name, data_source):
    with open(save_file_name, 'a+') as f:
        f.write(data_source)
        f.write("\n")


def parse_runtime(runtime_list, temp_aicore_path, device_id, runtime_file):
    prefix = '_'.join(runtime_list[0].split('_')[:-1])
    arranged_list = list()
    for i in range(len(runtime_list)):
        arranged_list.append(prefix + '_' + str(i))
    combine_file = os.path.join(temp_aicore_path, 'runtime_combine.%s.log' % device_id)
    with open(combine_file, 'wb') as out_file:
        for bin_file in arranged_list:
            with open(bin_file, 'rb') as in_file:
                infile_data = in_file.read()
                out_file.write(infile_data)
    runtime_task_trace_data, task_kernel = get_first_runtime_task_trace(combine_file)
    write_format(runtime_file, data_source='====================first runtime task trace data==================')
    write_format(runtime_file, data_source=tabulate(runtime_task_trace_data,
                                                    ['mode', 'rpttype', 'bufsize', 'reserved', 'timestamp', 'eventname',
                                                     'tasktype', 'streamid', 'task_id', 'thread', 'device_id',
                                                     'kernelname', 'persistant_1bit', 'reserved_7bit'],
                                                    tablefmt='simple'))
