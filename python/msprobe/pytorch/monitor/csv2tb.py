# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=duplicate-code

import datetime
import os
import re
from multiprocessing import Process

import pytz
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import read_csv, create_directory, remove_path
from msprobe.core.common.utils import check_process_num
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.monitor.utils import get_target_output_dir
from msprobe.pytorch.common.log import logger


all_data_type_list = [
    "actv",
    "actv_grad",
    "exp_avg",
    "exp_avg_sq",
    "grad_unreduced",
    "grad_reduced",
    "param_origin",
    "param_updated",
]
CSV_FILE_SUFFIX = r"_\d+-\d+\.csv"


def parse_step_line(line, ops):
    vp_id = line["vpp_stage"]
    module_name = line[MonitorConst.HEADER_NAME]
    step = line["step"]
    vpp_name = f"vp{vp_id}:{module_name}"
    if 'micro_step' in line:
        vpp_name = f'{vpp_name}{MonitorConst.NAME_SEP}micro{line["micro_step"]}'
    ops_result = {}
    for op in ops:
        ops_result[op] = line[op]
    return vpp_name, step, ops_result


def parse_step_fn(filepath):
    data = read_csv(filepath)
    ops = [k for k in data.keys() if k in MonitorConst.OP_LIST[:-2]]
    parse_step_result = {}

    for _, line in data.iterrows():
        vpp_name, step, ops_result = parse_step_line(line, ops)
        if vpp_name not in parse_step_result:
            parse_step_result[vpp_name] = {}
        if step in parse_step_result[vpp_name]:
            raise Exception(f"duplicated step({step})")  # pylint: disable=broad-exception-raised
        parse_step_result[vpp_name][step] = ops_result
    return parse_step_result


def write_step(output_dirpath, parse_step_result, rank, data_type):
    tb_output_path = os.path.join(output_dirpath, f"rank{rank}", data_type)
    if os.path.exists(tb_output_path):
        remove_path(tb_output_path)
        logger.warning(f"existing path {tb_output_path} will be recovered")
    writer = SummaryWriter(tb_output_path)
    for vpp_name, step_data_dict in parse_step_result.items():
        step_data_list = list(step_data_dict.items())
        step_data_list.sort(key=lambda x: x[0])
        for step_data in step_data_list:
            step = step_data[0]
            ops = step_data[1]
            for op, value in ops.items():
                tag = f"{vpp_name}/{op}"
                writer.add_scalar(tag, value, step)
    writer.close()


@recursion_depth_decorator("update_dict", max_depth=50)
def update_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                try:
                    update_dict(dict1[key], value)
                except Exception as e:
                    raise Exception(f"Error updating nested dict failed at key '{key}': {e}") from e  # pylint: disable=broad-exception-raised
            else:
                raise Exception(f"duplicate key: {key}")  # pylint: disable=broad-exception-raised
        else:
            dict1[key] = value
    return dict1


def csv2tb_by_step_work(target_output_dirs, output_dirpath, data_type_list):
    for directory in tqdm(target_output_dirs):
        dirpath = directory["path"]
        rank = directory["rank"]
        for data_type in data_type_list:
            all_step_result = {}
            for filename in os.listdir(dirpath):
                if not re.match(f"{data_type}{CSV_FILE_SUFFIX}", filename):
                    continue
                filepath = os.path.join(dirpath, filename)
                try:
                    parse_step_result = parse_step_fn(filepath)
                except Exception as e:
                    logger.error(f"csv2tensorboard parse {filepath} failed \n {e}")
                    break

                all_step_result = update_dict(all_step_result, parse_step_result)
            if all_step_result:
                write_step(output_dirpath, all_step_result, rank, data_type)


def check_data_type_list(data_type_list):
    if data_type_list is None:
        logger.info(f"data_type_list is None, use default all_data_type_list: {all_data_type_list}")
        return
    if not isinstance(data_type_list, list):
        raise ValueError(f"data_type_list({data_type_list}) is not a list")
    for data_type in data_type_list:
        if data_type not in all_data_type_list:
            raise ValueError(f"data type({data_type}) is not supported, supported data type: {all_data_type_list}")


def csv2tensorboard_by_step(
    monitor_path, time_start=None, time_end=None, process_num=1, data_type_list=None, output_dirpath=None
):
    check_process_num(process_num)
    check_data_type_list(data_type_list)
    target_output_dirs = get_target_output_dir(monitor_path, time_start, time_end)
    target_output_dirs = [{"rank": rank, "path": path} for rank, path in target_output_dirs.items()]
    if output_dirpath is None:
        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区
        cur_time = datetime.datetime.now(local_tz).strftime("%b%d_%H-%M-%S")
        output_dirpath = os.path.join(monitor_path, f"{cur_time}-csv2tensorboard_by_step")
    create_directory(output_dirpath)

    task_num = len(target_output_dirs)
    task_num_per_pro = task_num // process_num
    target_data_type = data_type_list if data_type_list else all_data_type_list

    processes = []
    for pro_id in range(process_num):
        task_start_id = pro_id * task_num_per_pro
        task_end_id = (pro_id + 1) * task_num_per_pro if pro_id != process_num - 1 else task_num
        task_dirs = target_output_dirs[task_start_id:task_end_id]

        p = Process(target=csv2tb_by_step_work, args=(task_dirs, output_dirpath, target_data_type))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    logger.info(f"output has been saved to: {output_dirpath}")
