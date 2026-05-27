#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import subprocess  # nosec
import os
import sys
import argparse
import time
import signal
import threading
from collections import namedtuple
from itertools import cycle
from tqdm import tqdm
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check import _acc_check_parser, preprocess_forward_content
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import (
    get_validated_result_csv_path,
    get_validated_details_csv_path,
)
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.common import parse_json_info_forward_backward
from msprobe.pytorch.common.log import logger
from msprobe.core.common.file_utils import FileChecker, check_link, create_directory, load_json, save_json, read_csv
from msprobe.core.common.file_utils import remove_path
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException


def split_json_file(input_file, num_splits, filter_api):
    forward_data, backward_data, real_data_path = parse_json_info_forward_backward(input_file)
    input_dir = os.path.dirname(os.path.abspath(input_file))
    if filter_api:
        forward_data = preprocess_forward_content(forward_data)
    for data_name in list(forward_data.keys()):
        forward_data[f"{data_name}.forward"] = forward_data.pop(data_name)
    for data_name in list(backward_data.keys()):
        backward_data[f"{data_name}.backward"] = backward_data.pop(data_name)

    input_data = load_json(input_file)
    if "dump_data_dir" not in input_data.keys():
        logger.error("Invalid input file, 'dump_data_dir' field is missing")
        raise CompareException("Invalid input file, 'dump_data_dir' field is missing")
    if input_data.get("data") is None:
        logger.error("Invalid input file, 'data' field is missing")
        raise CompareException("Invalid input file, 'data' field is missing")
    input_data.pop("data")

    items = list(forward_data.items())
    total_items = len(items)
    chunk_size = total_items // num_splits
    split_files = []

    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else total_items

        split_forward_data = dict(items[start:end])
        temp_data = {**input_data, "data": {**split_forward_data, **backward_data}}
        split_filename = os.path.join(input_dir, f"temp_part{i}.json")
        save_json(split_filename, temp_data)
        split_files.append(split_filename)

    return split_files, total_items


def signal_handler(signum, frame):
    logger.warning(f'Signal handler called with signal {signum}')
    raise KeyboardInterrupt()


ParallelUTConfig = namedtuple(
    'ParallelUTConfig',
    [
        'api_files',
        'out_path',
        'num_splits',
        'save_error_data_flag',
        'jit_compile_flag',
        'device_id',
        'result_csv_path',
        'total_items',
        'config_path',
    ],
)


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device_id)
    if config.save_error_data_flag:
        logger.info("acc_check task error data will be saved")
    logger.info(f"Starting parallel acc_check with {config.num_splits} processes")
    progress_bar = tqdm(total=config.total_items, desc="Total items", unit="items")

    def create_cmd(api_info, dev_id):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        acc_check_path = os.path.join(dirname, "acc_check.py")
        cmd = [
            sys.executable,
            acc_check_path,
            '-api_info',
            api_info,
            *(['-o', config.out_path] if config.out_path else []),
            '-d',
            str(dev_id),
            *(['-j'] if config.jit_compile_flag else []),
            *(['-save_error_data'] if config.save_error_data_flag else []),
            '-csv_path',
            config.result_csv_path,
            *(['-config', config.config_path] if config.config_path else []),
        ]
        return cmd

    def read_process_output(process):
        try:
            while True:
                if process.poll() is not None:
                    break
                output = process.stdout.readline()
                if output == '':
                    break
                if '[ERROR]' in output:
                    logger.warning(output)
                    sys.stdout.flush()
        except ValueError as e:
            logger.warning(f"An error occurred while reading subprocess output: {e}")
        finally:
            if process.poll() is None:
                try:
                    process.stdout.close()
                except Exception:  # nosec
                    pass

    def update_progress_bar(progress_bar, result_csv_path):
        while any(process.poll() is None for process in processes):
            result_file = read_csv(result_csv_path)
            completed_items = len(result_file)
            progress_bar.update(completed_items - progress_bar.n)
            time.sleep(1)

    for api_info in config.api_files:
        cmd = create_cmd(api_info, next(device_id_cycle))
        process = subprocess.Popen(  # nosec  # pylint: disable=consider-using-with
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, shell=False
        )
        processes.append(process)
        threading.Thread(target=read_process_output, args=(process,), daemon=True).start()

    progress_bar_thread = threading.Thread(target=update_progress_bar, args=(progress_bar, config.result_csv_path))
    progress_bar_thread.start()

    def clean_up():
        progress_bar.close()
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        for file in config.api_files:
            check_link(file)
            try:
                remove_path(file)
            except FileNotFoundError:
                logger.warning(f"File not found and could not be deleted: {file}")

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating processes and cleaning up...")
    except Exception as e:
        logger.warning(f"An unexpected issue occurred: {e}")
    finally:
        # 最后再更新一次进度条，避免因缓存写入等原因子进程结束而进度未刷新的问题
        if wait_for_file_write_complete(config.result_csv_path):
            result_file = read_csv(config.result_csv_path)
            completed_items = len(result_file)
            progress_bar.update(completed_items - progress_bar.n)
        if progress_bar.n < config.total_items:
            logger.warning(
                "The acc_check task has not been completed. The parameter '-csv_path' along with the "
                "path to "
                "the result CSV file will be utilized to resume the acc_check task."
            )
        clean_up()
        progress_bar_thread.join()
    try:
        comparator = Comparator(config.result_csv_path, config.result_csv_path, False)
        comparator.print_pretest_result()
    except FileNotFoundError as e:
        logger.warning(f"Error: {e}")
    except Exception as e:
        logger.warning(f"An unexpected issue occurred: {e}")


def wait_for_file_write_complete(file_path, timeout=3600):
    last_size = 0
    start_time = time.time()  # 记录开始时间
    while True:
        current_size = os.path.getsize(file_path)
        # 检查是否文件大小未变化
        if current_size == last_size:
            return True  # 文件写入完成，返回 True
        last_size = current_size
        # 检查是否超时
        if time.time() - start_time > timeout:
            logger.error("write the result csv file timeout.")
            return False  # 超时，返回 False
        time.sleep(0.1)  # 适当的延时


def prepare_config(args):
    api_info_file_checker = FileChecker(
        file_path=args.api_info_file,
        path_type=FileCheckConst.FILE,
        ability=FileCheckConst.READ_ABLE,
        file_type=FileCheckConst.JSON_SUFFIX,
    )
    api_info = api_info_file_checker.common_check()
    out_path = args.out_path if args.out_path else Const.DEFAULT_PATH
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    split_files, total_items = split_json_file(api_info, args.num_splits, args.filter_api)
    config_path = args.config_path if args.config_path else None
    if config_path:
        config_path_checker = FileChecker(
            config_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.JSON_SUFFIX
        )
        config_path = config_path_checker.common_check()
    result_csv_path = args.result_csv_path or os.path.join(
        out_path, f"accuracy_checking_result_{time.strftime('%Y%m%d%H%M%S')}.csv"
    )
    if not args.result_csv_path:
        details_csv_path = os.path.join(out_path, f"accuracy_checking_details_{time.strftime('%Y%m%d%H%M%S')}.csv")
        Comparator(result_csv_path, details_csv_path, False)
    else:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)
    logger.info(f"acc_check task result will be saved in {result_csv_path}")
    logger.info(f"acc_check task details will be saved in {details_csv_path}")
    return ParallelUTConfig(
        split_files,
        out_path,
        args.num_splits,
        args.save_error_data,
        args.jit_compile,
        args.device_id,
        result_csv_path,
        total_items,
        config_path,
    )


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser(description='Run acc_check in parallel')
    _acc_check_parser(parser)
    parser.add_argument(
        '-n',
        '--num_splits',
        type=int,
        choices=range(1, 65),
        default=8,
        help='Number of splits for parallel processing. Range: 1-64',
    )
    args = parser.parse_args()
    config = prepare_config(args)
    run_parallel_ut(config)
