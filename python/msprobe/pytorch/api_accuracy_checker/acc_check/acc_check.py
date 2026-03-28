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

import argparse
import os
import re
import sys
import time
import gc
from collections import namedtuple

try:
    import torch_npu
except ImportError:
    is_gpu = True
    current_device = "cuda"
else:
    is_gpu = False
    current_device = "npu"

import torch
from tqdm import tqdm

from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import BackwardMessage, UtDataInfo, \
    get_validated_result_csv_path, get_validated_details_csv_path, exec_api, record_skip_info, is_unsupported_api
from msprobe.pytorch.api_accuracy_checker.acc_check.data_generate import gen_api_params, gen_args
from msprobe.pytorch.api_accuracy_checker.common.utils import api_info_preprocess, \
    initialize_save_path, UtDataProcessor, extract_basic_api_segments, ApiData
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.common.config import CheckerConfig
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward
from msprobe.core.common.file_utils import FileChecker, change_mode, \
    create_directory, get_json_contents, read_csv, check_file_or_directory_path
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.dump.pt_config import parse_json_config
from msprobe.core.common.const import Const, FileCheckConst, CompareConst
from msprobe.core.common.utils import safe_get_value, CompareException, is_int, check_op_str_pattern_valid
from msprobe.pytorch.common.utils import seed_all
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import generate_cpu_params,\
    generate_device_params, \
    ExecParams


current_time = time.strftime("%Y%m%d%H%M%S")
UT_ERROR_DATA_DIR = 'ut_error_data' + current_time
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"


not_backward_list = ['repeat_interleave']
unsupported_backward_list = ['masked_select']
unsupported_api_list = ["to", "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", 
                        "empty_with_format"]


tqdm_params = {
    'smoothing': 0,  # 平滑进度条的预计剩余时间，取值范围0到1
    'desc': 'Processing',  # 进度条前的描述文字
    'leave': True,  # 迭代完成后保留进度条的显示
    'ncols': 75,  # 进度条的固定宽度
    'mininterval': 0.1,  # 更新进度条的最小间隔秒数
    'maxinterval': 1.0,  # 更新进度条的最大间隔秒数
    'miniters': 1,  # 更新进度条之间的最小迭代次数
    'ascii': None,  # 根据环境自动使用ASCII或Unicode字符
    'unit': 'it',  # 迭代单位
    'unit_scale': True,  # 自动根据单位缩放
    'dynamic_ncols': True,  # 动态调整进度条宽度以适应控制台
    'bar_format': '{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'  # 自定义进度条输出格式
}


seed_all()


def acc_check(config):
    logger.info("start acc_check test")

    logger.info(f"acc_check task result will be saved in {config.result_csv_path}")
    logger.info(f"acc_check task details will be saved in {config.details_csv_path}")

    if config.save_error_data:
        logger.info(f"acc_check task error_data will be saved in {config.error_data_path}")
    compare = Comparator(config.result_csv_path, config.details_csv_path, config.is_continue_acc_check, config=config)


    csv_df = read_csv(config.result_csv_path)
    try:
        api_name_set = {row[0] for row in csv_df.itertuples(index=False, name=None)}
    except IndexError:
        logger.error(f"Read {config.result_csv_path} error, api_name_set is empty.")
        api_name_set = set()
    run_api_offline(config, compare, api_name_set)
    for result_csv_path, details_csv_path in zip(compare.save_path_list, compare.detail_save_path_list):
        change_mode(result_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
        change_mode(details_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
        logger.info(f"acc_check task result csv is saved in {result_csv_path}")
        logger.info(f"acc_check task details csv is saved in {details_csv_path}")
    compare.print_pretest_result()


def run_api_offline(config, compare, api_name_set):
    err_column = CompareColumn()
    for _, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        check_op_str_pattern_valid(api_full_name)
        if api_full_name in api_name_set:
            continue
        if is_unsupported_api(api_full_name):
            skip_message = f"API {api_full_name} not support for acc_check. SKIP."
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, skip_message)
            record_skip_info(api_full_name, compare, compare_alg_results)
            continue
        _, api_name = extract_basic_api_segments(api_full_name)
        if not api_name:
            err_message = f"API {api_full_name} not support for acc_check. SKIP."
            logger.error(err_message)
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, err_message)
            record_skip_info(api_full_name, compare, compare_alg_results)
            continue
        try:
            if blacklist_and_whitelist_filter(api_name, config.black_list, config.white_list):
                skip_message = f"API {api_name} in black list or not in white list. SKIP."
                logger.info(skip_message)
                compare_alg_results = err_column.to_column_value(CompareConst.SKIP, skip_message)
                record_skip_info(api_full_name, compare, compare_alg_results)
                continue
            data_info = run_torch_api(api_full_name, config.real_data_path, config.backward_content, api_info_dict)
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name, data_info)
            if config.save_error_data:
                do_save_error_data(api_full_name, data_info, config.error_data_path, is_fwd_success, is_bwd_success)
        except Exception as err:
            if "expected scalar type Long" in str(err):
                logger.warning(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               "'int32_to_int64' list in accuracy_tools/msprobe/core/common/const.py file.")
            else:
                logger.error(f"Run {api_full_name} acc_check Error: %s" % str(err))
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, str(err))
            record_skip_info(api_full_name, compare, compare_alg_results)
        finally:
            if is_gpu:
                torch.cuda.empty_cache()
            else:
                torch.npu.empty_cache()
            gc.collect()


def blacklist_and_whitelist_filter(api_name, black_list, white_list):
    """
    run api(api_name) if api_name not in black_list and in white_list.
    If api is both in black_list and black_list, black_list first.
    return: False for exec api, True for not exec
    """
    black_list.extend(unsupported_api_list)
    if black_list and api_name in black_list:
        return True
    if white_list and api_name not in white_list:
        return True
    return False


def do_save_error_data(api_full_name, data_info, error_data_path, is_fwd_success, is_bwd_success):
    if not is_fwd_success or not is_bwd_success:
        processor = UtDataProcessor(error_data_path)
        for element in data_info.in_fwd_data_list:
            processor.save_tensors_in_element(api_full_name + '.forward.input', element)
        processor.save_tensors_in_element(api_full_name + '.forward.output.bench', data_info.bench_output)
        processor.save_tensors_in_element(api_full_name + '.forward.output.device', data_info.device_output)
        processor.save_tensors_in_element(api_full_name + '.backward.input', data_info.grad_in)
        processor.save_tensors_in_element(api_full_name + '.backward.output.bench', data_info.bench_grad)
        processor.save_tensors_in_element(api_full_name + '.backward.output.device', data_info.device_grad)


def run_torch_api(api_full_name, real_data_path, backward_content, api_info_dict):
    in_fwd_data_list = []
    backward_message = ''
    api_type, api_name = extract_basic_api_segments(api_full_name)
    args, kwargs, output_dtype = get_api_info(api_info_dict, api_name, real_data_path)
    need_grad = check_need_grad(api_info_dict)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = api_full_name in backward_content
    if not need_grad:
        logger.warning("%s %s" % (api_full_name, BackwardMessage.UNSUPPORT_BACKWARD_MESSAGE))
        backward_message += BackwardMessage.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        need_grad = False
        logger.info("%s %s" % (api_full_name, BackwardMessage.NO_BACKWARD_RESULT_MESSAGE))
        backward_message += BackwardMessage.NO_BACKWARD_RESULT_MESSAGE
    if api_name in unsupported_backward_list:
        need_grad = False
        logger.info("%s %s" % (api_full_name, BackwardMessage.UNSUPPORT_API_MESSAGE))
        backward_message += BackwardMessage.UNSUPPORT_API_MESSAGE
    need_backward = need_backward and need_grad
    
    device_info_kwargs = kwargs.get(Const.DEVICE)
    if device_info_kwargs and device_info_kwargs.get(Const.VALUE):
        kwargs[Const.DEVICE] = current_device
    device_args, device_kwargs, is_fp8 = generate_device_params(args, kwargs, need_backward, api_name)
    if kwargs.get(Const.DEVICE):
        del kwargs[Const.DEVICE]
    cpu_params = generate_cpu_params(args, kwargs, need_backward, api_name)
    cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
    autocast_dtype, is_autocast = cpu_params.autocast_dtype, cpu_params.is_autocast
    if not is_autocast and output_dtype:
        is_autocast = autocast_dtype != output_dtype
        autocast_dtype = output_dtype
    bench_grad_out, device_grad_out = None, None
    cpu_exec_params = ExecParams(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs, False, autocast_dtype)
    out = exec_api(cpu_exec_params)
    device_exec_params = ExecParams(api_type, api_name, current_device, device_args, device_kwargs, is_autocast,
                                     autocast_dtype)
    device_out = exec_api(device_exec_params)
    if is_fp8 and isinstance(device_out, torch.Tensor) and device_out.dtype == torch.float32:
        device_out = device_out.to(torch.float16)
    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "torch_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    grad, bench_grad = None, None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward and out is not None:
        if need_to_backward(grad_index, out):
            backward_args = backward_content[api_full_name].get("input")
            func_options = {
                'real_data_path': real_data_path
            }
            grad = gen_args(backward_args, api_name, func_options)
            grad = safe_get_value(grad, 0, "grad")
            grad_params = generate_cpu_params(grad, {}, False, api_name)
            bench_grad = grad_params.cpu_args
            bench_grad_out = run_backward(cpu_args, bench_grad, grad_index, out)
            device_grad = grad.clone().detach().to(current_device)
            device_grad_out = run_backward(device_args, device_grad, grad_index, device_out)
        else:
            backward_message += BackwardMessage.MULTIPLE_BACKWARD_MESSAGE
    if api_name == "npu_fusion_attention":
        out = safe_get_value(out, 0, "out")
        device_out = safe_get_value(device_out, 0, "device_out")

    return UtDataInfo(bench_grad_out, device_grad_out, device_out, out, bench_grad, in_fwd_data_list, backward_message,
                      rank=0, is_fp8=is_fp8)


def check_need_grad(api_info_dict):
    need_grad = True
    if api_info_dict.get(Const.INPUT_KWARGS) and "out" in api_info_dict.get(Const.INPUT_KWARGS):
        need_grad = False
    return need_grad


def get_api_info(api_info_dict, api_name, real_data_path):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = check_need_grad(api_info_dict)
    args, kwargs, output_dtype = gen_api_params(api_info_dict, api_name, need_grad, convert_type, real_data_path)
    return args, kwargs, output_dtype


def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True


def run_backward(args, grad, grad_index, out):
    if grad_index is not None:
        if not is_int(grad_index):
            logger.error(f"{grad_index} dtype is not int")
            raise TypeError(f"{grad_index} dtype is not int")
        if grad_index >= len(out):
            logger.error(f"Run backward error when grad_index is {grad_index}")
            raise IndexError(f"Run backward error when grad_index is {grad_index}")
        out[grad_index].backward(grad)
    else:
        out.backward(grad)

    grad_out = extract_tensors_grad(args)

    return grad_out


def extract_tensors_grad(args, depth=0):
    if depth > Const.MAX_DEPTH:
        logger.error("The depth of arg_in is too large, please check the arg_in.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
    grads = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            grads.append(arg.grad)
        elif isinstance(arg, (list, tuple)):
            grads.extend(extract_tensors_grad(arg, depth + 1))
    return grads


def initialize_save_error_data(error_data_path):
    create_directory(error_data_path)
    error_data_path_checker = FileChecker(error_data_path, FileCheckConst.DIR,
                                          ability=FileCheckConst.WRITE_ABLE)
    error_data_path = error_data_path_checker.common_check()
    error_data_path = initialize_save_path(error_data_path, UT_ERROR_DATA_DIR)
    return error_data_path


def _acc_check_parser(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", default="", type=str,
                        help="<required> The api param tool result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The acc_check task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)

    class UniqueDeviceAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            unique_values = set(values)
            if len(values) != len(unique_values):
                parser.error("device id must be unique")
            for device_id in values:
                if not 0 <= device_id:
                    parser.error("device id must be greater than or equal to 0")
            setattr(namespace, self.dest, values)

    parser.add_argument("-d", "--device", dest="device_id", nargs='+', type=int,
                        help="<optional> set device id to acc_check, must be unique and in range 0-7",
                        default=[0], required=False, action=UniqueDeviceAction)
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str,
                        help="<optional> The path of accuracy_checking_result_{timestamp}.csv, "
                             "when acc_check is interrupted, enter the file path to continue acc_check.",
                        required=False)
    parser.add_argument("-f", "--filter_api", dest="filter_api", action="store_true",
                        help="<optional> Whether to filter the api in the api_info_file.", required=False)
    parser.add_argument("-config", "--config_path", dest="config_path", default="", type=str,
                        help="<optional> The path of config.json", required=False)


def preprocess_forward_content(forward_content):
    processed_content = {}
    base_keys_variants = {}
    arg_cache = {}

    for key, value in forward_content.items():
        check_op_str_pattern_valid(key)
        base_key = key.rsplit(Const.SEP, 1)[0]

        if key not in arg_cache:
            filtered_new_args = [
                {k: v for k, v in arg.items() if k not in ['Max', 'Min']}
                for arg in value['input_args']
                if isinstance(arg, dict)
            ]
            arg_cache[key] = (filtered_new_args, value['input_kwargs'])

        filtered_new_args, new_kwargs = arg_cache[key]

        if base_key not in base_keys_variants:
            processed_content[key] = value
            base_keys_variants[base_key] = {key}
        else:
            is_duplicate = False
            for variant in base_keys_variants.get(base_key, []):
                try:
                    existing_args, existing_kwargs = arg_cache.get(variant)
                except KeyError as e:
                    logger.error(f"KeyError: {e} when processing {key}")
                if existing_args == filtered_new_args and existing_kwargs == new_kwargs:
                    is_duplicate = True
                    break

            if not is_duplicate:
                processed_content[key] = value
                base_keys_variants[base_key].add(key)

    return processed_content


def _acc_check(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _acc_check_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    acc_check_command(args)


def acc_check_command(args):
    if args.config_path:
        config_path_checker = FileChecker(args.config_path, FileCheckConst.FILE, 
                                          FileCheckConst.READ_ABLE, FileCheckConst.JSON_SUFFIX)
        checked_config_path = config_path_checker.common_check()
        _, task_config = parse_json_config(checked_config_path, Const.ACC_CHECK)
        checker_config = CheckerConfig(task_config)
    else:
        checker_config = CheckerConfig()
    
    if not args.api_info_file:
        logger.error("Please provide api_info_file for offline acc_check.")
        raise Exception("Please provide api_info_file for offline acc_check.")

    if not is_gpu:
        torch.npu.set_compile_mode(jit_compile=args.jit_compile)
        if args.jit_compile:
            torch.npu.config.allow_internal_format = True
        else:
            torch.npu.config.allow_internal_format = False
    used_device = current_device + ":" + str(args.device_id[0])
    try:
        if is_gpu:
            torch.cuda.set_device(used_device)
        else:
            torch.npu.set_device(used_device)
    except Exception as error:
        logger.error(f"Set device id failed. device id is: {args.device_id}")
        raise NotImplementedError from error

    # 在线预检场景下，不需要外出输出api信息，forward_content, backward_content, real_data_path设置为None
    # 离线场景下，forward_content, backward_content, real_data_path从api_info_file中解析
    forward_content, backward_content, real_data_path = None, None, None
    if args.api_info_file:
        api_info_file_checker = FileChecker(file_path=args.api_info_file, path_type=FileCheckConst.FILE, 
                                            ability=FileCheckConst.READ_ABLE, file_type=FileCheckConst.JSON_SUFFIX)
        checked_api_info = api_info_file_checker.common_check()
        forward_content, backward_content, real_data_path = parse_json_info_forward_backward(checked_api_info)
        if real_data_path:
            dump_path = os.path.dirname(checked_api_info)
            real_data_path = os.path.join(dump_path, Const.DUMP_TENSOR_DATA)
        if args.filter_api:
            logger.info("Start filtering the api in the api_info_file.")
            forward_content = preprocess_forward_content(forward_content)
            logger.info("Finish filtering the api in the api_info_file.")

    out_path = args.out_path if args.out_path else Const.DEFAULT_PATH
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data

    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)

    error_data_path = checker_config.error_data_path
    if save_error_data:
        if args.result_csv_path:
            parts_by_dot = result_csv_path.split(Const.SEP)
            if len(parts_by_dot) < 2 or not parts_by_dot[0]:
                raise ValueError("result_csv_path does not contain a valid file name with an extension.")
            file_name_part = parts_by_dot[0]
            parts_by_underscore = file_name_part.split(Const.REPLACEMENT_CHARACTER)
            if len(parts_by_underscore) < 2:
                raise ValueError("File name part does not contain enough '_' separated segments.")
            time_info = parts_by_underscore[-1]

            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = 'ut_error_data' + time_info
        error_data_path = initialize_save_error_data(error_data_path)
    config_params = {
        'forward_content': forward_content,
        'backward_content': backward_content,
        'result_csv_path': result_csv_path,
        'details_csv_path': details_csv_path,
        'save_error_data': save_error_data,
        'is_continue_acc_check': args.result_csv_path,
        'real_data_path': real_data_path,
        'error_data_path': error_data_path
    }
    acc_check_config = checker_config.get_acc_check_config(**config_params)
    acc_check(acc_check_config)
    logger.info("acc_check task completed.")


if __name__ == '__main__':
    _acc_check()
