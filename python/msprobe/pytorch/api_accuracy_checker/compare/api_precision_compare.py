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
import sys
from collections import namedtuple


from msprobe.core.common.file_utils import write_csv, read_csv
from msprobe.pytorch.api_accuracy_checker.common.config import msCheckerConfig
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import (
    API_PRECISION_COMPARE_RESULT_FILE_NAME,
    API_PRECISION_COMPARE_DETAILS_FILE_NAME,
    API_PRECISION_COMPARE_UNSUPPORT_LIST,
    ApiPrecisionCompareColumn,
    convert_str_to_float,
    CompareMessage,
)
from msprobe.pytorch.api_accuracy_checker.compare.compare_input import PrecisionCompareInput
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_register import StandardRegistry
from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpPrecisionCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.benchmark_compare import BenchmarkPrecisionCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import ApiPrecisionOutputColumn
from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import get_validated_result_csv_path
from msprobe.pytorch.api_accuracy_checker.common.utils import extract_detailed_api_segments, extract_basic_api_segments
from msprobe.core.common.file_utils import FileChecker, create_directory
from msprobe.pytorch.common.log import logger
from msprobe.core.common.utils import CompareException, check_op_str_pattern_valid
from msprobe.core.common.const import Const, CompareConst, FileCheckConst

CompareConfig = namedtuple('CompareConfig', ['npu_csv_path', 'gpu_csv_path', 'result_csv_path', 'details_csv_path'])
BenchmarkInfNanConsistency = namedtuple(
    'BenchmarkInfNanConsistency',
    [
        'small_value_inf_nan_consistency',
        'rmse_inf_nan_consistency',
        'max_rel_inf_nan_consistency',
        'mean_rel_inf_nan_consistency',
        'eb_inf_nan_consistency',
    ],
)
UNSUPPORTED_MESSAGE = 'This data type does not support benchmark compare.'


benchmark_message = {
    "small_value_err_status": {
        CompareConst.ERROR: "ERROR: 小值域错误比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 小值域错误比值超过阈值\n",
    },
    "rmse_status": {
        CompareConst.ERROR: "ERROR: 均方根误差比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 均方根误差比值超过阈值\n",
    },
    "max_rel_err_status": {
        CompareConst.ERROR: "ERROR: 相对误差最大值比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 相对误差最大值比值超过阈值\n",
    },
    "mean_rel_err_status": {
        CompareConst.ERROR: "ERROR: 相对误差平均值比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 相对误差平均值比值超过阈值\n",
    },
}


def write_detail_csv(content, save_path):
    rows = []
    content = [
        "{:.{}f}".format(item, msCheckerConfig.precision) if isinstance(item, float) else item for item in content
    ]
    rows.append(content)
    write_csv(rows, save_path)


def register_compare_func():
    registry = StandardRegistry()
    registry.register(CompareConst.ABSOLUTE_THRESHOLD, record_absolute_threshold_result)
    registry.register(CompareConst.BINARY_CONSISTENCY, record_binary_consistency_result)
    registry.register(CompareConst.ULP_COMPARE, record_ulp_compare_result)
    registry.register(CompareConst.THOUSANDTH_STANDARD, record_thousandth_threshold_result)
    registry.register(CompareConst.BENCHMARK, record_benchmark_compare_result)
    registry.register(CompareConst.ACCUMULATIVE_ERROR_COMPARE, record_accumulative_error_compare_result)
    return registry


def api_precision_compare(config):
    logger.info("Start compare task")
    logger.info(f"Compare task result will be saved in {config.result_csv_path}")
    logger.info(f"Compare task detail will be saved in {config.details_csv_path}")
    try:
        npu_data = read_csv(config.npu_csv_path)
    except Exception as err:
        logger.error("Open npu csv Error: %s" % str(err))
    check_csv_columns(npu_data.columns, "npu_csv")
    try:
        gpu_data = read_csv(config.gpu_csv_path)
    except Exception as err:
        logger.error("Open gpu csv Error: %s" % str(err))
    check_csv_columns(gpu_data.columns, "gpu_csv")
    detail_csv_title = [ApiPrecisionCompareColumn.get_detail_csv_title()]
    result_csv_title = [ApiPrecisionCompareColumn.get_result_csv_title()]
    write_csv(result_csv_title, config.result_csv_path)
    write_csv(detail_csv_title, config.details_csv_path)
    try:
        analyse_csv(npu_data, gpu_data, config)
    except Exception as err:
        logger.error("Analyse csv Error: %s" % str(err))


def analyse_csv(npu_data, gpu_data, config):
    forward_status, backward_status = [], []
    last_api_name, last_api_dtype = None, None
    last_api_skip_message = ''
    registry = register_compare_func()

    for _, row_npu in npu_data.iterrows():
        message = ''
        compare_column = ApiPrecisionOutputColumn()
        full_api_name_with_direction_status = row_npu[ApiPrecisionCompareColumn.API_NAME]
        check_op_str_pattern_valid(full_api_name_with_direction_status)
        row_gpu = gpu_data[gpu_data[ApiPrecisionCompareColumn.API_NAME] == full_api_name_with_direction_status]
        api_name, api_full_name, direction_status = extract_detailed_api_segments(full_api_name_with_direction_status)
        if not api_full_name:
            err_message = f"The API name {full_api_name_with_direction_status} is invalid."
            logger.error(err_message)
            compare_column.api_name = full_api_name_with_direction_status
            compare_column.compare_result = CompareConst.SKIP
            compare_column.compare_message = err_message
            write_detail_csv(compare_column.to_column_value(), config.details_csv_path)
            write_csv(
                [[full_api_name_with_direction_status, CompareConst.SKIP, CompareConst.SKIP, err_message]],
                config.result_csv_path,
            )
            continue
        if row_gpu.empty:
            logger.warning(f'This API : {full_api_name_with_direction_status} does not exist in the GPU data.')
            continue
        if len(row_gpu) > 1:
            msg = f'This API : {full_api_name_with_direction_status} has multiple records in the GPU data.'
            raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
        row_gpu = row_gpu.iloc[0]
        new_status = CompareConst.SPACE
        try:
            new_status = get_api_status(row_npu, row_gpu, api_name, compare_column, registry)
        except Exception as err:
            logger.error(f"Get api status error: {str(err)}")
            compare_column.api_name = full_api_name_with_direction_status
            compare_column.compare_result = CompareConst.SKIP
            compare_column.compare_message = str(err)
            write_detail_csv(compare_column.to_column_value(), config.details_csv_path)
            write_csv(
                [[full_api_name_with_direction_status, CompareConst.SKIP, CompareConst.SKIP, str(err)]],
                config.result_csv_path,
            )
            continue

        write_detail_csv(compare_column.to_column_value(), config.details_csv_path)

        if last_api_name is not None and api_full_name != last_api_name:
            if last_api_dtype in API_PRECISION_COMPARE_UNSUPPORT_LIST:
                message = UNSUPPORTED_MESSAGE
                write_csv([[last_api_name, CompareConst.SKIP, CompareConst.SKIP, message]], config.result_csv_path)
                print_test_success(last_api_name, CompareConst.SKIP, CompareConst.SKIP)
            else:
                forward_result = get_api_checker_result(forward_status)
                backward_result = get_api_checker_result(backward_status)
                _, base_api_name = extract_basic_api_segments(last_api_name)
                message += CompareMessage.get(base_api_name, "") if forward_result == CompareConst.ERROR else ""
                message += last_api_skip_message if forward_result == CompareConst.SKIP else ""
                write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)
                print_test_success(last_api_name, forward_result, backward_result)
                last_api_skip_message = ''
            forward_status, backward_status = [], []
            message = ''

        is_supported = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] not in API_PRECISION_COMPARE_UNSUPPORT_LIST
        last_api_name = api_full_name

        last_api_dtype = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]
        if not is_supported:
            continue

        if direction_status == 'forward':
            forward_status.append(new_status)
            last_api_skip_message = (
                str(row_npu[ApiPrecisionCompareColumn.MESSAGE]) if new_status == CompareConst.SKIP else ''
            )
        elif direction_status == 'backward':
            backward_status.append(new_status)
        else:
            logger.error(f"Invalid direction status: {direction_status}")

    if last_api_name is not None:
        if last_api_dtype in API_PRECISION_COMPARE_UNSUPPORT_LIST:
            message = UNSUPPORTED_MESSAGE
            write_csv([[last_api_name, CompareConst.SKIP, CompareConst.SKIP, message]], config.result_csv_path)
            print_test_success(last_api_name, CompareConst.SKIP, CompareConst.SKIP)
        else:
            forward_result = get_api_checker_result(forward_status)
            backward_result = get_api_checker_result(backward_status)
            _, base_api_name = extract_basic_api_segments(last_api_name)
            message += CompareMessage.get(base_api_name, "") if forward_result == CompareConst.ERROR else ""
            message += last_api_skip_message if forward_result == CompareConst.SKIP else ""
            write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)
            print_test_success(last_api_name, forward_result, backward_result)
            last_api_skip_message = ''


def get_api_status(row_npu, row_gpu, api_name, compare_column, registry):
    full_api_name_with_direction_status = row_npu[ApiPrecisionCompareColumn.API_NAME]
    # 当前API的输出为空（例如反向过程中requires_grad=False）,跳过比对
    if (
        row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE].isspace()
        or row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] in API_PRECISION_COMPARE_UNSUPPORT_LIST
        or row_npu[ApiPrecisionCompareColumn.SHAPE] == CompareConst.ZERO_SHAPE
    ):
        compare_column.api_name = full_api_name_with_direction_status
        compare_column.compare_result = CompareConst.SKIP
        compare_column.compare_message = row_npu[ApiPrecisionCompareColumn.MESSAGE]
        new_status = CompareConst.SKIP
    else:
        compare_column.api_name = full_api_name_with_direction_status
        dtype = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]
        input_data = PrecisionCompareInput(row_npu, row_gpu, dtype, compare_column, api_name=api_name)
        in_dtype = get_in_dtype(row_npu)
        comparison_func = registry.get_comparison_function(api_name, dtype, in_dtype)
        new_status = comparison_func(input_data)
    return new_status


def get_in_dtype(row_npu):
    if row_npu[ApiPrecisionCompareColumn.REL_ERR_RATIO].isspace():
        return "torch.float32"
    else:
        return "torch.float8_e4m3fn"


def print_test_success(api_full_name, forward_result, backward_result):
    is_fwd_success = forward_result == CompareConst.PASS
    is_bwd_success = backward_result in (CompareConst.PASS, CompareConst.SPACE)
    logger.info(
        f"running api_full_name {api_full_name} compare, "
        f"is_fwd_success: {is_fwd_success}, "
        f"is_bwd_success: {is_bwd_success}"
    )


def check_error_rate(npu_error_rate):
    return CompareConst.PASS if convert_str_to_float(npu_error_rate) == 0 else CompareConst.ERROR


def get_absolute_threshold_result(row_npu):
    inf_nan_error_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO])
    rel_err_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.REL_ERR_RATIO])
    abs_err_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.ABS_ERR_RATIO])

    inf_nan_result = CompareConst.PASS if inf_nan_error_ratio == 0 else CompareConst.ERROR
    rel_err_result = CompareConst.PASS if rel_err_ratio == 0 else CompareConst.ERROR
    abs_err_result = CompareConst.PASS if abs_err_ratio == 0 else CompareConst.ERROR

    if CompareConst.ERROR in [inf_nan_result, rel_err_result, abs_err_result]:
        absolute_threshold_result = CompareConst.ERROR
    else:
        absolute_threshold_result = CompareConst.PASS

    return {
        "inf_nan_error_ratio": inf_nan_error_ratio,
        "inf_nan_result": inf_nan_result,
        "rel_err_ratio": rel_err_ratio,
        "rel_err_result": rel_err_result,
        "abs_err_ratio": abs_err_ratio,
        "abs_err_result": abs_err_result,
        "absolute_threshold_result": absolute_threshold_result,
    }


def get_api_checker_result(status):
    if not status:
        return CompareConst.SPACE
    if all(item == CompareConst.SKIP for item in status):
        return CompareConst.SKIP
    for const in (CompareConst.ERROR, CompareConst.WARNING):
        if const in status:
            return const
    return CompareConst.PASS


def check_csv_columns(columns, csv_type):
    required_columns = ApiPrecisionCompareColumn.to_required_columns()
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        msg = f"The following columns {','.join(missing_columns)} are missing in{csv_type}"
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg)


def record_binary_consistency_result(input_data):
    row_npu = input_data.row_npu
    compare_column = input_data.compare_column
    new_status = check_error_rate(row_npu[ApiPrecisionCompareColumn.ERROR_RATE])
    compare_column.error_rate = row_npu[ApiPrecisionCompareColumn.ERROR_RATE]
    compare_column.error_rate_status = new_status
    compare_column.compare_result = new_status
    compare_column.compare_algorithm = CompareConst.BINARY_CONSISTENCY_ALGORITHM_NAME
    message = ''
    if compare_column.error_rate_status == CompareConst.ERROR:
        message += "ERROR: 二进制一致错误率超过阈值\n"
    compare_column.compare_message = message
    return new_status


def record_absolute_threshold_result(input_data):
    row_npu = input_data.row_npu
    compare_column = input_data.compare_column
    absolute_threshold_result = get_absolute_threshold_result(row_npu)
    compare_column.inf_nan_error_ratio = absolute_threshold_result.get("inf_nan_error_ratio")
    compare_column.inf_nan_error_ratio_status = absolute_threshold_result.get("inf_nan_result")
    compare_column.rel_err_ratio = absolute_threshold_result.get("rel_err_ratio")
    compare_column.rel_err_ratio_status = absolute_threshold_result.get("rel_err_result")
    compare_column.abs_err_ratio = absolute_threshold_result.get("abs_err_ratio")
    compare_column.abs_err_ratio_status = absolute_threshold_result.get("abs_err_result")
    compare_column.compare_result = absolute_threshold_result.get("absolute_threshold_result")
    compare_column.compare_algorithm = CompareConst.ABSOLUTE_THRESHOLD_ALGORITHM_NAME
    message = ''
    if compare_column.inf_nan_error_ratio_status == CompareConst.ERROR:
        message += "ERROR: inf/nan错误率超过阈值"
    if compare_column.rel_err_ratio_status == CompareConst.ERROR:
        message += "ERROR: 相对误差错误率超过阈值"
    if compare_column.abs_err_ratio_status == CompareConst.ERROR:
        message += "ERROR: 绝对误差错误率超过阈值"
    compare_column.compare_message = message
    return compare_column.compare_result


def record_benchmark_compare_result(input_data):
    bs = BenchmarkPrecisionCompare(input_data)
    compare_result = bs.compare()
    for status_attr, messages in benchmark_message.items():
        status_value = getattr(input_data.compare_column, status_attr)
        if status_value in messages:
            input_data.compare_column.compare_message += messages[status_value]
    return compare_result


def record_ulp_compare_result(input_data):
    us = UlpPrecisionCompare(input_data)
    compare_result = us.compare()
    return compare_result


def record_accumulative_error_compare_result(input_data):
    row_npu = input_data.row_npu
    compare_column = input_data.compare_column
    absolute_threshold_result = get_absolute_threshold_result(row_npu)
    threshold_result = absolute_threshold_result.get("absolute_threshold_result")
    eb, eb_result = check_eb(row_npu)
    accumulative_error_compare_result = CompareConst.PASS
    if CompareConst.ERROR in [threshold_result, eb_result]:
        accumulative_error_compare_result = CompareConst.ERROR

    compare_column.inf_nan_error_ratio = absolute_threshold_result.get("inf_nan_error_ratio")
    compare_column.inf_nan_error_ratio_status = absolute_threshold_result.get("inf_nan_result")
    compare_column.rel_err_ratio = absolute_threshold_result.get("rel_err_ratio")
    compare_column.rel_err_ratio_status = absolute_threshold_result.get("rel_err_result")
    compare_column.abs_err_ratio = absolute_threshold_result.get("abs_err_ratio")
    compare_column.abs_err_ratio_status = absolute_threshold_result.get("abs_err_result")
    compare_column.eb_ratio = eb
    compare_column.eb_status = eb_result
    compare_column.compare_result = accumulative_error_compare_result
    compare_column.compare_algorithm = CompareConst.ACCUMULATIVE_ERROR_COMPARE_ALGORITHM_NAME
    message = []
    if compare_column.inf_nan_error_ratio_status == CompareConst.ERROR:
        message.append("ERROR: inf/nan错误率超过阈值\n")
    if compare_column.rel_err_ratio_status == CompareConst.ERROR:
        message.append("ERROR: 相对误差错误率超过阈值\n")
    if compare_column.abs_err_ratio_status == CompareConst.ERROR:
        message.append("ERROR: 绝对误差错误率超过阈值\n")
    if compare_column.eb_status == CompareConst.ERROR:
        message.append("ERROR: 误差均衡性超过阈值\n")
    compare_column.compare_message = "\n".join(message)
    return compare_column.compare_result


def check_eb(row_npu):
    eb = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.EB])
    dtype = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]
    eb_threshold = StandardConfig.get_accumulative_error_eb_threshold(dtype)
    eb_result = CompareConst.PASS if eb <= eb_threshold else CompareConst.ERROR
    return eb, eb_result


def check_thousandth_rate(thousandth_rate):
    return (
        CompareConst.PASS
        if convert_str_to_float(thousandth_rate) >= CompareConst.THOUSANDTH_PASS_VALUE
        else CompareConst.ERROR
    )


def record_thousandth_threshold_result(input_data):
    row_npu = input_data.row_npu
    compare_column = input_data.compare_column
    new_status = check_thousandth_rate(row_npu[ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH])
    compare_column.rel_err_thousandth = row_npu[ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH]
    compare_column.rel_err_thousandth_status = new_status
    compare_column.compare_result = new_status
    compare_column.compare_algorithm = CompareConst.THOUSANDTH_STANDARD_ALGORITHM_NAME
    message = ''
    if compare_column.rel_err_thousandth_status == CompareConst.ERROR:
        message += "ERROR: 双千指标不达标\n"
    compare_column.compare_message = message
    return compare_column.compare_result


def _api_precision_compare(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _api_precision_compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    _api_precision_compare_command(args)
    logger.info("Compare task completed.")


def _api_precision_compare_command(args):
    npu_csv_path = get_validated_result_csv_path(args.npu_csv_path, 'detail')
    gpu_csv_path = get_validated_result_csv_path(args.gpu_csv_path, 'detail')
    out_path = args.out_path if args.out_path else Const.DEFAULT_PATH
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    result_csv_path = os.path.join(out_path, API_PRECISION_COMPARE_RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, API_PRECISION_COMPARE_DETAILS_FILE_NAME)
    compare_config = CompareConfig(npu_csv_path, gpu_csv_path, result_csv_path, details_csv_path)
    api_precision_compare(compare_config)


# pylint: disable=duplicate-code
def _api_precision_compare_parser(parser):
    parser.add_argument(
        "-npu",
        "--npu_csv_path",
        dest="npu_csv_path",
        default="",
        type=str,
        help="<Required> , Accuracy_checking_details.csv generated on the NPU by using the api_accuracy_checker tool.",
        required=True,
    )
    parser.add_argument(
        "-gpu",
        "--gpu_csv_path",
        dest="gpu_csv_path",
        default="",
        type=str,
        help="<Required> Accuracy_checking_details.csv generated on the GPU by using the api_accuracy_checker tool.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_path",
        dest="out_path",
        default="",
        type=str,
        help="<optional> The api precision compare task result out path.",
        required=False,
    )
