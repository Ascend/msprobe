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

# 进行比对及结果展示
import os
from collections import namedtuple

import numpy as np
from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import get_json_contents, write_csv
import torch
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_register import StandardRegistry
from msprobe.pytorch.api_accuracy_checker.precision_standard.absolute_threshold import AbsolutethdCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.benchmark_compare import BenchmarkCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.binary_consistency import BinaryCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.thousandth_standard import ThousandthStdCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.accumulative_error_compare import AccumulativeErrorCompare
from msprobe.pytorch.api_accuracy_checker.compare.compare_input import CompareInput
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_abs_err, get_max_abs_err, get_rel_err_ratio, \
    cosine_sim, get_rel_err_origin, get_abs_bench_with_eps, compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.common.config import msCheckerConfig
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import check_dtype_comparable, \
    DETAIL_TEST_ROWS, BENCHMARK_COMPARE_SUPPORT_LIST
from msprobe.pytorch.api_accuracy_checker.common.utils import extract_basic_api_segments
from msprobe.pytorch.common.log import logger
from msprobe.core.common.decorator import recursion_depth_decorator


ResultInfo = namedtuple('ResultInfo', ['full_api_name', 'fwd_success_status', 'bwd_success_status',
                                       'fwd_compare_alg_results', 'bwd_compare_alg_results', 'rank'])


INDEX_TEST_RESULT_GROUP = 3
BACKWARD_RESULT_GROUP = 4
INDEX_FIRST_GROUP = 0
INDEX_MESSAGE = -1


class Comparator:
    # consts for result csv
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_csv_path, details_csv_path, is_continue_acc_check,
                 stack_info_json_path=None, config=None):
        self.save_path_str = result_csv_path
        self.detail_save_path_str = details_csv_path
        self.save_path_list = [result_csv_path]
        self.detail_save_path_list = [details_csv_path]

        self.registry = self._register_compare_func()

        if not is_continue_acc_check:
            self.write_csv_title()
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None

    @staticmethod
    def get_path_from_rank(rank, path_list, path_pattern):
        return path_list[-1] if len(path_list) == 1 else path_pattern.format(rank)

    @staticmethod
    def print_pretest_result():
        logger.info("Successfully completed acc_check/multi_acc_check.")

    @staticmethod
    def _compare_dropout(bench_output, device_output):
        tensor_num = bench_output.numel()
        if tensor_num >= 100:
            if abs((bench_output == 0).sum() - (device_output == 0).cpu().sum()) / tensor_num < 0.1:
                return CompareConst.PASS, 1
            else:
                return CompareConst.ERROR, 0
        else:
            return CompareConst.PASS, 1

    @staticmethod
    def _compare_builtin_type(bench_output, device_output, compare_column):
        if not isinstance(bench_output, (bool, int, float, str)):
            return CompareConst.PASS, compare_column, ""
        if bench_output != device_output:
            return CompareConst.ERROR, compare_column, ""
        compare_column.error_rate = 0
        return CompareConst.PASS, compare_column, ""

    @staticmethod
    def _get_acc_check_detail(test_result):
        """get acc_check detail before write to csv, called by online acc_check"""
        test_rows = []
        try:
            subject_prefix = test_result[0]
            fwd_result = test_result[3]
            bwd_result = test_result[4]
        except IndexError as e:
            logger.error("List index out of bounds when writing detail CSV.")
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR, "list index out of bounds") from e

        if isinstance(fwd_result, list):
            for i, test_subject in enumerate(fwd_result):
                subject = subject_prefix + ".forward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision)
                                if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))
        if isinstance(bwd_result, list):
            for i, test_subject in enumerate(bwd_result):
                subject = subject_prefix + ".backward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision)
                                if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))
        return test_rows

    @staticmethod
    def _binary_standard_compare(input_data):
        binary_compare = BinaryCompare(input_data)
        binary_compare.compare()
    
    @staticmethod
    def _thousandth_standard_compare(input_data):
        thousandth_compare = ThousandthStdCompare(input_data)
        thousandth_compare.compare()
    
    @staticmethod
    def _absolute_standard_compare(input_data):
        absolute_compare = AbsolutethdCompare(input_data)
        absolute_compare.compare()

    @staticmethod
    def _ulp_compare(input_data):
        ulp_compare = UlpCompare(input_data)
        ulp_compare.compare()
    
    @staticmethod
    def _benchmark_compare(input_data):
        benchmark_compare = BenchmarkCompare(input_data)
        benchmark_compare.compare()
    
    @staticmethod
    def _accumulative_error_compare(input_data):
        accumulative_error_compare = AccumulativeErrorCompare(input_data)
        accumulative_error_compare.compare()

    def write_csv_title(self):
        summary_test_rows = [
            [self.COLUMN_API_NAME, 
             self.COLUMN_FORWARD_SUCCESS,
             self.COLUMN_BACKWARD_SUCCESS,
             "Message"]
            ]
        for save_path, detail_save_path in zip(self.save_path_list, self.detail_save_path_list):
            if not os.path.exists(save_path):
                write_csv(summary_test_rows, save_path)
            if not os.path.exists(detail_save_path):
                write_csv(DETAIL_TEST_ROWS, detail_save_path)

    @recursion_depth_decorator("compare_core")
    def _compare_core(self, api_name, bench_output, device_output, is_fp8):
        compare_column = CompareColumn()
        if not isinstance(bench_output, type(device_output)):
            status = CompareConst.ERROR
            message = "bench and npu output type is different."
        elif isinstance(bench_output, dict):
            b_keys, n_keys = set(bench_output.keys()), set(device_output.keys())
            if b_keys != n_keys:
                status = CompareConst.ERROR
                message = "bench and npu output dict keys are different."
            else:
                status, compare_column, message = self._compare_core(api_name, list(bench_output.values()),
                                                                     list(device_output.values()), is_fp8)
        elif isinstance(bench_output, torch.Tensor):
            copy_bench_out = bench_output.detach().clone()
            copy_device_output = device_output.detach().clone()
            compare_column.bench_type = str(copy_bench_out.dtype)
            compare_column.npu_type = str(copy_device_output.dtype)
            compare_column.shape = tuple(device_output.shape)
            status, compare_column, message = self._compare_torch_tensor(api_name, copy_bench_out, copy_device_output,
                                                                         compare_column, is_fp8)
        elif isinstance(bench_output, (bool, int, float, str)):
            compare_column.bench_type = str(type(bench_output))
            compare_column.npu_type = str(type(device_output))
            status, compare_column, message = self._compare_builtin_type(bench_output, device_output, compare_column)
        elif bench_output is None:
            status = CompareConst.SKIP
            message = "Bench output is None, skip this test."
        else:
            status = CompareConst.ERROR
            message = "Unexpected output type in compare_core: {}".format(type(bench_output))

        return status, compare_column, message
    
    def write_summary_csv(self, test_result):
        test_rows = []
        try:
            name = test_result[0]
            df_row = list(test_result[:INDEX_TEST_RESULT_GROUP])
            if test_result[1] == CompareConst.SKIP:
                df_row.append(test_result[INDEX_TEST_RESULT_GROUP][INDEX_FIRST_GROUP][INDEX_MESSAGE])
            elif test_result[2] == CompareConst.SKIP:
                df_row.append(test_result[BACKWARD_RESULT_GROUP][INDEX_FIRST_GROUP][INDEX_MESSAGE])
            if self.stack_info:
                stack_info = "\n".join(self.stack_info[name])
                df_row.append(stack_info)
            test_rows.append(df_row)
            save_path = self.get_path_from_rank(test_result[-1], self.save_path_list, self.save_path_str)
        except IndexError as e:
            logger.error("List index out of bounds when writing summary CSV.")
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR, "list index out of bounds") from e
        write_csv(test_rows, save_path)

    def write_detail_csv(self, test_result):
        test_rows = self._get_acc_check_detail(test_result)
        detail_save_path = self.get_path_from_rank(test_result[-1],
                                                   self.detail_save_path_list,
                                                   self.detail_save_path_str)
        write_csv(test_rows, detail_save_path)

    def record_results(self, args):
        self.write_summary_csv(args)
        self.write_detail_csv(args)


    def compare_output(self, full_api_name, data_info):
        """Get compare result and write to result and detail csv.
        """
        _, api_name = extract_basic_api_segments(full_api_name)
        if not api_name:
            raise ValueError(f"API name {full_api_name} has not been adapted.")
        bench_output, device_output = data_info.bench_output, data_info.device_output
        bench_grad, device_grad = data_info.bench_grad, data_info.device_grad
        backward_message = data_info.backward_message
        is_fp8 = data_info.is_fp8
        if "dropout" in full_api_name:
            fwd_success_status, fwd_compare_alg_results = self._compare_dropout(bench_output, device_output)
        else:
            fwd_success_status, fwd_compare_alg_results = self._compare_core_wrapper(api_name, bench_output,
                                                                                     device_output, is_fp8)
        if not (bench_grad and device_grad):
            bwd_success_status, bwd_compare_alg_results = (CompareConst.SPACE, [])
        else:
            if "dropout" in full_api_name:
                bwd_success_status, bwd_compare_alg_results = self._compare_dropout(bench_grad[0], device_grad[0])
            else:
                bwd_success_status, bwd_compare_alg_results = self._compare_core_wrapper(api_name, bench_grad,
                                                                                         device_grad, is_fp8)
        if backward_message:
            backward_column = CompareColumn()
            bwd_compare_alg_results = [backward_column.to_column_value(CompareConst.SKIP, backward_message)]
            bwd_success_status = CompareConst.SKIP
        else:
            bwd_success_status = bwd_success_status if bwd_compare_alg_results is not None else CompareConst.SPACE
        result_info = ResultInfo(full_api_name,
                                 fwd_success_status,
                                 bwd_success_status,
                                 fwd_compare_alg_results,
                                 bwd_compare_alg_results,
                                 data_info.rank)
        self.record_results(result_info)
        return fwd_success_status == CompareConst.PASS, bwd_success_status == CompareConst.PASS \
               or bwd_success_status == CompareConst.SPACE

    def _register_compare_func(self):
        registry = StandardRegistry()
        registry.register(CompareConst.ABSOLUTE_THRESHOLD, self._absolute_standard_compare)
        registry.register(CompareConst.BINARY_CONSISTENCY, self._binary_standard_compare)
        registry.register(CompareConst.ULP_COMPARE, self._ulp_compare)
        registry.register(CompareConst.THOUSANDTH_STANDARD, self._thousandth_standard_compare)
        registry.register(CompareConst.BENCHMARK, self._benchmark_compare)
        registry.register(CompareConst.ACCUMULATIVE_ERROR_COMPARE, self._accumulative_error_compare)
        return registry

    def _compare_core_wrapper(self, api_name, bench_output, device_output, is_fp8):
        detailed_result_total = []
        test_final_success = CompareConst.PASS
        if isinstance(bench_output, (list, tuple)):
            status, compare_result, message = [], [], []
            if len(bench_output) > len(device_output):
                status = [CompareConst.ERROR]
                message = ["bench and npu output structure is different."]
            else:
                device_output = device_output[:len(bench_output)]
                for b_out_i, n_out_i in zip(bench_output, device_output):
                    status_i, compare_result_i, message_i = self._compare_core(api_name, b_out_i, n_out_i, is_fp8)
                    status.append(status_i)
                    compare_result.append(compare_result_i)
                    message.append(message_i)
        else:
            status, compare_result, message = self._compare_core(api_name, bench_output, device_output, is_fp8)
        if not isinstance(status, list):
            detailed_result_total.append(compare_result.to_column_value(status, message))
            if status == CompareConst.ERROR:
                test_final_success = CompareConst.ERROR
            elif status == CompareConst.WARNING:
                test_final_success = CompareConst.WARNING
        else:
            for item, item_status in enumerate(status):
                detailed_result_total.append(compare_result[item].to_column_value(item_status, message[item]))
                if item_status == CompareConst.ERROR:
                    test_final_success = CompareConst.ERROR
                elif item_status == CompareConst.WARNING:
                    test_final_success = CompareConst.WARNING
        return test_final_success, detailed_result_total

    def _compare_torch_tensor(self, api_name, bench_output, device_output, compare_column, is_fp8):
        cpu_shape = bench_output.shape
        npu_shape = device_output.shape
        
        npu_dtype = device_output.dtype
        if is_fp8:
            in_dtype = torch.float8_e4m3fn
        else:
            in_dtype = torch.float32
        if npu_dtype == torch.bfloat16:
            bench_output = bench_output.to(torch.float32)
            device_output = device_output.to(torch.float32)
        bench_output = bench_output.cpu().numpy()
        device_output = device_output.cpu().numpy()
        if cpu_shape != npu_shape:
            return CompareConst.ERROR, compare_column, f"The shape of bench{str(cpu_shape)} " \
                                                       f"and npu{str(npu_shape)} not equal."
        if not check_dtype_comparable(bench_output, device_output):
            return CompareConst.ERROR, compare_column, f"Bench out dtype is {bench_output.dtype} but " \
                                                       f"npu output dtype is {device_output.dtype}, cannot compare."
        message = ""
        if bench_output.size == 0:
            return CompareConst.SKIP, compare_column, "There is not bench calculation result."
        if bench_output.dtype in [bool, np.uint8, np.int8, np.int16, np.uint16, np.uint32, np.int32,
                                  np.int64, np.uint64]:
            message += f"Current {bench_output.dtype} data, only perform binary comparison."                      
            err_rate, status, msg = compare_bool_tensor(bench_output, device_output)
            message += msg + "\n"
            compare_column.error_rate = err_rate
            return status, compare_column, message
        else:
            in_and_out_dtype = {
                'dtype': npu_dtype,
                'in_dtype': in_dtype
            }
            status, compare_column, message = self._compare_float_tensor(api_name, bench_output, device_output,
                                                                         compare_column, in_and_out_dtype)
            return status, compare_column, message

    def _perform_comparison(self, api_name, input_data, dtype, in_dtype):
        comparison_func = self.registry.get_comparison_function(api_name, dtype, in_dtype)
        comparison_func(input_data)
            
    def _compare_float_tensor(self, api_name, bench_output, device_output, compare_column, in_and_out_dtype):
        dtype = in_and_out_dtype.get('dtype')
        in_dtype = in_and_out_dtype.get('in_dtype')
        message = ""
        _, abs_bench_with_eps = get_abs_bench_with_eps(bench_output, dtype)
        abs_err = get_abs_err(bench_output, device_output)
        rel_err_orign = get_rel_err_origin(abs_err, abs_bench_with_eps)
        input_data = CompareInput(bench_output, device_output, compare_column, dtype, rel_err_orign)
        if str(dtype) in BENCHMARK_COMPARE_SUPPORT_LIST:
            self._perform_comparison(api_name, input_data, dtype, in_dtype)
        else:
            message += f"The data type {dtype} is not supported for new precision standard."

        cos_res, cos_status, msg = cosine_sim(bench_output, device_output)
        compare_column.cosine_sim = cos_res
        message += msg + "\n"
        if not cos_status:
            message += "Cosine similarity is less than 0.99, consider as error, skip other check and set to SPACE.\n"
            return CompareConst.ERROR, compare_column, message

        max_abs_res, max_abs_status = get_max_abs_err(abs_err)
        compare_column.max_abs_err = max_abs_res
        if max_abs_status:
            message += "Max abs error is less than 0.001, consider as pass, skip other check and set to SPACE.\n"
            return CompareConst.PASS, compare_column, message

        if dtype in [torch.float16, torch.bfloat16]:
            hundred_res, hundred_status = get_rel_err_ratio(rel_err_orign, CompareConst.HUNDRED_RATIO_THRESHOLD)
            compare_column.rel_err_hundredth = hundred_res
            if not hundred_status:
                message += "Relative error is greater than 0.01, consider as error, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.ERROR, compare_column, message
        thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_thousandth = thousand_res
        if dtype in [torch.float16, torch.bfloat16]:
            if thousand_status:
                message += "Relative error is less than 0.001, consider as pass, skip other check and set to SPACE.\n"
                return CompareConst.PASS, compare_column, message
            message += "Relative error is greater than 0.001, consider as warning, skip other check and set to SPACE.\n"
            return CompareConst.WARNING, compare_column, message
        ten_thousand_res, ten_thousand_status = get_rel_err_ratio(
                                                rel_err_orign, CompareConst.TEN_THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_ten_thousandth = ten_thousand_res
        if dtype in [torch.float32, torch.float64]:
            if not thousand_status:
                message += "Relative error is greater than 0.001, consider as error, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.ERROR, compare_column, message
            if not ten_thousand_status:
                message += "Relative error is greater than 0.0001, consider as warning, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.WARNING, compare_column, message
            message += "Relative error is less than 0.0001, consider as pass.\n"
        return CompareConst.PASS, compare_column, message
