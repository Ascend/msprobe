# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

import multiprocessing
from dataclasses import dataclass
from functools import partial
from typing import Any

import pandas as pd
from tqdm import tqdm

from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.common.const import CompareConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.output_postprocess import clean_single_tensor
from msprobe.core.compare.npy_compare import CompareResult, ValidateTensor, compare_ops_apply
from msprobe.core.compare.config import ModeConfig


@dataclass
class ComparisonResult:
    cos_result: list
    euc_dist_result: list
    max_err_result: list
    max_relative_err_result: list
    one_thousand_err_ratio_result: list
    five_thousand_err_ratio_result: list
    err_msgs: list


@dataclass
class CompareInfo:
    op_name: str
    data_name: Any
    dirty_valid_len: Any


# pylint: disable=consider-using-with
def _ms_graph_handle_multi_process(func, result_df, mode):
    process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
    df_chunk_size = len(result_df) // process_num
    if df_chunk_size > 0:
        df_chunks = [result_df.iloc[i : i + df_chunk_size].copy() for i in range(0, len(result_df), df_chunk_size)]
    else:
        df_chunks = [result_df.copy()]

    results = []
    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        logger.error('multiprocess compare failed! Reason: {}'.format(args))

    for df_chunk in df_chunks:
        result = pool.apply_async(func, args=(df_chunk, mode), error_callback=err_call)
        results.append(result)

    pool.close()

    try:
        final_results = [r.get(timeout=3600) for r in results]
    except Exception as e:
        logger.error(f"Task failed with exception: {e}")
        pool.terminate()
        raise CompareException(CompareException.MULTIPROCESS_ERROR) from e

    pool.join()
    return pd.concat(final_results, ignore_index=True)


def check_accuracy(cos, max_abs_err):
    if cos == CompareConst.SHAPE_UNMATCH:
        return CompareConst.ACCURACY_CHECK_UNMATCH
    if CompareConst.NONE in (cos, max_abs_err):
        return CompareConst.NONE
    if cos == "N/A" or max_abs_err == "N/A":
        return CompareConst.ACCURACY_CHECK_NO
    try:
        cos, max_abs_err = float(cos), float(max_abs_err)
    except ValueError:
        logger.warning("Cosine or MaxAbsErr can not get float value.")
        return CompareConst.NONE
    if cos < CompareConst.COS_THRESHOLD and max_abs_err > CompareConst.MAX_ABS_ERR_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    if cos < CompareConst.COS_MAX_THRESHOLD or max_abs_err > CompareConst.MAX_ABS_ERR_MAX_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    return CompareConst.ACCURACY_CHECK_YES


class CompareRealData:
    def __init__(self, file_reader, mode_config: ModeConfig, cross_frame):
        self.file_reader = file_reader
        self.mode_config = mode_config
        self.cross_frame = cross_frame

    @staticmethod
    def read_dump_data(result_df):
        """
        构建算子-真实数据名字典。key：npu_op_name, value: tensor_pair
        """
        try:
            npu_name_list = result_df.loc[0:, CompareConst.NPU_NAME].tolist()
            bench_name_list = result_df.loc[0:, CompareConst.BENCH_NAME].tolist()
            dump_tensor_pair_list = result_df.loc[0:, CompareConst.DATA_NAME].tolist()
            op_name_mapping_dict = {}
            for npu_name, bench_name, dump_tensor_pair in zip(npu_name_list, bench_name_list, dump_tensor_pair_list):
                op_name_mapping_dict[str(npu_name) + str(bench_name)] = dump_tensor_pair
            return op_name_mapping_dict
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        except KeyError as e:
            logger.error('result dataframe elements can not be access.')
            raise CompareException(CompareException.INVALID_KEY_ERROR) from e

    @staticmethod
    def clean_dirty_data(tensor, compare_info: CompareInfo):
        if compare_info.dirty_valid_len == -1 or "output" not in compare_info.op_name:
            return tensor

        return clean_single_tensor(tensor, compare_info.dirty_valid_len)

    @staticmethod
    def _save_cmp_result(offset, result: ComparisonResult, result_df, lock):
        """
            Save comparison results into the result DataFrame with thread safety.
        Args:
            offset: offset for index
            result: data struct of ComparisonResult
            result_df: result of DataFrame
            lock: thread lock

        Returns:
            comparison results in DataFrame
        """

        lock.acquire()
        try:
            for i, cos_item in enumerate(result.cos_result):
                process_index = i + offset
                result_df.loc[process_index, CompareConst.COSINE] = cos_item
                result_df.loc[process_index, CompareConst.EUC_DIST] = result.euc_dist_result[i]
                result_df.loc[process_index, CompareConst.MAX_ABS_ERR] = result.max_err_result[i]
                result_df.loc[process_index, CompareConst.MAX_RELATIVE_ERR] = result.max_relative_err_result[i]
                result_df.loc[process_index, CompareConst.ONE_THOUSANDTH_ERR_RATIO] = (
                    result.one_thousand_err_ratio_result
                )[i]
                result_df.loc[process_index, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = (
                    result.five_thousand_err_ratio_result
                )[i]
                result_df.loc[process_index, CompareConst.ERROR_MESSAGE] += result.err_msgs[i]
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        except IndexError as e:
            logger.error('result dataframe elements can not be access.')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e
        finally:
            lock.release()

    def compare_by_op(self, npu_info, bench_info, input_param):
        """
        按算子进行数据对比的主流程函数

        整体流程：
            1. 根据算子名定位 dump 文件
            2. 读取 tensor 数据
            3. 校验 tensor 数据合法性

        参数:
            npu_info (NPUInfo): NPU 侧算子信息
            bench_info (BenchInfo): Bench 侧算子信息
            input_param: npu_path/bench_path等参数
        返回:
            list:
                对比结果列表，包含：余弦相似度、欧氏距离、最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率、错误信息
        """

        # 1. 根据算子名定位 dump 文件
        dump_result = self.locate_dump_file(npu_info, bench_info)
        if dump_result.err_msg:
            return self.build_result(dump_result, npu_info, bench_info)

        # 2. 读取 tensor 数据
        read_result = self.read_tensor(dump_result, input_param, npu_info, bench_info)
        if read_result.err_msg:
            return self.build_result(read_result, npu_info, bench_info)

        # 3. 校验 tensor 数据合法性
        validate_tensor = ValidateTensor()
        validate_result = validate_tensor.check_tensor(read_result)
        return self.build_result(validate_result, npu_info, bench_info)

    @staticmethod
    def locate_dump_file(npu_info, bench_info):
        """
        根据算子名称查找对应的 dump 文件名

        参数:
            npu_info (NPUInfo): NPU 侧算子信息
            bench_info (BenchInfo): Bench 侧算子信息
        返回:
            CompareResult: 包含 dump 文件名或错误信息的结果对象
        """
        npu_data_name = npu_info.data_name
        bench_data_name = bench_info.data_name

        if str(npu_data_name) == CompareConst.NO_REAL_DATA_FLAG:
            return CompareResult(
                CompareConst.NO_REAL_DATA, CompareConst.NO_REAL_DATA, True, "NPU does not have data file."
            )

        if str(bench_data_name) == CompareConst.NO_REAL_DATA_FLAG:
            return CompareResult(
                CompareConst.NO_REAL_DATA, CompareConst.NO_REAL_DATA, True, "Bench does not have data file."
            )

        if (str(npu_data_name) == CompareConst.N_A) ^ (str(bench_data_name) == CompareConst.N_A):
            return CompareResult(
                CompareConst.API_UNMATCH, CompareConst.API_UNMATCH, True, "Bench api/module unmatched."
            )

        return CompareResult(npu_data_name, bench_data_name, False, "")

    def read_tensor(self, dump_result, input_param, npu_info, bench_info):
        """
        从 dump 文件中读取 tensor 数据

        参数:
            dump_result (CompareResult): 上一步定位到的 dump 文件名
        返回:
            CompareResult: 返回读取到的 tensor 数据或错误信息
        """
        data_path_dict = {
            "npu_dir": input_param.get(CompareConst.NPU_DUMP_DATA_DIR),
            "npu_data_name": dump_result.n_value,
            "bench_dir": input_param.get(CompareConst.BENCH_DUMP_DATA_DIR),
            "bench_data_name": dump_result.b_value,
        }

        try:
            n_value, b_value = self.file_reader(data_path_dict, self.cross_frame, self.mode_config.backend)
            if not self.mode_config.consistent_check:
                n_value = self.clean_dirty_data(n_value, npu_info)
                b_value = self.clean_dirty_data(b_value, bench_info)
            return CompareResult(n_value, b_value, False, "")

        except IOError as error:
            return CompareResult(
                CompareConst.READ_NONE,
                CompareConst.READ_NONE,
                True,
                f"Dump file: {error.filename} not found or read failed.",
            )

        except (FileCheckException, CompareException):
            return CompareResult(
                CompareConst.READ_NONE,
                CompareConst.READ_NONE,
                True,
                f"Dump file: {dump_result.n_value} or {dump_result.b_value} not found or read failed.",
            )

    def build_result(self, result, npu_info, bench_info):
        """
        构建最终的对比结果
        负责调用指标计算函数，并补充模糊匹配提示信息。
        """
        result_list, err_msg = compare_ops_apply(result.n_value, result.b_value, result.error_flag, result.err_msg)
        npu_op_name = npu_info.op_name
        bench_op_name = bench_info.op_name
        if self.mode_config.fuzzy_match and npu_op_name != bench_op_name and bench_op_name != CompareConst.N_A:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list

    def compare_ops(self, idx, dump_path_dict, result_df, lock, input_param):
        cos_result = []
        euc_dist_result = []
        max_err_result = []
        max_relative_err_result = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        err_mess = []

        is_print_compare_log = input_param.get("is_print_compare_log")

        for i in range(len(result_df)):
            current_row = result_df.iloc[i]
            npu_op_name = current_row[CompareConst.NPU_NAME]
            bench_op_name = current_row[CompareConst.BENCH_NAME]
            data_name_pair = dump_path_dict.get(str(npu_op_name) + str(bench_op_name))
            npu_data_name, bench_data_name = data_name_pair

            npu_dirty_valid_len = -1
            bench_dirty_valid_len = -1
            if not self.mode_config.consistent_check:
                dirty_valid_len_pair = current_row[CompareConst.DIRTY_VALID_LEN]
                npu_dirty_valid_len, bench_dirty_valid_len = dirty_valid_len_pair  # 固定结构为 [npu len, bench len]

            npu_info = CompareInfo(npu_op_name, npu_data_name, npu_dirty_valid_len)
            bench_info = CompareInfo(bench_op_name, bench_data_name, bench_dirty_valid_len)

            if is_print_compare_log:
                logger.info("start compare: {}".format(npu_op_name))

            (
                cos_sim,
                euc_dist,
                max_abs_err,
                max_relative_err,
                one_thousand_err_ratio,
                five_thousand_err_ratio,
                err_msg,
            ) = self.compare_by_op(npu_info, bench_info, input_param)

            if is_print_compare_log:
                if "does not have data file" in err_msg:
                    logger.info(f"[{npu_op_name}] Compare result: {err_msg} ")
                elif "Bench api/module unmatched" in err_msg:
                    logger.info(f"[{npu_op_name}] Compare result: {err_msg} ")
                else:
                    logger.info(
                        f"[{npu_op_name}] Compare result: cosine {cos_sim}, euc_dist {euc_dist}, "
                        f"max_abs_err {max_abs_err}, max_relative_err {max_relative_err}, "
                        f"one_thousand_err_ratio {one_thousand_err_ratio}, "
                        f"five_thousand_err_ratio {five_thousand_err_ratio}, {err_msg}"
                    )
            cos_result.append(cos_sim)
            euc_dist_result.append(euc_dist)
            max_err_result.append(max_abs_err)
            max_relative_err_result.append(max_relative_err)
            one_thousand_err_ratio_result.append(one_thousand_err_ratio)
            five_thousand_err_ratio_result.append(five_thousand_err_ratio)
            err_mess.append(err_msg)

        cr = ComparisonResult(
            cos_result=cos_result,
            euc_dist_result=euc_dist_result,
            max_err_result=max_err_result,
            max_relative_err_result=max_relative_err_result,
            one_thousand_err_ratio_result=one_thousand_err_ratio_result,
            five_thousand_err_ratio_result=five_thousand_err_ratio_result,
            err_msgs=err_mess,
        )

        return self._save_cmp_result(idx, cr, result_df, lock)

    def do_multi_process(self, input_param, result_df):
        try:
            result_df = self._handle_multi_process(
                self.compare_ops, input_param, result_df, multiprocessing.Manager().RLock()
            )
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e

    def _handle_multi_process(self, func, input_param, result_df, lock):
        process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
        op_name_mapping_dict = self.read_dump_data(result_df)

        df_chunk_size = len(result_df) // process_num
        if df_chunk_size > 0:
            df_chunks = [result_df.iloc[i : i + df_chunk_size].copy() for i in range(0, len(result_df), df_chunk_size)]
        else:
            df_chunks = [result_df.copy()]

        results = []
        progress_bar = tqdm(total=len(result_df), desc="API/Module Item Compare Process", unit="row", ncols=100)

        def update_progress(size, progress_lock, extra_param=None):
            with progress_lock:
                progress_bar.update(size)

        pool = multiprocessing.Pool(process_num)

        def err_call(args):
            logger.error('multiprocess compare failed! Reason: {}'.format(args))

        for process_idx, df_chunk in enumerate(df_chunks):
            idx = df_chunk_size * process_idx
            chunk_size = len(df_chunk)
            result = pool.apply_async(
                func,
                args=(idx, op_name_mapping_dict, df_chunk, lock, input_param),
                error_callback=err_call,
                callback=partial(update_progress, chunk_size, lock),
            )
            results.append(result)

        pool.close()

        try:
            final_results = [r.get(timeout=3600) for r in results]
        except Exception as e:
            logger.error(f"Task failed with exception: {e}")
            pool.terminate()
            raise CompareException(CompareException.MULTIPROCESS_ERROR) from e

        pool.join()
        return pd.concat(final_results, ignore_index=True)
