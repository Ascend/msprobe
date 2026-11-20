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

import os
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    write_df_to_csv,
    create_directory,
    check_path_before_create,
    load_npy,
)
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.compare.npy_compare import compare_ops_apply
from msprobe.core.compare.utils import check_input_param_path_and_framework
from msprobe.core.compare.multiprocessing_compute import check_accuracy
from msprobe.mindspore.compare.utils import check_name_map_dict


def common_dir_compare(args) -> Optional[pd.DataFrame]:
    """
    高级目录比对函数，完全镜像输入目录结构
    """

    check_input_param_path_and_framework(args, target_framework=Const.MS_FRAMEWORK)

    npu_root = Path(args.target_path)
    bench_root = Path(args.golden_path)
    name_map_dict = getattr(args, 'map_dict', {})
    check_name_map_dict(name_map_dict)
    file_tree = build_mirror_file_tree(npu_root, bench_root)

    # 处理文件比对
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    partial(process_directory_pair, name_map_dict=name_map_dict, output_dir=args.output_path),
                    file_tree.items(),
                ),
                total=len(file_tree),
                desc="Processing directories",
            )
        )


def process_directory_pair(item: Tuple[Path, Tuple[Path, Path]], name_map_dict: Dict, output_dir: str):
    """
    处理一个目录对

    Args:
        item: (相对路径, (npu目录, bench目录))元组
        output_dir: 输出根目录

    Returns:
        比对结果的DataFrame（仅平铺结构时返回）
    """
    rel_path, (npu_dir, bench_dir) = item

    # 创建镜像输出目录
    output_path = Path(output_dir) / rel_path
    create_directory(output_path)

    # 生成文件映射
    npu_files = find_npy_files(npu_dir)
    bench_files = find_npy_files(bench_dir)
    map_dict = generate_map_dict(npu_files, bench_files, name_map_dict)

    if not map_dict:
        logger.warning(f"No file pairs found in {rel_path}")
        return None

    # 执行比对
    result_df = do_multi_process(process_chunk, map_dict)
    check_path_before_create(output_path)
    # 保存结果
    result_path = os.path.join(output_path, 'result.csv')
    write_df_to_csv(result_df, result_path, encoding='utf-8-sig')
    logger.info(f"Results saved to {result_path}")
    return None


def build_mirror_file_tree(npu_root: Path, bench_root: Path) -> Dict[Path, Tuple[Path, Path]]:
    """
    构建镜像文件树，键为相对路径，值为(npu_path, bench_path)元组

    Args:
        npu_root: NPU数据根目录
        bench_root: 基准数据根目录

    Returns:
        文件树字典
    """
    file_tree = {}

    # 遍历NPU目录构建树结构
    # 使用os.walk遍历目录,限制深度为10层
    for root, dirs, files in os.walk(npu_root):
        # 计算当前目录深度
        depth = len(Path(root).relative_to(npu_root).parts)
        if depth > 10:
            dirs.clear()  # 清空dirs列表以阻止继续递归
            continue

        # 检查当前目录下是否有npy文件
        if any(f.endswith('.npy') for f in files):
            # 获取相对路径
            dir_path = Path(root).relative_to(npu_root)
            npu_dir_pair = os.path.join(npu_root, dir_path)
            bench_dir_pair = os.path.join(bench_root, dir_path)

            try:
                check_file_or_directory_path(bench_dir_pair, isdir=True)
            except FileCheckException:
                continue

            # 添加到文件树
            if dir_path not in file_tree:
                file_tree[dir_path] = (npu_dir_pair, bench_dir_pair)

    return file_tree


def find_npy_files(directory):
    npy_files_dict = {}
    # 限制递归深度为1层,即只遍历当前目录和其直接子目录
    for root, dirs, files in os.walk(directory, topdown=True):
        # 计算当前目录深度
        depth = root[len(directory) :].count(os.sep)
        # 如果深度超过10层则跳过
        if depth > 10:
            dirs.clear()
        for file in files:
            if file.endswith(".npy"):
                # 正确移除文件扩展名
                base_name = os.path.splitext(file)
                if not base_name or len(base_name) < 1:
                    logger.warning("Invalid file encountered.")
                    continue
                file_name = base_name[0]

                logger.info(f"Generating file info for file: {file}")

                # 使用一致的分割逻辑
                file_ele = file_name.split('_')

                if len(file_ele) < 2:
                    continue

                key = '_'.join(file_ele[:-2])
                if key:
                    # 文件的完整路径
                    value = os.path.join(root, file)
                    # 添加到字典中
                    if key not in npy_files_dict:
                        npy_files_dict[key] = []
                    npy_files_dict[key].append(value)
    return npy_files_dict


def generate_map_dict(npu_file_dict, bench_file_dict, name_map_dict=None):
    result_dict = {}
    for k, npu_file_list in npu_file_dict.items():
        bench_file_list = bench_file_dict.get(k)
        if not bench_file_list and k in name_map_dict:
            bench_file_list = bench_file_dict.get(name_map_dict.get(k))
        bench_length = len(bench_file_list)
        if not (bench_file_list and bench_length):
            continue
        for i, npu_file in enumerate(npu_file_list):
            if i >= bench_length:
                break
            bench_file = bench_file_list[i]
            result_dict[f"{k}_{i}"] = (npu_file, bench_file)
    return result_dict


def do_multi_process(func, map_dict):
    lock = multiprocessing.Manager().RLock()
    result_len = len(map_dict)
    process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
    # every block size
    df_chunk_size = result_len // process_num

    # generate the same len of map_dict df
    result_df = initialize_result_df(result_len)
    if df_chunk_size > 0:
        df_chunks = [result_df.iloc[i : i + df_chunk_size] for i in range(0, len(result_df), df_chunk_size)]
    else:
        df_chunks = [result_df]
        process_num = 1
    logger.info(f"Using {process_num} processes with chunk size {df_chunk_size}")

    # 分割字典
    map_chunks = split_dict(map_dict, df_chunk_size)

    # 创建结果列表和进程池
    results = []
    # 进程池生命周期在下方显式管理(close/terminate/join)，不使用with
    pool = multiprocessing.Pool(process_num)  # pylint: disable=consider-using-with

    progress_bar = tqdm(total=len(result_df), desc="API/Module Item Compare Process", unit="row", ncols=100)

    def update_progress(size, progress_lock, extra_param=None):
        with progress_lock:
            progress_bar.update(size)

    def err_call(args):
        logger.error('multiprocess compare failed! Reason: {}'.format(args))

    results = []

    # 提交任务到进程池
    for process_idx, (df_chunk, map_chunk) in enumerate(zip(df_chunks, map_chunks)):
        start_idx = df_chunk_size * process_idx
        result = pool.apply_async(
            func,
            args=(df_chunk, start_idx, map_chunk, lock),
            error_callback=err_call,
            callback=partial(update_progress, len(map_chunk), lock),
        )
        results.append(result)
    pool.close()

    try:
        final_results = [r.get(timeout=3600) for r in results]
    except Exception as e:
        logger.error(f"Task failed with exception: {e}")
        pool.terminate()
        return pd.DataFrame({})
    # 等待所有任务完成
    pool.join()
    return pd.concat(final_results, ignore_index=True)


def initialize_result_df(total_size):
    """预分配结果DataFrame"""
    columns = [
        CompareConst.NAME,
        CompareConst.NPU_DTYPE,
        CompareConst.BENCH_DTYPE,
        CompareConst.NPU_SHAPE,
        CompareConst.BENCH_SHAPE,
        CompareConst.COSINE,
        CompareConst.EUC_DIST,
        CompareConst.MAX_ABS_ERR,
        CompareConst.MAX_RELATIVE_ERR,
        CompareConst.ONE_THOUSANDTH_ERR_RATIO,
        CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
        CompareConst.NPU_MAX,
        CompareConst.NPU_MIN,
        CompareConst.NPU_MEAN,
        CompareConst.NPU_NORM,
        CompareConst.BENCH_MAX,
        CompareConst.BENCH_MIN,
        CompareConst.BENCH_MEAN,
        CompareConst.BENCH_NORM,
        CompareConst.ACCURACY,
        CompareConst.ERROR_MESSAGE,
        CompareConst.DATA_NAME,
    ]
    return pd.DataFrame(index=range(total_size), columns=columns)


def split_dict(input_dict, chunk_size):
    """将字典按指定chunk_size分割"""
    items = list(input_dict.items())
    if chunk_size > 0:
        return [dict(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return [input_dict]


def get_tensor_stats(tensor: np.ndarray) -> Tuple[float, float, float, float]:
    """获取张量的统计信息"""
    t_max = np.max(tensor)
    t_min = np.min(tensor)
    t_mean = np.mean(tensor)
    t_l2norm = np.linalg.norm(tensor)
    return t_max, t_min, t_mean, t_l2norm


def process_chunk(df, start_idx, map_chunk, lock):
    """处理一个数据块"""
    err_mess = []
    results = []
    for name, file_pair in map_chunk.items():
        err_msg = ""
        npu_file, bench_file = file_pair
        n_value = load_npy(npu_file)
        # if need to support cross frame b_value need to add load_pt
        b_value = load_npy(bench_file)
        error_flag = False

        err_list, err_msg = compare_ops_apply(n_value, b_value, error_flag, err_msg)
        cos_sim, euc_dist, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio = err_list
        a_max, a_min, a_mean, a_l2norm = get_tensor_stats(n_value)
        b_max, b_min, b_mean, b_l2norm = get_tensor_stats(b_value)
        err_mess.append(err_msg)
        # 使用示例
        result = ComparisonResult(
            name=name,  # CompareConst.NAME
            npu_dtype=n_value.dtype,  # CompareConst.NPU_DTYPE
            bench_dtype=b_value.dtype,  # CompareConst.BENCH_DTYPE
            npu_shape=n_value.shape,  # CompareConst.NPU_SHAPE
            bench_shape=b_value.shape,  # CompareConst.BENCH_SHAPE
            cosine=cos_sim,  # CompareConst.COSINE
            euc_dist=euc_dist,  # CompareConst.EUC_DIST
            max_abs_err=max_abs_err,  # CompareConst.MAX_ABS_ERR
            max_relative_err=max_relative_err,  # CompareConst.MAX_RELATIVE_ERR
            one_thousandth_err_ratio=one_thousand_err_ratio,  # CompareConst.ONE_THOUSANDTH_ERR_RATIO
            five_thousandth_err_ratio=five_thousand_err_ratio,  # CompareConst.FIVE_THOUSANDTHS_ERR_RATIO
            npu_max=a_max,  # CompareConst.NPU_MAX
            npu_min=a_min,  # CompareConst.NPU_MIN
            npu_mean=a_mean,  # CompareConst.NPU_MEAN
            npu_norm=a_l2norm,  # CompareConst.NPU_NORM
            bench_max=b_max,  # CompareConst.BENCH_MAX
            bench_min=b_min,  # CompareConst.BENCH_MIN
            bench_mean=b_mean,  # CompareConst.BENCH_MEAN
            bench_norm=b_l2norm,  # CompareConst.BENCH_NORM
            accuracy=check_accuracy(cos_sim, max_abs_err),  # CompareConst.ACCURACY
            error_message=err_msg,  # CompareConst.ERROR_MESSAGE
            data_name=[npu_file, bench_file],  # CompareConst.DATA_NAME
        )
        results.append(result)
    return _save_part_df(df, start_idx, results, lock)


@dataclass
class ComparisonResult:
    name: str  # CompareConst.NAME
    npu_dtype: Any  # CompareConst.NPU_DTYPE
    bench_dtype: Any  # CompareConst.BENCH_DTYPE
    npu_shape: Tuple[int, ...]  # CompareConst.NPU_SHAPE
    bench_shape: Tuple[int, ...]  # CompareConst.BENCH_SHAPE
    cosine: float  # Cons   t.COSINE
    euc_dist: float  # CompareConst.EUC_DIST
    max_abs_err: float  # CompareConst.MAX_ABS_ERR
    max_relative_err: float  # CompareConst.MAX_RELATIVE_ERR
    one_thousandth_err_ratio: float  # CompareConst.ONE_THOUSANDTH_ERR_RATIO
    five_thousandth_err_ratio: float  # CompareConst.FIVE_THOUSANDTHS_ERR_RATIO
    npu_max: float  # CompareConst.NPU_MAX
    npu_min: float  # CompareConst.NPU_MIN
    npu_mean: float  # CompareConst.NPU_MEAN
    npu_norm: float  # CompareConst.NPU_NORM
    bench_max: float  # CompareConst.BENCH_MAX
    bench_min: float  # CompareConst.BENCH_MIN
    bench_mean: float  # CompareConst.BENCH_MEAN
    bench_norm: float  # CompareConst.BENCH_NORM
    accuracy: bool  # CompareConst.ACCURACY
    error_message: str  # CompareConst.ERROR_MESSAGE
    data_name: List[str]  # CompareConst.DATA_NAME


def _save_part_df(df, start_idx, results, lock):
    with lock:
        try:
            for i, result in enumerate(results):
                process_index = i + start_idx
                df.loc[process_index, CompareConst.NAME] = result.name
                df.loc[process_index, CompareConst.NPU_DTYPE] = result.npu_dtype
                df.loc[process_index, CompareConst.BENCH_DTYPE] = result.bench_dtype
                df.loc[process_index, CompareConst.NPU_SHAPE] = str(result.npu_shape)  # 通常将tuple转为字符串存储
                df.loc[process_index, CompareConst.BENCH_SHAPE] = str(result.bench_shape)
                df.loc[process_index, CompareConst.COSINE] = result.cosine
                df.loc[process_index, CompareConst.EUC_DIST] = result.euc_dist
                df.loc[process_index, CompareConst.MAX_ABS_ERR] = result.max_abs_err
                df.loc[process_index, CompareConst.MAX_RELATIVE_ERR] = result.max_relative_err
                df.loc[process_index, CompareConst.ONE_THOUSANDTH_ERR_RATIO] = result.one_thousandth_err_ratio
                df.loc[process_index, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = result.five_thousandth_err_ratio
                df.loc[process_index, CompareConst.NPU_MAX] = result.npu_max
                df.loc[process_index, CompareConst.NPU_MIN] = result.npu_min
                df.loc[process_index, CompareConst.NPU_MEAN] = result.npu_mean
                df.loc[process_index, CompareConst.NPU_NORM] = result.npu_norm
                df.loc[process_index, CompareConst.BENCH_MAX] = result.bench_max
                df.loc[process_index, CompareConst.BENCH_MIN] = result.bench_min
                df.loc[process_index, CompareConst.BENCH_MEAN] = result.bench_mean
                df.loc[process_index, CompareConst.BENCH_NORM] = result.bench_norm
                df.loc[process_index, CompareConst.ACCURACY] = result.accuracy
                df.loc[process_index, CompareConst.ERROR_MESSAGE] = result.error_message
                df.loc[process_index, CompareConst.DATA_NAME] = str(result.data_name)  # 列表转为字符串存储
            return df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        except IndexError as e:
            logger.error('result dataframe elements can not be access.')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e
