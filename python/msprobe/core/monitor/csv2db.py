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
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from msprobe.core.common.const import MonitorConst, Data2DBConst
from msprobe.core.common.data_concat import (
    ConcatPlan,
    DatasetAxis,
    build_concat_plan,
    build_single_plan,
)
from msprobe.core.common.file_utils import read_csv
from msprobe.core.common.log import logger
from msprobe.core.monitor.db_utils import MonitorDB, update_ordered_dict, get_ordered_list
from msprobe.core.common.data2db_utils import process_tensor_value
from tqdm import tqdm


@dataclass
class CSV2DBConfig:
    """Configuration for CSV to database conversion"""

    target_output_dirs: Dict[int, str]
    process_num: int = 1
    db_file: str = None
    micro_step: bool = True
    mapping: Dict = None
    compare_output_dirs: Optional[Dict[int, str]] = None
    concat_dim: str = Data2DBConst.CONCAT_DIM_STEP
    concat_plan: Optional[ConcatPlan] = None


def get_metric_name_from_filename(file_name: str) -> Tuple[str, str, str]:
    """Extract metric name from filename"""
    parts = file_name.split('_')
    if len(parts) < 2:
        return ""
    metric_candidate = "_".join(parts[:-1])
    if metric_candidate in Data2DBConst.METRICS_TRENDVIS_SUPPORTED:
        metric_name = metric_candidate
    else:
        return ""
    # step pattern
    pattern = f"{re.escape(metric_name)}{MonitorConst.CSV_FILE_PATTERN}"
    match = re.match(pattern, file_name)
    if not match:
        return ""
    return metric_name


def _pre_scan_single_rank(rank: int, files: List[str], mapping: Dict[str, str], use_micro_step: bool) -> Dict[str, Any]:
    """Pre-scan files for a single rank to collect metadata"""
    micro_step_dict = defaultdict(int)  # 记录step下的micro_step数量, micro_step最大值
    metric_stats = defaultdict(set)
    targets = defaultdict(dict)  # 记录targets
    step_keys = set()  # 实际出现过的迭代步号

    for file_path in files:
        file_name = os.path.basename(file_path)
        metric_name = get_metric_name_from_filename(file_name)
        if not metric_name:
            continue

        data = read_csv(file_path)
        stats = [k for k in data.columns if k in Data2DBConst.OP_TRENDVIS_SUPPORTED]
        metric_stats[metric_name].update(stats)

        for _, row in data.iterrows():
            try:
                name = str(row[MonitorConst.HEADER_NAME])
                vpp_stage = int(row['vpp_stage'])
                step = int(row.get('step', Data2DBConst.DEFAULT_INT_VALUE))
                micro_step = int(row.get('micro_step', Data2DBConst.DEFAULT_INT_VALUE))
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Skip invalid row in {file_path}: {e}")
                continue
            step_keys.add(step)
            # 使用 mapping
            if mapping:
                for key, value in mapping.items():
                    name = name.replace(key, value)

            if use_micro_step:
                if micro_step >= micro_step_dict[step]:
                    # count of micro step
                    micro_step_dict[step] = micro_step + 1
                micro_step = Data2DBConst.DEFAULT_INT_VALUE

            # 记录target
            target = (name, vpp_stage, micro_step)
            targets[metric_name][target] = None

    return {
        'rank': rank,
        'micro_step_dict': dict(micro_step_dict),
        'metric_stats': metric_stats,
        'targets': {m: list(t.keys()) for m, t in targets.items()},
        'step_keys': sorted(step_keys),
    }


def _scan_dataset(
    target_output_dirs: Dict[int, str],
    mapping: Dict[str, str],
    use_micro_step: bool,
    process_num: int,
) -> Tuple[List[Tuple[int, List[str]]], List[Dict[str, Any]]]:
    """扫描一份数据, 收集每个 rank 下的 csv 文件并预扫描其内容。

    :param target_output_dirs: rank -> 目录
    :param mapping: 名称映射
    :param use_micro_step: 是否启用微步计数
    :param process_num: 并行进程数
    :return: (批次任务列表, 预扫描结果列表)
    """
    rank_files = defaultdict(list)
    for rank, dir_path in target_output_dirs.items():
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if get_metric_name_from_filename(file):
                rank_files[rank].append(os.path.join(dir_path, file))

    if not rank_files:
        return [], []

    batch_size = Data2DBConst.FILE_BATCH_SIZE  # 默认每批次100个文件
    tasks = []  # 每个元素: (rank, batch_files)

    for rank, files in rank_files.items():
        if not files:
            logger.warning(f"Rank {rank} has no files, skipped")
            continue
        # 按批次切分文件列表（保持原始顺序）
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            tasks.append((rank, batch_files))

    total_batches = len(tasks)
    total_files = sum(len(files) for _, files in tasks)
    logger.info(
        f"Split {len(rank_files)} ranks into {total_batches} batches "
        f"(batch_size={batch_size}, total_files={total_files})"
    )

    # Parallel
    with ProcessPoolExecutor(max_workers=process_num) as executor:
        futures = {
            executor.submit(_pre_scan_single_rank, rank, files, mapping or {}, use_micro_step): rank
            for rank, files in tasks
        }

        results = []
        with tqdm(total=len(futures), desc="Pre-scanning batches") as pbar:
            for future in as_completed(futures):
                rank = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch failed for rank {rank}: {e}")
                pbar.update(1)

    return tasks, results


def _collect_axis(results: List[Dict[str, Any]], use_micro_step: bool) -> DatasetAxis:
    """从预扫描结果中归纳出一份数据的 step 轴与 rank 轴信息。"""
    step_keys = set()
    micro_counts: Dict[int, int] = defaultdict(int)
    rank_ids = []
    max_rank = 0

    for res in results:
        rank = int(res["rank"])
        rank_ids.append(rank)
        max_rank = max(max_rank, rank)
        step_keys.update(res.get("step_keys", []))
        if use_micro_step:
            for step, micro_val in res.get("micro_step_dict", {}).items():
                micro_counts[step] = max(micro_counts[step], micro_val)

    return DatasetAxis(
        step_keys=sorted(step_keys),
        # 关闭微步计数时微步信息不参与拼接, 明确记为未知而不是空字典
        micro_counts=dict(micro_counts) if use_micro_step else None,
        rank_ids=sorted(set(rank_ids)),
        max_rank=max_rank,
    )


def _pre_scan(config: CSV2DBConfig, monitor_db: MonitorDB):
    """Pre-scan all targets, metrics, and statistics"""
    logger.info("Scanning dimensions...")
    datasets = [(0, config.target_output_dirs)]
    if config.compare_output_dirs:
        datasets.append((1, config.compare_output_dirs))

    all_tasks = []
    all_results = []
    axes = []
    for dataset_id, output_dirs in datasets:
        tasks, results = _scan_dataset(output_dirs, config.mapping or {}, config.micro_step, config.process_num)
        all_tasks.extend((dataset_id, rank, files) for rank, files in tasks)
        all_results.append(results)
        axes.append(_collect_axis(results, config.micro_step))

    if not axes[0].step_keys:
        # 被测数据没有可用 CSV 时必须报错退出, 不能只导入对比数据
        logger.error(
            "No valid CSV data found in the data path. "
            "Check that the data path contains monitor CSV files with the expected columns."
        )
        return [], {}, {}

    if len(axes) == 2 and not axes[1].step_keys:
        # 对比目录存在但没有可用 CSV 时降级为单数据源, 避免生成跨度为 0 的拼接计划
        logger.warning(
            "No valid CSV data found in the compare data path. "
            "Two data sources must be of the same format. Fall back to the single data source."
        )
        all_tasks = [task for task in all_tasks if task[0] == 0]
        all_results = all_results[:1]
        axes = axes[:1]
        config.compare_output_dirs = None

    if not all_tasks:
        logger.warning("No valid CSV files found")
        return [], {}, {}

    # Aggregate targets and stats
    targets = defaultdict(OrderedDict)
    all_stats = set()
    metric_stats_agg = defaultdict(set)

    for results in all_results:
        for res in results:
            for metric, t_list in res['targets'].items():
                targets[metric] = update_ordered_dict(targets[metric], t_list)
            for metric, stats in res['metric_stats'].items():
                metric_stats_agg[metric].update(stats)
                all_stats.update(stats)

    # 计算拼接计划
    if len(axes) == 2:
        concat_plan = build_concat_plan(axes[0], axes[1], config.concat_dim, config.micro_step)
    else:
        concat_plan = build_single_plan(axes[0], config.micro_step)
    config.concat_plan = concat_plan
    for warning in concat_plan.warnings:
        logger.warning(warning)

    # 插入数据库
    global_stats = {
        "min_step": concat_plan.min_step,
        "max_step": concat_plan.max_step,
        "max_rank": concat_plan.max_rank,
        **metric_stats_agg,
    }
    monitor_db.init_schema()
    monitor_db.insert_dimensions(targets)
    monitor_db.init_global_stats_data(global_stats)
    monitor_db.create_trend_data(get_ordered_list(all_stats, Data2DBConst.OP_TRENDVIS_SUPPORTED))

    metric_id_dict = monitor_db.get_metric_mapping()
    target_dict = monitor_db.get_target_mapping()
    monitor_db.extract_tags_from_processed_targets(targets, metric_id_dict, target_dict)

    return all_tasks, metric_id_dict, target_dict


def process_single_rank(
    task: Tuple[int, int, List[str]],
    metric_id_dict: Dict[str, Tuple[int, List[str]]],
    target_dict: Dict[Tuple[str, int, int], int],
    config: CSV2DBConfig,
) -> int:
    """Process data import for a single rank

    :param task: (dataset_id, rank, files), dataset_id 为 0 表示被测数据, 1 表示对比数据
    :param metric_id_dict: metric 名称到 ID 及统计量的映射
    :param target_dict: target 三元组到 ID 的映射
    :param config: 导入配置, 其中 concat_plan 提供迭代步映射与 rank 偏移
    """
    dataset_id, rank, files = task
    # rank 来源于目录名中的正则捕获, 是字符串, 统一转为整数以便叠加 rank 偏移
    rank = int(rank)
    mapper = config.concat_plan.mapper_for(dataset_id)
    rank_offset = config.concat_plan.rank_offset_for(dataset_id)
    db = MonitorDB(config.db_file)
    total_inserted = 0
    batch_data = []

    all_stats = get_ordered_list(
        set().union(*(stats for _, stats in metric_id_dict.values())), Data2DBConst.OP_TRENDVIS_SUPPORTED
    )

    for file_path in files:
        filename = os.path.basename(file_path)
        metric_name = get_metric_name_from_filename(filename)
        if not metric_name or metric_name not in metric_id_dict:
            continue

        metric_id, _ = metric_id_dict[metric_name]

        for _, row in read_csv(file_path).iterrows():
            try:
                name = str(row[MonitorConst.HEADER_NAME])
                vpp_stage = int(row['vpp_stage'])
                original_step = int(row.get('step', Data2DBConst.DEFAULT_INT_VALUE))
                micro_step_val = int(row.get('micro_step', Data2DBConst.DEFAULT_INT_VALUE))
            except (ValueError, KeyError, TypeError) as e:
                logger.error(f"Skip invalid row in {file_path}: {e}")
                continue

            # 应用 mapping
            if config.mapping:
                for k, v in config.mapping.items():
                    name = name.replace(k, v)

            # 开启微步计数时按全局微步计数器映射, 否则按迭代步序号映射
            global_step = mapper.map(original_step, micro_step_val)
            micro_step_for_target = Data2DBConst.DEFAULT_INT_VALUE if config.micro_step else micro_step_val

            target_id = target_dict.get((name, vpp_stage, micro_step_for_target))
            if target_id is None:
                continue

            # row data
            row_data = [rank + rank_offset, global_step, target_id, metric_id]
            row_data.extend(process_tensor_value(row[stat]) if stat in row else None for stat in all_stats)
            batch_data.append(tuple(row_data))

            if len(batch_data) >= Data2DBConst.BATCH_SIZE:
                inserted = db.insert_rows(batch_data)
                total_inserted += inserted or 0
                batch_data = []

    if batch_data:
        inserted = db.insert_rows(batch_data)
        total_inserted += inserted or 0

    logger.info(f"Rank {rank} inserted {total_inserted} rows")
    return total_inserted


def import_data(config: CSV2DBConfig) -> bool:
    """Main method to import data into database"""
    monitor_db = MonitorDB(config.db_file)

    tasks, metric_id_dict, target_dict = _pre_scan(config, monitor_db)
    if not tasks:
        logger.error("No valid data files found during pre-scan")
        return False

    total_files = sum(len(files) for _, _, files in tasks)
    logger.info(f"Starting data import for {len(tasks)} batches, {total_files} files...")

    all_succeeded = True
    with ProcessPoolExecutor(max_workers=config.process_num) as executor:
        futures = {
            executor.submit(process_single_rank, task, metric_id_dict, target_dict, config): task[1] for task in tasks
        }

        with tqdm(as_completed(futures), total=len(futures), desc="Import progress") as pbar:
            for future in pbar:
                rank = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to process batch of Rank {rank}: {e}")
                    all_succeeded = False
    if not all_succeeded:
        logger.warning("Some files failed to import data. Please check the log for details.")

    return True
