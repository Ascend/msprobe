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
from typing import Dict, List, Tuple, Any

from msprobe.core.common.const import MonitorConst, Data2DBConst
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
    min_step = None
    max_step = 0
    micro_step_dict = defaultdict(int)  # 记录step下的micro_step数量, micro_step最大值
    metric_stats = defaultdict(set)
    targets = defaultdict(dict)  # 记录targets

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
            # 使用 mapping
            if mapping:
                for key, value in mapping.items():
                    name = name.replace(key, value)

            if use_micro_step:
                if micro_step >= micro_step_dict[step]:
                    # count of micro step
                    micro_step_dict[step] = micro_step + 1
                micro_step = Data2DBConst.DEFAULT_INT_VALUE
            else:
                # 整体的step范围
                min_step = step if min_step is None else min(min_step, step)
                max_step = max(max_step, step)

            # 记录target
            target = (name, vpp_stage, micro_step)
            targets[metric_name][target] = None

    return {
        'rank': rank,
        'min_step': min_step,
        'max_step': max_step,
        'micro_step_dict': dict(micro_step_dict),
        'metric_stats': metric_stats,
        'targets': {m: list(t.keys()) for m, t in targets.items()},
    }


def _build_global_micro_step_mapping(results: List[Dict], use_micro_step: bool) -> Tuple[int, int, Dict[int, int]]:
    """Build global step range and micro_step prefix sum mapping"""
    if not use_micro_step:
        # 如果不开mciro_step 则直接统计step最大最小值
        all_min_steps = [r['min_step'] for r in results if r['min_step'] is not None]
        all_max_steps = [r['max_step'] for r in results]
        min_step = min(all_min_steps) if all_min_steps else 0
        max_step = max(all_max_steps) if all_max_steps else 0
        return min_step, max_step, {}

    # 开启micro-step
    # 合并step中micro_step最大数量
    merged = defaultdict(int)
    for res in results:
        for step, micro_val in res.get('micro_step_dict', {}).items():
            if micro_val > merged[step]:
                merged[step] = micro_val

    if not merged:
        return 0, 0, {}

    # 按step排序 计算prefix sum
    sorted_steps = sorted(merged.items())
    prefix_sum = {}
    cumsum = 0
    for step, micro_val in sorted_steps:
        prefix_sum[step] = cumsum
        cumsum += micro_val

    min_step = 0
    max_step = cumsum - 1  # steps are 0-indexed
    return min_step, max_step, prefix_sum


def _pre_scan(config: CSV2DBConfig, monitor_db: MonitorDB):
    """Pre-scan all targets, metrics, and statistics"""
    logger.info("Scanning dimensions...")
    rank_files = defaultdict(list)

    # 收集每个rank下csv文件
    for rank, dir_path in config.target_output_dirs.items():
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if get_metric_name_from_filename(file):
                rank_files[rank].append(os.path.join(dir_path, file))

    if not rank_files:
        logger.warning("No valid CSV files found")
        return {}, {}, {}, {}

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
    total_files = sum(len(f) for _, f in tasks)
    logger.info(
        f"Split {len(rank_files)} ranks into {total_batches} batches "
        f"(batch_size={batch_size}, total_files={total_files})"
    )

    # Parallel
    with ProcessPoolExecutor(max_workers=config.process_num) as executor:
        futures = {
            executor.submit(_pre_scan_single_rank, rank, files, config.mapping or {}, config.micro_step): rank
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

    if not results:
        return {}, {}, {}, {}

    # Aggregate targets and stats
    targets = defaultdict(OrderedDict)
    all_stats = set()
    metric_stats_agg = defaultdict(set)

    for res in results:
        for metric, t_list in res['targets'].items():
            targets[metric] = update_ordered_dict(targets[metric], t_list)
        for metric, stats in res['metric_stats'].items():
            metric_stats_agg[metric].update(stats)
            all_stats.update(stats)

    # 无可用 CSV 时必须报错退出，不能只导入空库并返回成功
    # _pre_scan_single_rank 对无效行仅 warning 后 continue，可能返回 targets 全空的结果；
    # 此时 metric_stats_agg 可能非空（只读到了列名），必须显式判定是否存在有效数据行
    has_valid_rows = bool(targets)
    if config.micro_step:
        # micro_step 模式下 targets 为空即无有效行（无 min_step 可核对）
        has_valid_rows = has_valid_rows or any(res.get('micro_step_dict') for res in results)
    else:
        has_valid_rows = has_valid_rows or any(res.get('min_step') is not None for res in results)
    if not has_valid_rows:
        logger.error(
            "No valid CSV data found in the data path. "
            "Check that the data path contains monitor CSV files with the expected columns."
        )
        return {}, {}, {}, {}

    # 提取全局最大最小步数
    min_step, max_step, micro_step_prefix = _build_global_micro_step_mapping(results, config.micro_step)

    max_rank = max(int(r['rank']) for r in results)

    # 插入数据库
    global_stats = {"min_step": min_step, "max_step": max_step, "max_rank": max_rank, **metric_stats_agg}
    monitor_db.init_schema()
    monitor_db.insert_dimensions(targets)
    monitor_db.init_global_stats_data(global_stats)
    monitor_db.create_trend_data(get_ordered_list(all_stats, Data2DBConst.OP_TRENDVIS_SUPPORTED))

    metric_id_dict = monitor_db.get_metric_mapping()
    target_dict = monitor_db.get_target_mapping()
    monitor_db.extract_tags_from_processed_targets(targets, metric_id_dict, target_dict)

    return tasks, metric_id_dict, target_dict, micro_step_prefix


def process_single_rank(
    task: Tuple[int, List[str]],
    metric_id_dict: Dict[str, Tuple[int, List[str]]],
    target_dict: Dict[Tuple[str, int, int], int],
    micro_step_prefix: Dict[int, int],
    config: CSV2DBConfig,
) -> int:
    """Process data import for a single rank"""
    rank, files = task
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

            if config.micro_step:
                # 计算 global step index: prefix[step] + micro_step
                base_offset = micro_step_prefix.get(original_step, 0)
                global_step = base_offset + micro_step_val
                micro_step_for_target = Data2DBConst.DEFAULT_INT_VALUE
            else:
                global_step = original_step
                micro_step_for_target = micro_step_val

            target_id = target_dict.get((name, vpp_stage, micro_step_for_target))
            if target_id is None:
                continue

            # row data
            row_data = [rank, global_step, target_id, metric_id]
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

    tasks, metric_id_dict, target_dict, micro_step_prefix = _pre_scan(config, monitor_db)
    if not tasks:
        logger.error("No valid data files found during pre-scan")
        return False

    total_files = sum(len(files) for _, files in tasks)
    logger.info(f"Starting data import for {len(tasks)} batches, {total_files} files...")

    all_succeeded = True
    with ProcessPoolExecutor(max_workers=config.process_num) as executor:
        futures = {
            executor.submit(
                process_single_rank, (rank, files), metric_id_dict, target_dict, micro_step_prefix, config
            ): rank
            for rank, files in tasks
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
