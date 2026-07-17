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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tqdm import tqdm

from msprobe.core.common.const import Const, Data2DBConst
from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    load_json,
    load_construct_json,
)
from msprobe.core.common.data2db_utils import process_tensor_value
from msprobe.core.common.log import logger
from msprobe.core.dump.dump2db.db_utils import DumpDB


def extract_root_nodes(data):
    """
    从给定的数据中提取每个key的最终父节点
    """
    # 存储每个key的父节点关系
    parent_map = {}
    micro_step = 0
    # 遍历数据，建立父子关系
    for key, value in data.items():
        if value is None:
            if f"{Const.SEP}{Const.BACKWARD}" in key:
                parent_map[key] = None
                continue
            if not isinstance(key, str):
                continue
            # 如果父节点是null，则该key自己就是根节点
            root_flag = max(key.startswith(prefix) for prefix in Data2DBConst.ROOT_NODE_PREFIX)
            if root_flag:
                parent_map[key] = micro_step
                micro_step += 1
        else:
            parent_map[key] = value

    # 查找每个key的根节点（递归查找直到找到没有父节点的节点）
    root_nodes = {}

    def find_root(node):
        """递归查找根节点"""
        if isinstance(node, str):
            node = node.replace(f"{Const.SEP}{Const.BACKWARD}", f"{Const.SEP}{Const.FORWARD}").replace(
                f"{Const.SEP}{Const.PARAMS_GRAD}", f"{Const.SEP}{Const.FORWARD}"
            )
        if node not in parent_map:
            return node
        return find_root(parent_map[node])

    # 为每个key找到其根节点
    for key in data.keys():
        parent = find_root(key)
        if parent is not None and isinstance(parent, int):
            root_nodes[key] = parent

    return root_nodes


def reindex_keys_with_mapping(original_dict):
    """
    将同一个micro step的module和api，重新编号，同时返回映射关系

    Args:
        original_dict: 原始字典，格式为 {'Tensor.add.10.forward': 3, ...}
    Returns:
        key_mapping: 映射字典 {'Tensor.add.10.forward': 新k'Tensor.add.0.forward'})
    """
    # 按照原始value分组
    grouped_by_value = defaultdict(list)
    for key, value in original_dict.items():
        grouped_by_value[value].append(key)

    key_mapping = {}  # 存储原始key到新key的映射
    for _, keys in grouped_by_value.items():
        prefix_groups = defaultdict(list)
        for key in keys:
            parts = key.split('.')
            # 检查倒数第一位是否为数字: Module
            if len(parts) >= 1 and parts[-1].isdigit():
                prefix = '.'.join(parts[:-1])  # 除最后一位外的其他部分
                # 31
                number = int(parts[-1])
                prefix_groups[prefix].append((number, key))
            elif len(parts) >= 2 and parts[-2].isdigit():
                prefix = '.'.join(parts[:-2]) + '.' + parts[-1]  # 除倒数第二位外的其他部分
                number = int(parts[-2])
                prefix_groups[prefix].append((number, key))
            else:
                # 如果最后两位都不是数字，直接使用整个key作为前缀
                prefix_groups[key].append((0, key))

        # 对每组前缀进行重新编号
        for prefix, items in prefix_groups.items():
            # 按原数字排序
            sorted_items = sorted(items, key=lambda x: x[0])
            for new_index, (_, original_key) in enumerate(sorted_items):
                # 重新构造key：前缀 + 新索引
                original_parts = original_key.split('.')
                if len(original_parts) >= 1 and original_parts[-1].isdigit():
                    new_key = f"{prefix}.{new_index}"
                elif len(original_parts) >= 2 and original_parts[-2].isdigit():
                    parts = original_parts.copy()
                    parts[-2] = str(new_index)
                    new_key = '.'.join(parts)
                else:
                    new_key = original_key
                # 记录映射关系
                key_mapping[original_key] = new_key
    return key_mapping


def scan_files(data_dir):
    logger.info("Scanning data directory...")
    # 扫描所有step和rank
    valid_ranks = defaultdict(list)

    # 首先扫描目录结构获取有效rank
    for step_dir in os.listdir(data_dir):
        if not step_dir.startswith('step'):
            continue
        try:
            step = int(step_dir[4:])  # 提取step数字
        except ValueError:
            continue
        step_path = os.path.join(data_dir, step_dir)

        # 分别收集当前step的rank和proc信息
        normal_ranks = []
        proc_items = []

        for item_dir in os.listdir(step_path):
            if item_dir.startswith('rank') or item_dir.startswith('proc'):
                try:
                    original_id = int(item_dir[4:])  # 提取rank或proc数字
                except ValueError:
                    continue
                item_path = os.path.join(step_path, item_dir)
                json_path = os.path.join(item_path, "dump.json")
                if not os.path.exists(json_path):
                    continue
                construct_path = os.path.join(item_path, "construct.json")
                if item_dir.startswith('rank'):
                    normal_ranks.append((original_id, json_path, construct_path))
                else:  # proc directory
                    proc_items.append((original_id, json_path, construct_path))

        # 处理并分配连续的rank编号
        if normal_ranks or proc_items:
            proc_items.sort(key=lambda x: x[0])
            normal_ranks.sort(key=lambda x: x[0])

            # 计算下一个可用的rank编号
            if normal_ranks:
                max_rank = normal_ranks[-1][0]
                start_rank = max_rank + 1
            else:
                start_rank = 0

            # 组合结果：先放正常rank，再放映射后的proc
            final_ranks = normal_ranks[:]
            # 记录proc到rank的映射关系
            proc_to_rank_mapping = {}
            for i, (proc_num, json_path, construct_path) in enumerate(proc_items):
                mapped_rank = start_rank + i
                proc_to_rank_mapping[f"proc{proc_num}"] = mapped_rank
                final_ranks.append((mapped_rank, json_path, construct_path))
            # 如果有proc映射，记录日志
            if proc_to_rank_mapping:
                logger.info(f"Step {step}: proc to rank mapping - {proc_to_rank_mapping}")
            # 按rank排序后存储
            valid_ranks[step] = final_ranks

    return valid_ranks


@dataclass
class TensorProcessingParams:
    """Tensor处理函数的参数载体"""

    tensor_data: Dict[str, Any]
    target_prefix: str
    vpp_stage: int
    micro_step: int
    step: int
    rank: int
    metric_id: int
    metric_type: Optional[str]


class DumpRecordBuilder:
    def __init__(self, db: DumpDB, data_dir, mapping, micro_step=True):
        self.db = db
        self.data_dir = data_dir
        self.mapping = mapping
        self.micro_step = micro_step

    @staticmethod
    def extract_target_info(full_key):
        """
        从完整的key中提取target信息
        格式: Module.layerN.operation.M
        """
        vpp_stage = 0  # 暂不提取vpp信息
        micro_step = 0
        target_prefix = full_key

        parts = full_key.split(Const.SEP)
        if parts and parts[-1].isdigit():
            micro_step = int(parts[-1])
            # 重新构建target_prefix（去掉最后一个数字部分）
            target_prefix = Const.SEP.join(parts[:-1])

        return target_prefix, vpp_stage, micro_step

    @staticmethod
    def parse_tensor_target(metric_type, tensor_type, tensor_idx):
        """根据tensor类型和索引生成target后缀"""
        if metric_type in [Data2DBConst.FORWARD, Data2DBConst.RECOMPUTE]:
            if tensor_type == Const.INPUT_ARGS:
                return f".input.{tensor_idx}"
            elif tensor_type == Const.OUTPUT:
                return f".output.{tensor_idx}"
            elif tensor_type == Const.PARAMS:
                return f".parameters.{tensor_idx}"
        elif metric_type == Data2DBConst.BACKWARD:
            if tensor_type == Const.INPUT:
                return f".input.{tensor_idx}"
            elif tensor_type == Const.OUTPUT:
                return f".output.{tensor_idx}"
        return ""

    def import_data(self, valid_ranks):
        """导入数据"""
        current_start_step = 0
        max_rank = 0
        for step in tqdm(sorted(valid_ranks.keys()), desc="Processing steps"):
            step_dir = f"step{step}"
            step_path = os.path.join(self.data_dir, step_dir)

            check_file_or_directory_path(step_path, isdir=True)

            # 为当前step的所有有效rank创建进度条
            if not valid_ranks[step]:
                continue
            # 设置起始步,如果用step则为当前step，否则为micro_step累计
            if not self.micro_step:
                current_start_step = step
            total_micro_step = []
            for rank, json_path, construct_path in tqdm(valid_ranks[step], desc=f"Step {step} ranks", leave=False):
                max_rank = max(max_rank, rank)
                total_micro_step.append(self._process_dump_file(json_path, construct_path, current_start_step, rank))
            if self.micro_step:
                logger.info(f"Step {step} processing completed. Total micro steps identified: {total_micro_step}")
                total_micro_step = max(total_micro_step)
                current_start_step += total_micro_step

        # 更新全局统计信息
        global_stats = {
            "max_rank": max_rank,
            "min_step": 0 if self.micro_step else min(valid_ranks.keys()),
            # micro_step下实际步数为下一个micro_step起始步数-1
            "max_step": current_start_step - 1 if self.micro_step else current_start_step,
        }
        for metric_name in Data2DBConst.METRICS:
            global_stats[metric_name] = Data2DBConst.ORDERED_STAT
        self.db.init_global_stats_data(global_stats)
        self.db.extract_tags_from_processed_targets()

    def _determine_metric_type(self, full_key, tensor_data):
        for key, value in self.mapping.items():
            full_key = full_key.replace(key, value)

        # 确定metric类型
        metric_type = None
        if f"{Const.SEP}{Const.FORWARD}{Const.SEP}" in full_key or full_key.endswith(f"{Const.SEP}{Const.FORWARD}"):
            if tensor_data.get('is_recompute', False):
                metric_type = Data2DBConst.RECOMPUTE
            else:
                metric_type = Data2DBConst.FORWARD
            full_key = full_key.replace(f"{Const.SEP}{Const.FORWARD}{Const.SEP}", Const.SEP)  # module
            full_key = full_key.replace(f"{Const.SEP}{Const.FORWARD}", "")  # api
        elif f"{Const.SEP}{Const.BACKWARD}{Const.SEP}" in full_key or full_key.endswith(f"{Const.SEP}{Const.BACKWARD}"):
            metric_type = Data2DBConst.BACKWARD
            full_key = full_key.replace(f"{Const.SEP}{Const.BACKWARD}{Const.SEP}", Const.SEP)
            full_key = full_key.replace(f"{Const.SEP}{Const.BACKWARD}", "")
        elif full_key.endswith(Const.PARAMS_GRAD):
            metric_type = Data2DBConst.PARAMETERS_GRAD
            full_key = full_key.replace(f"{Const.SEP}{Const.PARAMS_GRAD}", "")
        # fsdp
        elif len(full_key.split(Const.SEP)) >= 3:
            parts = full_key.split(Const.SEP)
            if parts[-2] == Const.PARAMS_GRAD:
                metric_type = Data2DBConst.PARAMETERS_GRAD
                full_key = Const.SEP.join(parts[:-2])

        return metric_type, full_key

    def _add_tensor_data(self, tensor, target_name, tensor_params: TensorProcessingParams, batch_data):
        """添加tensor数据方法"""

        cache_key = (target_name, tensor_params.vpp_stage, tensor_params.micro_step)
        # 如果缓存中不存在，创建临时ID
        cache_id_dict = self.db.cache_targets(cache_key, tensor_params.metric_id)

        # 准备数据行, 这里id还是个临时id, 需要更新后读取, 第三个实际为{"id": 0}
        row_data = [tensor_params.rank, tensor_params.step, cache_id_dict, tensor_params.metric_id]
        for stat in Data2DBConst.ORDERED_STAT:
            value = tensor.get(stat.capitalize(), None)
            row_data.append(process_tensor_value(value))
        batch_data.append((row_data))

    def _process_forward_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理forward/recompute数据"""
        # 处理input_args
        if Const.INPUT_ARGS in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.INPUT_ARGS]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(Data2DBConst.FORWARD, Const.INPUT_ARGS, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

        # 处理output
        if Const.OUTPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.OUTPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(Data2DBConst.FORWARD, Const.OUTPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

        # 处理parameters
        if Const.PARAMS in tensor_params.tensor_data:
            for param_name, param_tensors in tensor_params.tensor_data[Const.PARAMS].items():
                if isinstance(param_tensors, dict):
                    tensor = param_tensors
                elif isinstance(param_tensors, list) and param_tensors and isinstance(param_tensors[0], dict):
                    tensor = param_tensors[0]
                else:
                    continue
                if tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(
                        Data2DBConst.FORWARD, Const.PARAMS, param_name
                    )
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

    def _process_parameters_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理parameters数据"""
        for param_name, param_tensors in tensor_params.tensor_data.items():
            if isinstance(param_tensors, dict):
                tensor = param_tensors
            elif isinstance(param_tensors, list) and param_tensors and isinstance(param_tensors[0], dict):
                tensor = param_tensors[0]
            else:
                continue
            if tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                    continue
                target_name = tensor_params.target_prefix + f".{param_name}"
                self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

    def _process_backward_data(self, tensor_params: TensorProcessingParams, batch_data):
        """处理backward数据"""
        # 处理input
        if Const.INPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.INPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(Data2DBConst.BACKWARD, Const.INPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

        # 处理output
        if Const.OUTPUT in tensor_params.tensor_data:
            for idx, tensor in enumerate(tensor_params.tensor_data[Const.OUTPUT]):
                if isinstance(tensor, dict) and tensor.get(Const.TYPE) in Data2DBConst.SUPPORT_TYPE:
                    if tensor.get(Const.DTYPE) not in Data2DBConst.SUPPORT_DTYPE:
                        continue
                    target_suffix = DumpRecordBuilder.parse_tensor_target(Data2DBConst.BACKWARD, Const.OUTPUT, idx)
                    target_name = tensor_params.target_prefix + target_suffix
                    self._add_tensor_data(tensor, target_name, tensor_params, batch_data)

    def _process_dump_file(self, json_path, construct_path, start_step, rank):
        """处理单个dump.json文件"""
        data = load_json(json_path)
        micro_step_dict = {}
        total_micro_step = 1
        index_mapping = {}
        if self.micro_step:
            construct_dict, _ = load_construct_json(construct_path)
            micro_step_dict = extract_root_nodes(construct_dict)
            # 当前从0记起，实际micro+1
            total_micro_step = max(micro_step_dict.values() or [0]) + 1
            index_mapping = reindex_keys_with_mapping(micro_step_dict)
        if 'data' not in data or not isinstance(data['data'], dict):
            return total_micro_step

        batch_data = []
        for _, (ori_key, tensor_data) in enumerate(data['data'].items()):
            if not isinstance(ori_key, str) or not isinstance(tensor_data, dict):
                continue
            # 对于micro_step内, 对index重新计数
            full_key = index_mapping.get(ori_key, None) if index_mapping else ori_key
            if not full_key:
                continue
            mstep = micro_step_dict.get(ori_key, 0)

            metric_type, full_key = self._determine_metric_type(full_key, tensor_data)

            if not metric_type:
                continue

            # 使用缓存获取metric_id和stats
            metric_id = self.db.get_metric_id(metric_type)
            target_prefix, vpp_stage, target_index = DumpRecordBuilder.extract_target_info(full_key)

            tensor_params = TensorProcessingParams(
                tensor_data=tensor_data,
                target_prefix=target_prefix,
                vpp_stage=vpp_stage,
                micro_step=target_index,
                step=start_step + mstep,
                rank=rank,
                metric_id=metric_id,
                metric_type=metric_type,
            )

            # 处理不同类型的tensor数据
            if metric_type in [Data2DBConst.FORWARD, Data2DBConst.RECOMPUTE]:
                self._process_forward_data(tensor_params, batch_data)
            elif metric_type == Data2DBConst.PARAMETERS_GRAD:
                self._process_parameters_data(tensor_params, batch_data)
            elif metric_type == Data2DBConst.BACKWARD:
                self._process_backward_data(tensor_params, batch_data)

            if len(batch_data) % Data2DBConst.BATCH_SIZE == 0 and len(batch_data) > 0:
                self.db.batch_insert_targets()
                self.db.batch_insert_data(batch_data)
                batch_data = []

        self.db.batch_insert_targets()
        self.db.batch_insert_data(batch_data)
        return total_micro_step
