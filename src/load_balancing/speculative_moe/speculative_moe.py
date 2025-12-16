# coding=utf-8
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

import math
import random
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from multiprocessing import Pool
from typing import List

import numpy as np
from ortools.sat.python import cp_model

logger = logging.getLogger("msit_logger")


def parse_ep_file(ep_file_path, ep_file=None, n_share_expert_devices=0):
    experts_table = {}

    # should not be here
    if ep_file is None:
        raise ValueError("Eplb file should be loaded before running algorithm.")

    layer_count = ep_file["moe_layer_count"]
    layer_list = ep_file["layer_list"]

    for layer_list_idx in range(layer_count):
        layer_info = layer_list[layer_list_idx]
        layer_id = layer_info["layer_id"]
        device_count = layer_info["device_count"]
        device_list = layer_info["device_list"]
        layer_expert_table = []

        for device_list_idx in range(device_count):
            device_info = device_list[device_list_idx]
            device_expert = device_info["device_expert"]
            layer_expert_table.append(device_expert)

        experts_table[layer_id] = layer_expert_table[n_share_expert_devices:]

    return experts_table


def all_to_all_algorithm_multi_process(d2e_table, 
                                       sample_top_k_matrix, 
                                       config, 
                                       shared_status, 
                                       selected_layers,
                                       cpu_per_process: int = 94):
    d2e_table = parse_ep_file(config.deploy_fp, d2e_table, config.n_share_expert_devices)
    n_processes = (mp.cpu_count() - 4) // cpu_per_process

    worker_args = []
    for layer_idx in range(selected_layers[0], selected_layers[1] + 1):
        shared_status[layer_idx] = ("Launching All2AllBalance", "")
        worker_args.append((d2e_table, sample_top_k_matrix, config, [layer_idx], cpu_per_process, shared_status))

    task_remain = len(worker_args)
    d2e_tables = {}
    start_idx = 0
    while task_remain > 0:
        task_num = min(n_processes, task_remain)
        with mp.Pool(task_num) as pool:
            cur_results = pool.starmap(all_to_all_algorithm, worker_args[start_idx: start_idx + task_num])
        task_remain -= task_num
        start_idx += task_num

        for d2e_part in cur_results:
            d2e_tables.update(d2e_part)

    deploy_table = dump_tables(d2e_tables, 
                               selected_layers=config.selected_layers, 
                               n_share_expert_devices=config.n_share_expert_devices)

    for layer_idx in range(selected_layers[0], selected_layers[1] + 1):
        shared_status[layer_idx] = ("All2AllBalance End", "")

    return deploy_table


def all_to_all_algorithm(d2e_table,
                         sample_top_k_matrix,
                         config,
                         layer_indexes: List[int],
                         cpu_per_process: int,
                         shared_status):
    """单进程speculative_moe_algo, 负责对特定layer_indexes进行专家布放求解"""
    d2e_table_full = {}
    for layer_idx in layer_indexes:
        deploy_algo = All2AllBalance(d2e_table[layer_idx], sample_top_k_matrix[layer_idx], config, layer_idx,
                                     shared_status)
        d2e_table_full[layer_idx] = deploy_algo.handle_affinity()
    return d2e_table_full


def generate_expert_map_json(tables, n_share_expert_devices=0, selected_layers=None):
    if selected_layers is None:
        selected_layers = [0, len(tables) - 1]
    expert_data = {}
    expert_data["moe_layer_count"] = len(tables)
    layer_list = []
    for i, layer_id in enumerate(range(selected_layers[0], selected_layers[1] + 1)):
        layer = {"layer_id": layer_id, "device_count": len(tables[i]) + n_share_expert_devices}
        device_list = []
        for j in range(n_share_expert_devices):
            device = {}
            device["device_id"] = j
            device["device_expert"] = [0]
            device_list.append(device)
        for j in range(len(tables[i])):
            device = {}
            device["device_id"] = j + n_share_expert_devices
            device["device_expert"] = list(tables[i][j])
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)

    expert_data["layer_list"] = layer_list

    return expert_data


def dump_tables(d2e_tables, selected_layers=None, n_share_expert_devices=0):
    layer_list = []
    keys = list(d2e_tables.keys())
    keys.sort()
    for layer_idx in keys:
        device_list = [d for d in d2e_tables[layer_idx]]
        layer_list.append(device_list)

    deploy_table = generate_expert_map_json(
        layer_list,
        selected_layers=selected_layers, n_share_expert_devices=n_share_expert_devices)

    return deploy_table


def generate_n_stage_hotness(expert_hotness, n_stage=-1, mean_=96):
    n_layer, n_expert, n_interval = expert_hotness.shape
    if n_stage == -1:
        n_stage = min(512, n_interval)
    step = n_interval // n_stage
    remainder = n_interval % n_stage  # 计算余数
    dfs = []
    start_idx = 0
    for i in range(n_stage):
        # 如果还有剩余数据，则给当前阶段分配一个额外的数据点
        extra = 1 if i < remainder else 0
        end_idx = start_idx + step + extra
        dfs.append(expert_hotness[:, :, start_idx:end_idx].sum(-1))
        start_idx = end_idx  # 更新下一个阶段的起始索引

    stage_weights = [np.sum(stage) for stage in dfs]
    total_weight = sum(stage_weights)
    target_total = mean_ * n_stage
    scale_factors = [max(1, int(w / total_weight * target_total)) for w in stage_weights]
    return np.array(dfs), np.array(scale_factors)


class All2AllBalance(object):
    def __init__(self,
                 d2e_table,
                 sample_top_k_matrix,
                 config,
                 layer_idx: int,
                 shared_status,
                 temperature: int = 100000000,
                 cooling_rate: float = 0.99,
                 max_iterations: int = 5000):
        self.d2e_table = d2e_table
        self.sample_top_k_matrix = sample_top_k_matrix
        self.layer_idx = layer_idx
        self.n_nodes = config.n_nodes
        self.n_devices = config.n_devices
        self.n_experts = config.n_experts
        self.n_red_experts = config.redundant_experts
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.n_devices_per_node = None
        self.shared_status = shared_status

    @staticmethod
    def calculate_benefit(traffic_matrix_all):
        return np.max(traffic_matrix_all)

    def handle_affinity(self):
        self.shared_status[self.layer_idx] = ("All2AllBalance Start", "")
        # 定义搜索初始位置，即每个device按顺序依次放入machine
        device_to_machine = [[] for m in range(self.n_nodes)]
        self.n_devices_per_node = int(self.n_devices / self.n_nodes)
        for m in range(self.n_nodes):
            for k in range(self.n_devices_per_node * m, self.n_devices_per_node * (m + 1)):
                device_to_machine[m].append(k)
        current_assignments = [row[:] for row in device_to_machine]
        # 计算初始分配下的目标函数值
        current_cost = self.evaluate_benefit(self.d2e_table, self.sample_top_k_matrix)
        best_d2e_table = self.d2e_table
        best_cost = current_cost
        # 开始进行优化搜索，最多搜索max_iterations次
        for it in range(self.max_iterations):
            # 基于现有分配生成新的分配
            neighbor_assignments = [row[:] for row in current_assignments]
            # 随机选取两个需要进行交换的device，并获取他们目前所在的machine，如果两个device已经在同一个machine上，这进行下一次搜索
            [k1, k2] = random.sample(list(range(self.n_devices)), 2)
            m1, m2 = -1, -1
            for m in range(self.n_nodes):
                if k1 in current_assignments[m]:
                    m1 = m
                if k2 in current_assignments[m]:
                    m2 = m
            if m1 == m2:
                continue
            else:
                neighbor_assignments[m1].remove(k1)
                neighbor_assignments[m2].remove(k2)
                neighbor_assignments[m1].append(k2)
                neighbor_assignments[m2].append(k1)
            # 根据新的device_2_machine对应表，生成新解的d2e_table，并计算新解的目标函数值
            d2e_table_new = []
            for m in range(self.n_nodes):
                for k in neighbor_assignments[m]:
                    d2e_table_new.append(self.d2e_table[k])
            neighbor_cost = self.evaluate_benefit(d2e_table_new, self.sample_top_k_matrix)
            # 计算现有解和新解在目标函数值上的变化
            delta_cost = neighbor_cost - current_cost
            # 记录搜索过程中的最优解
            if neighbor_cost < best_cost:
                best_d2e_table = [row[:] for row in d2e_table_new]
                best_cost = neighbor_cost
            # 根据模拟退火的思想，判断是否采纳新解
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / self.temperature):
                current_assignments = [row[:] for row in neighbor_assignments]
                current_cost = neighbor_cost
            # 更新退火温度
            self.temperature *= self.cooling_rate

            self.shared_status[self.layer_idx] = (
                "All2AllBalance Start", 
                f'Iteration {it}: Temp={self.temperature:.2f}, \
                Current Score={current_cost:.2f}, Best Score={best_cost:.2f}'
            )

        # 更新d2e_table
        return best_d2e_table

    def evaluate_benefit(self, expert_map, data_trace):
        # 根据expert_map，获取逻辑专家和物理专家的映射关系
        map_logical_physical = self.generate_map_logical_physical(expert_map)
        # 根据data_hot数据，获取流量矩阵
        traffic_matrix_all = self.generate_traffic_matrix(map_logical_physical, data_trace)
        # 根据流量矩阵和权重配比，获取最终的取值
        benefit = self.calculate_benefit(traffic_matrix_all)
        return benefit

    def generate_map_logical_physical(self, expert_map):
        # 构建映射关系
        map_logical_physical = {exp: [] for exp in range(self.n_experts)}
        # 遍历所有rank的所有逻辑专家
        expert_num_per_rank = int((self.n_experts + self.n_red_experts) / self.n_devices)
        for dev_id, dev_exp in enumerate(expert_map):
            for index, exp_logical in enumerate(dev_exp): 
                exp_physical = dev_id * expert_num_per_rank + index  # 逻辑专家所在的物理位置
                map_logical_physical[exp_logical].append(exp_physical)
        return map_logical_physical

    def generate_traffic_matrix(self, map_logical_physical, data_trace):
        # 统计去往不同server的token总数
        traffic = np.zeros((self.n_nodes, self.n_devices_per_node), dtype=int)
        shape = data_trace.shape
        # 遍历文件夹中的所有文件
        for rank_id in range(shape[0]):
            for iteration in range(shape[2]):
                local_rank = rank_id % self.n_devices_per_node
                experts_top_k = data_trace[rank_id][iteration]
                # 计算每组需要选择的专家数
                physical_experts = []
                for exp in experts_top_k:
                    physical = map_logical_physical[exp]
                    num_chosen = len(physical)
                    chosen = rank_id % num_chosen
                    physical_experts.append(physical[chosen])
                is_server_activate = self.find_server_activated(physical_experts)
                for i in range(self.n_nodes):
                    traffic[i][local_rank] += is_server_activate[i]
        return traffic

    def find_server_activated(self, physical_experts):
        is_server_activate = [0 for _ in range(self.n_nodes)]
        for physical_expert_id in physical_experts:
            expert_n_per_node = int((self.n_experts + self.n_red_experts) // self.n_nodes)
            index = int(physical_expert_id // expert_n_per_node)
            is_server_activate[index] = 1
        return is_server_activate


class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, layer_idx, sum_weights_lengths, shared_status):
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_status = shared_status
        self.sum_weights_lengths = sum_weights_lengths

    def on_solution_callback(self):
        obj = self.ObjectiveValue()  # best solution value
        bound = self.BestObjectiveBound()  # best bound
        self.shared_status[self.layer_idx] = \
            ("ExpILPSolver Solving",
             f"Layer {self.layer_idx} | "
             f"Best obj: {round((obj / self.sum_weights_lengths), 3)} | "
             f"Bound: {round((bound / self.sum_weights_lengths), 3)}")


class ExpSolver(object):
    def __init__(self,
                 expert_weights_all,
                 config,
                 shared_status,
                 ):
        self.config = config

        self.n_experts = config.n_experts
        self.n_red_experts = config.redundant_experts
        self.total_layers = config.n_layers
        self.selected_layers = config.selected_layers
        self.layer_idxes = list(range(self.selected_layers[0], self.selected_layers[1] + 1))
        self.n_devices = config.n_devices
        self.n_nodes = config.n_nodes

        self.enhanced = config.enhanced
        self.mode = "enhanced" if self.enhanced else "basic"

        self.expert_weights_all = expert_weights_all[self.layer_idxes]
        self.expert_weights = None
        self.expert_weights_intervals = None

        for layer_idx in self.layer_idxes:
            shared_status[layer_idx] = ("Init ExpSolver", "")
        self.shared_status = shared_status
        self.load_parse_data()

    def load_parse_data(self):
        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("Data PreProcessing", "")

        expert_weights_intervals_basic, expert_weights = generate_n_stage_hotness(
            self.expert_weights_all, n_stage=-1 if self.enhanced else self.config.num_stages
        )
        self.expert_weights_intervals = expert_weights_intervals_basic
        self.expert_weights = expert_weights

    def fit(self, cpu_per_process=1):
        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("ExpSolver Fitting", "")

        n_processes = (mp.cpu_count() - 4) // cpu_per_process
        worker_args = [(layer_id,) for layer_id in self.layer_idxes]
        task_remain = len(worker_args)
        d2e_tables = {}
        objs = {}
        n_duplicates_basic = {}
        n_duplicates_enhanced = {}
        start_idx = 0
        while task_remain > 0:
            task_num = min(n_processes, task_remain)
            with mp.Pool(task_num) as pool:
                cur_results = pool.starmap(self.fit_, worker_args[start_idx: start_idx + task_num])

            for i, (d2e_part, obj_part, duplicates_basic_part, duplicates_enhanced_part) in enumerate(cur_results):
                d2e_tables[worker_args[start_idx + i][0]] = d2e_part
                objs[worker_args[start_idx + i][0]] = obj_part
                n_duplicates_basic[worker_args[start_idx + i][0]] = duplicates_basic_part
                n_duplicates_enhanced[worker_args[start_idx + i][0]] = duplicates_enhanced_part

            task_remain -= task_num
            start_idx += task_num

        deploy_table = dump_tables(d2e_tables,
                    selected_layers=self.selected_layers,
                    n_share_expert_devices=self.config.n_share_expert_devices)
        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("ExpSolver End", 
                                             f"Layer:{layer_idx} obj is :{objs.get(layer_idx, 'Not exist')}")

        return d2e_tables, objs, n_duplicates_basic, n_duplicates_enhanced, deploy_table

    def fit_(self, layer_idx):
        n_duplicates_basic, n_duplicates_enhanced = self.cal_n_exp_duplicates(
                                                            self.expert_weights_intervals[:, layer_idx], 
                                                            self.mode)
        d2e_table, obj = self.cal_d2e_table(self.expert_weights_intervals[:, layer_idx], 
                                            self.expert_weights, 
                                            n_duplicates_basic)

        return d2e_table, obj, n_duplicates_basic, n_duplicates_enhanced

    def cal_n_exp_duplicates(self, weights_intervals: np.ndarray, mode: str = 'basic'):
        ratio = 1
        n_duplicates_basic = np.ones((self.n_experts,), dtype=int)
        n_duplicates_enhanced = np.ones((self.n_experts,), dtype=int)
        if mode == 'enhanced':
            ratio = 1.2
        for i in range(int(self.n_red_experts * ratio)):
            if mode == 'enhanced':
                divided_weights_intervals = weights_intervals / n_duplicates_enhanced
            else:
                divided_weights_intervals = weights_intervals / n_duplicates_basic
            benefit = defaultdict(lambda: - 0.1)
            if mode == 'enhanced':
                mask = n_duplicates_enhanced < self.n_devices
            else:
                mask = n_duplicates_basic < self.n_devices
            if not np.where(mask)[0].size:
                raise ValueError('n_red_experts is too large, please check the input')
            for j, _ in enumerate(weights_intervals):
                max_index = np.where(mask)[0][0]
                second_max_index = -1
                for index in range(max_index + 1, len(divided_weights_intervals[j])):
                    if mask[index]:
                        if divided_weights_intervals[j][index] > divided_weights_intervals[j][max_index]:
                            second_max_index = max_index
                            max_index = index
                        elif divided_weights_intervals[j][index] < divided_weights_intervals[j][max_index]:
                            if (second_max_index == -1 or
                                    divided_weights_intervals[j][index] > divided_weights_intervals[j][
                                        second_max_index]):
                                second_max_index = index
                        else:
                            second_max_index = index
                if mode == 'enhanced':
                    tmp_value = weights_intervals[j][max_index] / (n_duplicates_enhanced[max_index] + 1)
                else:
                    tmp_value = weights_intervals[j][max_index] / (n_duplicates_basic[max_index] + 1)
                benefit[max_index] += (divided_weights_intervals[j][max_index] -
                                       max(tmp_value, divided_weights_intervals[j][second_max_index]))
            replicate_exp = max(benefit, key=benefit.get)
            if mode == 'enhanced':
                n_duplicates_enhanced[replicate_exp] += 1
                if i == self.n_red_experts - 1:
                    n_duplicates_basic = n_duplicates_enhanced.copy()
            else:
                n_duplicates_basic[replicate_exp] += 1
        if mode == 'enhanced':
            return n_duplicates_basic, n_duplicates_enhanced
        else:
            return n_duplicates_basic, n_duplicates_basic

    def cal_d2e_table(self, weights_intervals: np.ndarray, weights_lengths: np.ndarray,
                      n_duplicates: np.ndarray):
        expert_weights = np.sum(weights_intervals, axis=0)
        d2e_table = np.full(
            (self.n_devices, (self.n_experts + self.n_red_experts) // self.n_devices), -1, dtype=int)
        dev_weights = np.zeros((len(weights_intervals), self.n_devices), dtype=float)
        dev_phy_exp_n = np.zeros((self.n_devices,), dtype=int)
        sort_indices = sorted(range(self.n_experts), key=lambda i: expert_weights[i] / n_duplicates[i], reverse=True)
        tmp_idx = 0
        for log_exp_id in sort_indices:
            for _ in range(n_duplicates[log_exp_id]):
                max_avg_ratio, max_min_ratio = math.inf, math.inf
                opt_dev_id = -1
                for dev_id in range(self.n_devices):
                    if log_exp_id not in d2e_table[dev_id]:
                        if dev_phy_exp_n[dev_id] < (self.n_experts + self.n_red_experts) // self.n_devices:
                            tmp_dev_weights = dev_weights.copy()
                            tmp_dev_weights[:, dev_id] += weights_intervals[:, log_exp_id] / n_duplicates[log_exp_id]
                            tmp_max_avg_ratio = np.sum(
                                (np.max(tmp_dev_weights, axis=1) / (np.mean(tmp_dev_weights, axis=1) + 1e-10)
                                 * weights_lengths))
                            tmp_max_min_ratio = np.sum(
                                (np.max(tmp_dev_weights, axis=1) / (np.min(tmp_dev_weights, axis=1) + 1e-10)
                                 * weights_lengths))
                            if (tmp_max_avg_ratio, tmp_max_min_ratio) < (max_avg_ratio, max_min_ratio):
                                opt_dev_id = dev_id
                                max_avg_ratio = tmp_max_avg_ratio
                                max_min_ratio = tmp_max_min_ratio
                dev_weights[:, opt_dev_id] += weights_intervals[:, log_exp_id] / n_duplicates[log_exp_id]
                d2e_table[opt_dev_id][dev_phy_exp_n[opt_dev_id]] = log_exp_id
                dev_phy_exp_n[opt_dev_id] += 1
                tmp_idx += 1

        obj = [np.mean(np.max(dev_weights, axis=1) / np.mean(dev_weights, axis=1)),
               np.mean(np.max(dev_weights, axis=1) / np.mean(dev_weights, axis=1))]
        if np.any(d2e_table == -1):
            raise ValueError('Can not compute the deployment table successfully, n_red_experts should be smaller')
        return d2e_table, obj


class ExpILPSolver(object):
    def __init__(self,
                 expert_weights_all,
                 config,
                 shared_status,
                 n_duplicates_basic,
                 n_duplicates,
                 d2e_table_basic
                 ):
        self.config = config
        self.selected_layers = config.selected_layers
        self.layer_idxes = list(range(self.selected_layers[0], self.selected_layers[1] + 1))
        self.n_nodes = config.n_nodes
        self.n_devices = config.n_devices
        self.n_experts = config.n_experts
        self.n_red_experts = config.redundant_experts
        self.max_time_in_seconds = config.max_time_in_seconds
        self.cpu_per_process = config.cpu_per_process
        self.n_duplicates_basic = n_duplicates_basic
        self.n_duplicates = n_duplicates
        self.d2e_table_basic = d2e_table_basic

        self.obj = np.zeros((2,), dtype=int)
        self.shared_status = shared_status

        self.expert_weights_all = expert_weights_all
        self.weights_intervals = None  # 8 x 256
        self.weight_lengths = None

        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("Init ExpILPSolver", "")

        self.load_parse_data()

    def load_parse_data(self):
        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("Data PreProcessing", "")

        expert_weights_intervals_basic, expert_weights = generate_n_stage_hotness(
            self.expert_weights_all, n_stage=self.config.num_stages
        )
        self.weights_intervals = expert_weights_intervals_basic
        self.weight_lengths = expert_weights

    def fit(self, cpu_per_process=11):
        for layer_idx in self.layer_idxes:
            self.shared_status[layer_idx] = ("ExpILPSolver Fitting", "")

        n_processes = (mp.cpu_count() - 4) // cpu_per_process
        worker_args = [(layer_id,) for layer_id in self.layer_idxes]
        task_remain = len(worker_args)
        d2e_tables = {}
        objs = {}
        start_idx = 0
        while task_remain > 0:
            task_num = min(n_processes, task_remain)
            with mp.Pool(task_num) as pool:
                cur_results = pool.starmap(self.fit_, worker_args[start_idx: start_idx + task_num])

            for i, (d2e_part, obj_part) in enumerate(cur_results):
                d2e_tables[worker_args[start_idx + i][0]] = d2e_part
                objs[worker_args[start_idx + i][0]] = obj_part

            task_remain -= task_num
            start_idx += task_num

        deploy_table = dump_tables(d2e_tables,
                    selected_layers=self.selected_layers,
                    n_share_expert_devices=self.config.n_share_expert_devices)

        for layer_idx in self.layer_idxes:
            obj_value = objs.get(layer_idx, "Not exist")  # 使用dict.get()方法，设置默认值为"Not exist"
            self.shared_status[layer_idx] = ("ExpILPSolver End", f"Layer:{layer_idx} obj is :{obj_value}")

        return d2e_tables, objs, deploy_table

    def fit_(self, layer_idx):
        d2e_table, obj = self.solve(self.weights_intervals[:, layer_idx], self.weight_lengths,
                                    self.n_duplicates[layer_idx], self.n_duplicates_basic[layer_idx],
                                    self.d2e_table_basic[layer_idx], layer_idx)

        return d2e_table, obj

    def solve(self, weights_intervals, weight_lengths, n_duplicates, n_duplicates_basic, d2e_table_basic, layer_idx):
        n_it_intvs = len(weights_intervals)
        low_bound, up_bound = 1, 5000
        coefficient = 100  # slack_factor的整数化系数
        model = cp_model.CpModel()
        up = n_duplicates
        scaling_factor = 1
        for i in range(1, 10 + 1):
            scaling_factor = abs(scaling_factor * i) // math.gcd(scaling_factor, i)
        n_instances = self.n_experts + self.n_red_experts

        # 定义专家布放bool变量C，C[e][k] = 1表示专家e有副本布放在设备k上
        c = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for k in range(self.n_devices):
                var = model.NewBoolVar(f'C_{e}_{k}')
                c[e].append(var)
                if e in d2e_table_basic[k]:
                    model.AddHint(c[e][k], 1)
                else:
                    model.AddHint(c[e][k], 0)
        # 定义slack_factor变量，衡量单卡专家热度最大值和平均值的比例，表征通信/计算的拖尾严重程度
        sf = []
        
        for m in range(n_it_intvs):
            var = model.NewIntVar(low_bound, up_bound, f'SF_{m}')
            sf.append(var)
        # 定义 N 和 L bool变量，其中N[e][n] = 1 表示专家e共布放n+1个副本，
        # L[e][n][k] = 1表示专家e共布放n+1个副本，且其中有一个副本布放在设备k上
        num = [[] for _ in range(self.n_experts)]
        lum = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for n in range(up[e]):
                var = model.NewBoolVar(f'N_{e}_{n + 1}')
                num[e].append(var)
                if n + 1 == n_duplicates_basic[e]:
                    model.AddHint(num[e][n], 1)
                else:
                    model.AddHint(num[e][n], 0)
                lum[e].append([])
                for k in range(self.n_devices):
                    var = model.NewBoolVar(f'L_{e}_{n + 1}_{k}')
                    lum[e][n].append(var)
                    if e in d2e_table_basic[k] and n + 1 == n_duplicates_basic[e]:
                        model.AddHint(lum[e][n][k], 1)
                    else:
                        model.AddHint(lum[e][n][k], 1)

        # 定义 N 限制条件
        total = 0
        for e in range(self.n_experts):
            model.Add(sum(num[e]) == 1)
            for n in range(up[e]):
                total += (n + 1) * num[e][n]
        model.Add(total == n_instances)

        # 定义 L 限制条件
        for e in range(self.n_experts):
            for n in range(up[e]):
                for k in range(self.n_devices):
                    model.Add(lum[e][n][k] <= num[e][n])
                    model.Add(lum[e][n][k] <= c[e][k])
                    model.Add(lum[e][n][k] >= num[e][n] + c[e][k] - 1)

        # 定义 c 和 N 之间的约束条件
        for e in range(self.n_experts):
            total = 0
            for n in range(up[e]):
                total += (n + 1) * num[e][n]
            model.Add(sum(c[e]) == total)

        # 定义每设备专家数均分约束
        for k in range(self.n_devices):
            model.Add(
                sum(c[e][k] for e in range(self.n_experts)) == n_instances // self.n_devices
            )
        # 定义 SF约束，对每个阶段m，定义单卡专家热度最大值不超过平均值的SF[m]倍
        for m in range(n_it_intvs):
            for k in range(self.n_devices):
                total = 0
                for e in range(self.n_experts):
                    for n in range(up[e]):
                        total += int(scaling_factor * coefficient * weights_intervals[m][e] / (n + 1)) * lum[e][n][k]
                model.Add(
                    total <=
                    int(scaling_factor * sum(weights_intervals[m]) / self.n_devices) * (coefficient + sf[m])
                )
                if n_it_intvs == 1:
                    model.Add(
                        total >=
                        int(scaling_factor * sum(weights_intervals[m]) / self.n_devices) * (coefficient - sf[m])
                    )

        sf_up = model.NewIntVar(low_bound, up_bound * sum(weight_lengths), f'SF_UP')

        model.Add(sum([sf[m] * weight_lengths[m]
                       for m in range(n_it_intvs)]) <= sf_up)
        obj = sf_up
        model.Minimize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self.cpu_per_process
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        status = solver.Solve(model, MySolutionCallback(layer_idx, sum(weight_lengths), self.shared_status))

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            d2e_table = np.zeros(
                (self.n_devices, (self.n_experts + self.n_red_experts) // self.n_devices), dtype=int)
            d2e_idxes = [0 for _ in range(self.n_devices)]
            for e in range(self.n_experts):
                for k in range(self.n_devices):
                    if solver.Value(c[e][k]) == 1:
                        d2e_table[k][d2e_idxes[k]] = e
                        d2e_idxes[k] += 1
            n_iterations = sum([weight_lengths[m] for m in range(n_it_intvs)])
            mean_sf = 0
            for m in range(n_it_intvs):
                mean_sf += solver.Value(sf[m]) / coefficient
            obj = [solver.Value(obj) / coefficient / n_iterations, mean_sf / n_it_intvs]
        else:
            d2e_table = d2e_table_basic
            obj = [99999, 99999]

        del model
        del solver

        return d2e_table, obj


class SimulateAnnealing(object):
    def __init__(self,
                 config,
                 shared_status
                 ):
        self.n_devices = config.n_devices  # 64
        self.n_nodes = config.n_nodes  # 8
        self.n_logical_experts = config.n_experts  # 256
        self.n_physical_experts = config.redundant_experts + config.n_experts  # 320
        self.n_exp_per_node = self.n_physical_experts / self.n_nodes  # 40
        self.n_exp_per_device = self.n_physical_experts / self.n_devices  # 5

        self.shared_status = shared_status

    # 黑盒评估函数_全量
    def black_box_evaluation(self, sequence, dynamic_expert_hot_layer):
        map_logical_physical = self.seq_2_map(sequence)
        # 最小化的目标函数
        rank_hot_iteration = np.zeros((self.n_devices, dynamic_expert_hot_layer.shape[1]), dtype=float)

        for exp, iteration in enumerate(dynamic_expert_hot_layer):
            logical_exp = int(exp)
            # 计算每组需要选择的专家数
            physical_experts = map_logical_physical[logical_exp]
            num_chosen = len(physical_experts)
            for physical_expert in physical_experts:
                physical_rank = int(physical_expert // self.n_exp_per_device)
                rank_hot_iteration[physical_rank] += [x / num_chosen for x in iteration]

        return sum(np.max(rank_hot_iteration, axis=0)), rank_hot_iteration

    # 黑盒评估函数_增量
    def black_box_evaluation_incremental(self, idx1, idx2, sequence_new, rank_hot_iteration_old,
                                         dynamic_expert_hot_layer):
        map_logical_physical = self.seq_2_map(sequence_new)
        # 最小化的目标函数
        rank_hot_iteration_new = rank_hot_iteration_old.copy()

        rank1 = int(idx1 // self.n_exp_per_device)
        rank2 = int(idx2 // self.n_exp_per_device)

        exp1 = sequence_new[idx1]
        exp2 = sequence_new[idx2]

        physical_experts1 = map_logical_physical[exp1]
        num_chosen1 = len(physical_experts1)

        physical_experts2 = map_logical_physical[exp2]
        num_chosen2 = len(physical_experts2)

        hot_exp1 = [x / num_chosen1 for x in dynamic_expert_hot_layer[exp1]]
        hot_exp2 = [x / num_chosen2 for x in dynamic_expert_hot_layer[exp2]]

        rank_hot_iteration_new[rank1] = rank_hot_iteration_new[rank1] - hot_exp2 + hot_exp1
        rank_hot_iteration_new[rank2] = rank_hot_iteration_new[rank2] - hot_exp1 + hot_exp2

        return sum(np.max(rank_hot_iteration_new, axis=0)), rank_hot_iteration_new

    def simulated_annealing(self, 
                            layer, 
                            initial_sequence, 
                            dynamic_expert_hot_layer, 
                            max_iter=10000, 
                            initial_temp=1000,
                            cooling_rate=0.99):
        current_sequence = initial_sequence.copy()
        current_score, cur_rank_iteration = self.black_box_evaluation(current_sequence, dynamic_expert_hot_layer)

        best_sequence = current_sequence.copy()
        best_score = current_score

        temp = initial_temp

        for i in range(max_iter):
            # 生成邻域解 - 随机交换两个元素
            new_sequence = current_sequence.copy()
            idx1, idx2 = random.sample(range(len(new_sequence)), 2)
            new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

            # 计算新解得分
            new_score, rank_iteration = self.black_box_evaluation_incremental(idx1, idx2, new_sequence,
                                                                              cur_rank_iteration,
                                                                              dynamic_expert_hot_layer)

            # 决定是否接受新解
            if new_score < current_score or random.random() < math.exp((current_score - new_score) / temp):
                current_sequence = new_sequence
                current_score = new_score
                cur_rank_iteration = rank_iteration

                # 更新最佳解
                if current_score < best_score:
                    best_sequence = current_sequence.copy()
                    best_score = current_score

            # 降低温度
            temp = 1 * ((max_iter - i) / max_iter) * ((max_iter - i) / max_iter)
            if i % 5000 == 0:
                self.shared_status[layer] = (
                    "Simulate_Annealing",
                    f"Iteration {i}: Temp={temp:.2f}, Current Score={current_score:.2f}, Best Score={best_score:.2f}")

        return best_sequence, best_score

    def map_2_seq(self, map_logical_physical_layer):
        seq = [0] * self.n_physical_experts
        for dev, phys in enumerate(map_logical_physical_layer):
            for i, phy in enumerate(phys):
                seq[dev * len(phys) + i] = phy

        return seq

    def seq_2_map(self, sequence_layer):
        map_logical_physical_layer = {}
        for index, exp_logical in enumerate(sequence_layer):
            exp_physical = index
            if exp_logical in map_logical_physical_layer:
                map_logical_physical_layer[exp_logical].append(exp_physical)
            else:
                map_logical_physical_layer[exp_logical] = [exp_physical]

        return map_logical_physical_layer


def second_optim(map_logical_physical, dynamic_expert_hot, config, shared_status, max_search=50000, n_search_multi=3):
    deploy_algo = SimulateAnnealing(config, shared_status)

    # 生成初始解（这里假设序列是0-319的排列，根据实际情况调整）
    map_logical_physical = parse_ep_file(config.deploy_fp, map_logical_physical,
                                         n_share_expert_devices=config.n_share_expert_devices)
    params = []
    initial_scores = []
    n_layers = range(config.selected_layers[0], config.selected_layers[1] + 1)
    for layer in n_layers:
        shared_status[layer] = ("Launching Simulate_Annealing", "")
        dynamic_expert_hot_layer = dynamic_expert_hot[layer]
        map_logical_physical_layer = map_logical_physical[layer]

        initial_sequence = deploy_algo.map_2_seq(map_logical_physical_layer)
        initial_score, _ = deploy_algo.black_box_evaluation(initial_sequence, dynamic_expert_hot_layer)
        initial_scores.append(initial_score)

        # 入参打包
        for _ in range(n_search_multi):
            params.append((layer, initial_sequence, dynamic_expert_hot_layer, max_search, 10000, 0.999))

    # 创建进程池
    with Pool() as pool:
        # 使用 starmap 并行执行 func，参数按元组解包
        results = pool.starmap(deploy_algo.simulated_annealing, params)

    benefit = []
    layout_all_layer = [None for _ in n_layers]
    layer2result = [[] for _ in n_layers]
    for param, result in zip(params, results):
        layer, initial_sequence, dynamic_expert_hot_layer, _, _, _ = param
        best_sequence, best_score = result
        initial_score, rank_iteration = deploy_algo.black_box_evaluation(initial_sequence, dynamic_expert_hot_layer)

        layer2result[layer].append(
            {"best_score": best_score, "best_sequence": best_sequence, "initial_score": initial_score})

    for layer in n_layers:
        best_sequence = min(layer2result[layer], key=lambda x: x["best_score"])["best_sequence"]
        best_score = min(layer2result[layer], key=lambda x: x["best_score"])["best_score"]
        initial_score = min(layer2result[layer], key=lambda x: x["best_score"])["initial_score"]

        layout = np.reshape(best_sequence, (config.n_devices, -1)).tolist()
        layout_all_layer[layer] = layout
        benefit.append((initial_score - best_score) / initial_score)

        shared_status[layer] = (
            "Simulate_Annealing End",
            f"Initial score: {initial_score:.2f}, Best score: {best_score:.2f}, Benefit: {benefit[-1] * 100:.2f}%")

    deploy_table = generate_expert_map_json(
        layout_all_layer,
        selected_layers=config.selected_layers,
        n_share_expert_devices=config.n_share_expert_devices)

    return deploy_table