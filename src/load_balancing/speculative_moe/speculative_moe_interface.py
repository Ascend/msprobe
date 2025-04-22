# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
 @copyright Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 @brief speculative-moe algorithm, output the expert placement table
 @date 2025-03-21
"""
import math
import logging
import csv
from collections import defaultdict
from typing import List
import multiprocessing as mp

import numpy as np
from ortools.sat.python import cp_model


logger = logging.getLogger("msit_logger")


class SpeculativeMoe(object):
    def __init__(self,
                 n_devices: int,
                 n_nodes: int,
                 n_experts: int,
                 n_red_experts: int,
                 trace_fp: str,
                 layer_idxes: List[int],
                 total_layers: int = 58,
                 cpu_per_process: int = 8,
                 max_time_in_seconds: int = 300,
                 max_steps: int = 10,
                 step_time: int = 60
                 ):
        self.n_devices = n_devices
        self.n_nodes = n_nodes
        self.n_experts = n_experts
        self.n_red_experts = n_red_experts
        self.trace_fp = trace_fp
        self.layer_idxes = layer_idxes
        self.total_layers = total_layers
        self.cpu_per_process = cpu_per_process
        self.max_time_in_seconds = max_time_in_seconds
        self.max_steps = max_steps
        self.step_time = step_time
        self.expert_weights = []

    def load_parse_data(self):
        data = []
        with open(self.trace_fp, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row, i in zip(reader, range(self.total_layers)):
                if i in self.layer_idxes:
                    data.append([int(item) for item in row])
        self.expert_weights = data
        return data

    def expert_grouping(self, local_layer_idx: int, layer_idx: int):
        """专家聚合函数，基于trace解析信息，输出每个device的专家布放信息"""
        model = None
        # 基于初始max_time_in_seconds进行ILP求解，如果超时未返回有效解，则增加step_time重复计算，直到返回有效解
        cur_step = 0
        while True:
            try:
                model = ILPSolver(expert_weights=self.expert_weights[local_layer_idx],
                                  layer_idx=layer_idx,
                                  n_nodes=self.n_nodes,
                                  n_devices=self.n_devices,
                                  n_experts=self.n_experts,
                                  n_red_experts=self.n_red_experts,
                                  max_time_in_seconds=self.max_time_in_seconds,
                                  cpu_per_process=self.cpu_per_process)
                model.fit()
                break
            except IndexError as e:
                self.max_time_in_seconds += self.step_time
                cur_step += 1
                if cur_step >= self.max_steps:
                    break
                continue
        return model.d2e_table if model is not None else None


class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, layer_idx):
        super().__init__()  # 使用super()调用父类的初始化方法
        self.layer_idx = layer_idx

    def on_solution_callback(self):
        obj = self.ObjectiveValue()  # best solution value
        bound = self.BestObjectiveBound()  # best bound
        logger.info(f"object: {obj}, bound: {bound}, layer: {self.layer_idx}")


class ILPSolver(object):
    def __init__(self,
                 expert_weights: List[int],
                 layer_idx,
                 n_nodes: int = 8,
                 n_devices: int = 64,
                 n_experts: int = 256,
                 n_red_experts: int = 64,
                 max_time_in_seconds: int = 300,
                 cpu_per_process: int = 8):
        self.weights = expert_weights
        self.layer_idx = layer_idx
        self.n_nodes = n_nodes
        self.n_devices = n_devices
        self.n_experts = n_experts
        self.n_red_experts = n_red_experts
        self.max_time_in_seconds = max_time_in_seconds
        self.cpu_per_process = cpu_per_process
        self.d2e_table = None

    @staticmethod
    def get_up(weights, n_experts, n_red_experts, hot_exp_n: int = 16, max_warm_dup_n: int = 2,
               max_hot_dup_n: int = 6, uni_up: int = 2, mode: int = 2):
        """获取每个专家的最大副本上限"""
        up = []
        # 模式1下，每个专家副本上限统一设置为固定值uni_up
        if mode == 1:
            for _ in range(n_experts):
                up.append(uni_up)
        # 模式2下，每个专家副本上限，根据冷热程度，设置为不同值
        elif mode == 2:
            up = [1] * n_experts
            sort_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=False)
            cold_exp_n = n_experts - n_red_experts
            warm_exp_n = n_red_experts - hot_exp_n
            reverse_sort_indices = sorted(sort_indices[-hot_exp_n:], reverse=True)
            sort_indices[-hot_exp_n:] = reverse_sort_indices
            warm_exp_idxes = sort_indices[cold_exp_n: cold_exp_n + warm_exp_n]
            hot_exp_idxes = sort_indices[-hot_exp_n:]
            for index in warm_exp_idxes:
                up[index] = max_warm_dup_n
            for index in hot_exp_idxes:
                up[index] = max_hot_dup_n
        return up

    def fit(self,
            hot_exp_n: int = 16,
            max_warm_dup_n: int = 2,
            max_hot_dup_n: int = 6
            ):
        low_bound, up_bound = 1, 2000
        coefficient = 100  # slack_factor的整数化系数
        model = cp_model.CpModel()
        n_red_expert_per_dev = self.n_red_experts // self.n_devices
        up1 = ILPSolver.get_up(self.weights, self.n_experts, self.n_red_experts, hot_exp_n,
                              max_warm_dup_n, max_hot_dup_n, 1, 2)
        max_dup_n = max(up1)
        scaling_factor = 1
        for i in range(1, max_dup_n + 1):
            scaling_factor = abs(scaling_factor * i) // math.gcd(scaling_factor, i)
        n_instances = self.n_experts
        if up1 != [1] * self.n_experts:
            n_instances += self.n_devices * n_red_expert_per_dev

        # 定义专家布放bool变量C，C[e][k] = 1表示专家e有副本布放在设备k上
        c = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for k in range(self.n_devices):
                var = model.NewBoolVar(f'C_{e}_{k}')
                c[e].append(var)

        # 定义slack_factor变量，衡量单卡专家热度最大值和平均值的比例，表征通信/计算的拖尾严重程度
        sf = model.NewIntVar(low_bound, up_bound, 'slack_factor')
        model.AddHint(sf, 1)
        # 定义 N 和 m 变量bool变量，其中N[e][n] = 1 表示专家e共布放n+1个副本，
        # m[e][n][k] = 1表示专家e共布放n+1个副本，且其中有一个副本布放在设备k上
        n1 = [[] for _ in range(self.n_experts)]
        m = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for n in range(up1[e]):
                var = model.NewBoolVar(f'N_{e}_{n + 1}')
                n1[e].append(var)
                m[e].append([])
                for k in range(self.n_devices):
                    var = model.NewBoolVar(f'L_{e}_{n + 1}_{k}')
                    m[e][n].append(var)

        # 定义 N 限制条件
        total = 0
        for e in range(self.n_experts):
            model.Add(sum(n1[e]) == 1)
            for n in range(up1[e]):
                total += (n + 1) * n1[e][n]
        model.Add(total == n_instances)

        # 定义 m 限制条件
        for e in range(self.n_experts):
            for n in range(up1[e]):
                for k in range(self.n_devices):
                    model.Add(m[e][n][k] <= n1[e][n])
                    model.Add(m[e][n][k] <= c[e][k])
                    model.Add(m[e][n][k] >= n1[e][n] + c[e][k] - 1)

        # 定义 C 和 N 之间的约束条件
        for e in range(self.n_experts):
            total = 0
            for n in range(up1[e]):
                total += (n + 1) * n1[e][n]
            model.Add(sum(c[e]) == total)

        # 定义每设备专家数均分约束
        for k in range(self.n_devices):
            model.Add(
                sum(c[e][k] for e in range(self.n_experts)) == n_instances // self.n_devices
            )
        # 定义 slack_factor约束，单卡专家热度最大值不超过平均值的slack_factor倍
        for k in range(self.n_devices):
            total = 0
            for e in range(self.n_experts):
                for n in range(up1[e]):
                    total += int(scaling_factor * coefficient * self.weights[e] / (n + 1)) * m[e][n][k]
            model.Add(
                total <=
                int(scaling_factor * sum(self.weights) / self.n_devices) * (coefficient + sf)
            )
            model.Add(
                total >=
                int(scaling_factor * sum(self.weights) / self.n_devices) * (coefficient - sf)
            )
        obj = -sf
        model.Maximize(obj)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self.cpu_per_process
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        solver.Solve(model, MySolutionCallback(self.layer_idx))

        self.d2e_table = np.zeros(
            (self.n_devices, (self.n_experts + self.n_red_experts) // self.n_devices), dtype=int)
        d2e_idxes = [0 for _ in range(self.n_devices)]
        for e in range(self.n_experts):
            for k in range(self.n_devices):
                if solver.Value(c[e][k]) == 1:
                    self.d2e_table[k][d2e_idxes[k]] = e
                    d2e_idxes[k] += 1
        logger.info(f'layer_idx: {self.layer_idx}')
        logger.debug(self.d2e_table)
        logger.debug(solver.Value(sf) / coefficient)
        del model
        del solver
        

def speculative_moe_algo_multi_process(n_devices: int,
                                       n_nodes: int,
                                       n_layers: int,
                                       n_experts: int,
                                       n_red_experts: int,
                                       trace_fp: str,
                                       cpu_per_process: int = 8):
    """多进程执行speculative_moe_algo, 每个进程独立求解不同layers的专家布放结果"""
    n_processes = (mp.cpu_count() - 3) // cpu_per_process
    worker_args = []
    for layer_idx in range(n_layers):
        worker_args.append(
            (trace_fp, [layer_idx], cpu_per_process, n_devices, n_nodes, n_experts, n_red_experts, n_layers))
    task_remain = len(worker_args)
    results = []
    start_idx = 0
    while task_remain > 0:
        task_num = min(n_processes, task_remain)
        with mp.Pool(task_num) as pool:
            cur_results = pool.starmap(speculative_moe_algo, worker_args[start_idx: start_idx + task_num])
        task_remain -= task_num
        start_idx += task_num
        for result in cur_results:
            if result is None:
                logger.error('ERROR: can not solve the e2d tables for all layers successfully')
                return None
        results.extend(cur_results)
    return results


def speculative_moe_algo(trace_fp: str,
                         layer_indexes: List[int],
                         cpu_per_process: int,
                         n_devices: int = 64,
                         n_nodes: int = 8,
                         n_experts: int = 256,
                         n_red_experts: int = 64,
                         n_layers: int = 58):
    """单进程speculative_moe_algo, 负责对特定layer_indexes进行专家布放求解"""
    deploy_algo = SpeculativeMoe(
        n_devices, n_nodes, n_experts, n_red_experts, trace_fp, layer_indexes, n_layers, cpu_per_process)
    deploy_algo.load_parse_data()
    d2e_tables = defaultdict()
    for i, layer_idx in enumerate(deploy_algo.layer_idxes):
        logger.debug('expert grouping: layer %d', layer_idx)
        d2e_table = deploy_algo.expert_grouping(i, layer_idx)
        if d2e_table is None:
            logger.error('ERROR: cannot solver feasible d2e table in given steps')
            return None
        d2e_tables[deploy_algo.layer_idxes[i]] = d2e_table
    d2e_tables = dict(sorted(d2e_tables.items()))
    return d2e_tables