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
                 max_time_in_seconds: int = 600,
                 max_steps: int = 10,
                 step_time: int = 600
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
                    data.append([int(float(item)) for item in row])
        self.expert_weights = data

    def expert_grouping(self, local_layer_idx: int, layer_idx: int):
        model = None
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
                logger.debug('cur_step: %d layer: %s', cur_step, self.layer_idxes)
                if cur_step >= self.max_steps:
                    break
                continue
        return (model.d2e_table, model.obj) if model is not None else None


class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def on_solution_callback(self):
        obj = self.ObjectiveValue()
        bound = self.BestObjectiveBound()
        logger.info(f"object: {obj}, bound: {bound}, layer: {self.layer_idx}")
    

class ILPSolver(object):
    def __init__(self,
                 expert_weights: List[int],
                 layer_idx: int,
                 n_nodes: int,
                 n_devices: int,
                 n_experts: int,
                 n_red_experts: int,
                 max_time_in_seconds: int,
                 cpu_per_process: int):
        self.weights = expert_weights
        self.layer_idx = layer_idx
        self.n_nodes = n_nodes
        self.n_devices = n_devices
        self.n_experts = n_experts
        self.n_red_experts = n_red_experts
        self.max_time_in_seconds = max_time_in_seconds
        self.cpu_per_process = cpu_per_process
        self.d2e_table = None
        self.obj = 0

    def fit(self):
        low_bound, up_bound = 1, 5000
        coefficient = 100 
        model = cp_model.CpModel()
        up_0 = self.get_up(self.weights, self.n_experts, self.n_red_experts, 1, 2)
        scaling_factor = 1
        for i in range(1, 10 + 1):
            scaling_factor = abs(scaling_factor * i) // math.gcd(scaling_factor, i)
        n_instances = self.n_experts
        if up_0 != [1] * self.n_experts:
            n_instances += self.n_red_experts
        c, sf, _, _ = self.add_variables_and_constraints(model, low_bound, up_bound, up_0, n_instances, scaling_factor,
                                                            coefficient)
        model.Minimize(sf)
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
        self.obj = solver.Value(sf) / coefficient
        logger.info(f'obj: {solver.Value(sf) / coefficient} layer_idx: {self.layer_idx}')
        del model
        del solver

    def add_variables_and_constraints(self, 
                                    model, 
                                    low_bound, 
                                    up_bound, 
                                    up_0, 
                                    n_instances, 
                                    scaling_factor, 
                                    coefficient=100):
        c = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for k in range(self.n_devices):
                var = model.NewBoolVar(f'C_{e}_{k}')
                c[e].append(var)
        sf = model.NewIntVar(low_bound, up_bound, 'slack_factor')
        _n = [[] for _ in range(self.n_experts)]
        _l = [[] for _ in range(self.n_experts)]
        for e in range(self.n_experts):
            for n in range(up_0[e]):
                var = model.NewBoolVar(f'N_{e}_{n + 1}')
                _n[e].append(var)
                _l[e].append([])
                for k in range(self.n_devices):
                    var = model.NewBoolVar(f'L_{e}_{n + 1}_{k}')
                    _l[e][n].append(var)
        total = 0
        for e in range(self.n_experts):
            model.Add(sum(_n[e]) == 1)
            for n in range(up_0[e]):
                total += (n + 1) * _n[e][n]
        model.Add(total == n_instances)
        for e in range(self.n_experts):
            for n in range(up_0[e]):
                for k in range(self.n_devices):
                    model.Add(_l[e][n][k] <= _n[e][n])
                    model.Add(_l[e][n][k] <= c[e][k])
                    model.Add(_l[e][n][k] >= _n[e][n] + c[e][k] - 1)
        for e in range(self.n_experts):
            total = 0
            for n in range(up_0[e]):
                total += (n + 1) * _n[e][n]
            model.Add(sum(c[e]) == total)
        for k in range(self.n_devices):
            model.Add(
                sum(c[e][k] for e in range(self.n_experts)) == n_instances // self.n_devices
            )
        for k in range(self.n_devices):
            total = 0
            for e in range(self.n_experts):
                for n in range(up_0[e]):
                    total += int(scaling_factor * coefficient * self.weights[e] / (n + 1)) * _l[e][n][k]
            model.Add(total <= int(scaling_factor * sum(self.weights) / self.n_devices) * (coefficient + sf))
        return c, sf, _n, _l

    def get_up(self, weights, n_experts, n_red_expert, uni_up: int = 2, mode: int = 2):
        up = []
        if mode == 1:
            for _ in range(n_experts):
                up.append(uni_up)
        elif mode == 2:
            up = [1] * n_experts
            sort_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
            sorted_weights = sorted(weights, reverse=True)
            mean_weight = sum(weights) // len(weights)
            index = 0
            for i, _ in enumerate(sort_indices):
                if sorted_weights[i] / mean_weight >= 0.75:
                    up[sort_indices[i]] = int(sorted_weights[i] / (0.75 * mean_weight)) + 2
                else:
                    index = i
                    break
            while sum(up) < n_experts + 1 * n_red_expert:
                for i in range(index):
                    up[sort_indices[i]] += 1
        return up
    

def speculative_moe_algo(n_devices: int,
                         n_nodes: int,
                         n_experts: int,
                         n_red_experts: int,
                         trace_fp: str,
                         layer_indexes: List[int],
                         n_layers: int = 58,
                         cpu_per_process: int = 4,
                         max_time_in_seconds: int = 900):

    deploy_algo = SpeculativeMoe(n_devices, n_nodes, n_experts, n_red_experts, trace_fp, layer_indexes, n_layers,
                                 cpu_per_process, max_time_in_seconds)
    deploy_algo.load_parse_data()
    d2e_tables = defaultdict()
    objs = defaultdict()
    for i, layer_idx in enumerate(deploy_algo.layer_idxes):
        logger.info(f'expert grouping: layer {layer_idx}')
        d2e_table, obj = deploy_algo.expert_grouping(i, layer_idx)
        if d2e_table is None:
            raise ValueError('ERROR: cannot solver feasible d2e table in given steps')
        d2e_tables[layer_idx] = d2e_table
        objs[layer_idx] = obj
    d2e_tables = dict(sorted(d2e_tables.items()))
    objs = dict(sorted(objs.items()))
    return d2e_tables, objs


def speculative_moe_algo_multi_process_a3(n_devices: int, 
                                          n_nodes: int,
                                          n_layers: int,
                                          n_experts: int,
                                          n_red_experts: int,
                                          trace_fp: str, 
                                          max_time_in_seconds: int = 900, 
                                          cpu_per_process: int = 4
                                          ):
    n_processes = (mp.cpu_count() - 3) // cpu_per_process
    worker_args = []
    for layer_idx in range(n_layers):
        worker_args.append(
            (n_devices, n_nodes, n_experts, n_red_experts, trace_fp, [layer_idx], n_layers, cpu_per_process,
             max_time_in_seconds))
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
                raise ValueError('can not solve the e2d tables for all layers successfully')
        results.extend(cur_results)
    return results