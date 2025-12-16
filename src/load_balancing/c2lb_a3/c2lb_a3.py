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

"""
 @copyright Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 @brief speculative-moe algorithm, output the expert placement table
 @date 2025-03-21
"""

import json
import logging
import copy
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger("msit_logger")


def are_elements_equal(list1, list2):
    return Counter(list1) == Counter(list2)


def initialize_boxes(card_num):
    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num
    box_counts = [0] * card_num
    return boxes, boxes_weights, box_weights, box_counts


def process_redundancy_experts(weights, route_expert_redundancy, select_expert,
                    max_avg_weight=None, min_avg_weight=None, route_expert_num=None, i=None, index=None,
                    boxes=None, boxes_weights=None, box_weights=None, box_counts=None):
    cur_expert_id = weights[0][0]
    select_expert.add(cur_expert_id)
    tmp_raw_weight = weights[0][1] * (len(route_expert_redundancy[cur_expert_id]) + 1)
    route_expert_redundancy[cur_expert_id].append(route_expert_num + i)
    tmp_num_redundancy = len(route_expert_redundancy[cur_expert_id]) + 1
    avg_weight = tmp_raw_weight / tmp_num_redundancy
    box_flag = True
    if tmp_num_redundancy < len(weights):
        weights_last_index = len(weights) - 1
        tmp_max_box_weight = avg_weight + weights[weights_last_index - tmp_num_redundancy + 1][1]
        tmp_min_box_weight1 = avg_weight + weights[weights_last_index][1]
        last_expert_id = set()
        for k in range(1, tmp_num_redundancy + 1):
            weights_last_index = len(weights) - k
            last_expert_id.add(weights[weights_last_index][0])
        intersection = last_expert_id & select_expert
        if tmp_min_box_weight1 > min_avg_weight and tmp_max_box_weight < max_avg_weight and not intersection:
            weights.pop(0)
            box_flag = False
            for _ in range(1, tmp_num_redundancy + 1):
                boxes[index].append(cur_expert_id)
                boxes_weights[index].append(avg_weight)
                box_weights[index] += avg_weight
                box_counts[index] += 1

                weights_last_index = len(weights) - 1
                boxes[index].append(weights[weights_last_index][0])
                boxes_weights[index].append(weights[weights_last_index][1])
                box_weights[index] += weights[weights_last_index][1]
                box_counts[index] += 1
                index += 1
                weights.pop(weights_last_index)
    if box_flag:
        weights[0] = (cur_expert_id, avg_weight)
    else:
        route_expert_redundancy[cur_expert_id] = []
    return weights, route_expert_redundancy, select_expert, index, boxes, boxes_weights, box_weights, box_counts


def pack_remaining_experts(origin_weights, route_expert_redundancy, new_expert_num=-1, route_expert_num=-1, 
                        items_per_box=-1, remaining_items=-1, index=-1, card_num=-1, 
                        boxes=None, boxes_weights=None, box_weights=None, box_counts=None):
    all_weights = np.zeros((new_expert_num,), dtype='object')
    len_weights = len(origin_weights)
    all_weights[: len_weights] = origin_weights
    test_num = 0
    weight_dict = dict(origin_weights)
    for i in range(route_expert_num + 1):
        redundancy_num = len(route_expert_redundancy[i])
        test_num += redundancy_num

        for _ in range(redundancy_num):
            if i in weight_dict.keys():
                all_weights[len_weights] = (i, weight_dict[i])
                len_weights += 1

    sorted_indices = np.argsort([t[1] for t in all_weights], kind='stable')[::-1]
    all_weights = [all_weights[idx] for idx in sorted_indices]
    for item_id, weight in all_weights:
        min_box_index = -1
        for i in range(index, card_num):
            item_count_flag = box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0)
            box_flag = min_box_index == -1 or box_weights[i] < box_weights[min_box_index]
            if item_count_flag and box_flag:
                min_box_index = i

        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1
    return boxes, boxes_weights, box_weights, box_counts


def a3_new_compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, ave_weights, threshold):
    route_expert_num = len(origin_weights)
    expert_num = route_expert_num + num_redundancy_expert
    items_per_box = expert_num // card_num
    remaining_items = expert_num % card_num

    route_expert_redundancy = [[]for _ in range(route_expert_num + 1)]
    max_avg_weight = (1 + threshold) * ave_weights
    min_avg_weight = (1 - threshold) * ave_weights

    boxes, boxes_weights, box_weights, box_counts = initialize_boxes(card_num)

    index = 0
    select_expert = set()
    for i in range(num_redundancy_expert):
        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        weights = [origin_weights[idx] for idx in sorted_indices]
        origin_weights, route_expert_redundancy, select_expert, index, \
                boxes, boxes_weights, box_weights, box_counts = process_redundancy_experts(
                weights, route_expert_redundancy, select_expert,
                max_avg_weight, min_avg_weight, route_expert_num, i, index, 
                boxes, boxes_weights, box_weights, box_counts
            )

    new_expert_num = expert_num - index * 2

    boxes, boxes_weights, box_weights, box_counts = pack_remaining_experts(
        origin_weights, route_expert_redundancy, new_expert_num, route_expert_num,
        items_per_box, remaining_items, index, card_num,
        boxes, boxes_weights, box_weights, box_counts
    )

    result = []
    max_weight = 0
    for i in range(card_num):
        if box_weights[i] >= max_weight:
            max_weight = box_weights[i]
        result.append({
            "box_index": i + 1,
            "items": boxes[i],
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],
            "item_count": box_counts[i]
        })

    return result, boxes, max_weight


def get_redundant_experts(origin_weights, num_redundancy_expert, route_expert_num, is_only):
    route_expert_redundancy = [[] for _ in range(route_expert_num + 1)]
    if is_only == 1:
        sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
        weights = [origin_weights[idx] for idx in sorted_indices]
        for i in range(num_redundancy_expert):
            route_expert_redundancy[weights[i][0]].append(route_expert_num + i)
            avg_weight = weights[i][1] / (len(route_expert_redundancy[weights[0][0]]) + 1)
            weights[i] = (weights[i][0], avg_weight)
    else:
        for i in range(num_redundancy_expert):
            sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
            weights = [origin_weights[idx] for idx in sorted_indices]
            tmp_raw_weight = weights[0][1] * (len(route_expert_redundancy[weights[0][0]]) + 1)
            route_expert_redundancy[weights[0][0]].append(route_expert_num + i)
            avg_weight = tmp_raw_weight / (len(route_expert_redundancy[weights[0][0]]) + 1)
            weights[0] = (weights[0][0], avg_weight)
            origin_weights = weights
    return route_expert_redundancy, origin_weights


def deploy_all_experts(route_expert_redundancy, origin_weights, card_num, route_expert_num, num_redundancy_expert):
    expert_num = route_expert_num + num_redundancy_expert
    items_per_box = expert_num // card_num 
    remaining_items = expert_num % card_num 
    
    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num  
    box_counts = [0] * card_num 

    all_weights = np.zeros((expert_num,), dtype='object')
    all_weights[: route_expert_num] = origin_weights

    weight_dict = dict(origin_weights)
    index = route_expert_num
    for i in range(route_expert_num + 1):
        redundancy_num = len(route_expert_redundancy[i])
        if i in weight_dict:  
            for _ in range(redundancy_num):
                all_weights[index] = (i, weight_dict[i])
                index += 1

    sorted_indices = np.argsort([t[1] for t in all_weights], kind='stable')[::-1]
    all_weights = [all_weights[idx] for idx in sorted_indices]
    for item_id, weight in all_weights:
        min_box_index = -1
        for i in range(card_num):
            item_count_flag = box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0)
            box_flag = min_box_index == -1 or box_weights[i] < box_weights[min_box_index]
            if item_count_flag and box_flag:
                min_box_index = i

        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    return boxes, boxes_weights, box_weights, box_counts


def new_compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, average_weight, is_only):
    route_expert_num = len(origin_weights)
    route_expert_redundancy, origin_weights = get_redundant_experts(
        origin_weights, num_redundancy_expert, route_expert_num, is_only
    )

    boxes, boxes_weights, box_weights, box_counts = deploy_all_experts(
        route_expert_redundancy, origin_weights, card_num, route_expert_num, num_redundancy_expert
    )

    result = []
    max_weight = 0
    for i in range(card_num):
        if box_weights[i] >= max_weight:
            max_weight = box_weights[i]
        result.append({
            "box_index": i + 1,
            "items": boxes[i],  # List of item IDs in the box
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],  # Total weight in this box
            "item_count": box_counts[i]  # Number of items in the box
        })

    return result, boxes, max_weight


def recover_layer_results(layer_deployment, layer_raw_workload):
    tmp_experts = layer_deployment.flatten()
    expert_num = np.max(tmp_experts) + 1
    tmp_workload = copy.deepcopy(layer_raw_workload) / np.bincount(tmp_experts, minlength=expert_num)
    boxes = copy.deepcopy(layer_deployment)
    boxes_weights = tmp_workload[layer_deployment]
    box_weights = np.sum(boxes_weights, axis=1)
    box_counts = np.array([len(b) for b in layer_deployment])
    return boxes, boxes_weights, box_weights, box_counts


def iter_swap_experts(boxes, boxes_weights, box_weights, box_counts):
    card_num = len(boxes)
    logger.info(f"Before swap, max box weight is: {np.max(box_weights)}, min box weight is: {np.min(box_weights)}")
    # Step 4-2: 迭代优化放置策略
    swap_num = 0
    while True:
        sorted_box_indices = np.argsort(box_weights)[::-1] # 根据每张卡上workload降序排序
        max_workload_box_index = sorted_box_indices[0] # workload 最大的卡
        swap_flag = False # 判断是否发生了专家交换
        for swap_index in range(card_num-1, 0, -1):
            min_workload_box_index = sorted_box_indices[swap_index] # workload 最小的卡
            new_boxes_weight1, new_boxes_weight2, new_box_expert1, new_box_expert2, swap_flag = \
                swap_box(boxes_weights[max_workload_box_index], boxes_weights[min_workload_box_index], 
                    boxes[max_workload_box_index], boxes[min_workload_box_index])
            if swap_flag: # 发生了交换
                swap_num += 1
                boxes_weights[max_workload_box_index] = new_boxes_weight1
                box_weights[max_workload_box_index] = np.sum(boxes_weights[max_workload_box_index])
                boxes_weights[min_workload_box_index] = new_boxes_weight2
                box_weights[min_workload_box_index] = np.sum(boxes_weights[min_workload_box_index])
                boxes[max_workload_box_index] = new_box_expert1
                boxes[min_workload_box_index] = new_box_expert2
                break # 发生了交换则跳出循环
        if swap_flag: # 如果没有发生交换，退出
            break
        if swap_num % 100 == 0:
            logger.info(
                f"swap_num={swap_num}, max box weight is: {np.max(box_weights)},"
                f"min box weight is: {np.min(box_weights)}"
            )
    logger.info(
        f"After {swap_num} swap, max box weight is: {np.max(box_weights)}," 
        f"min box weight is: {np.min(box_weights)}"
    )
    return boxes, boxes_weights, box_weights, box_counts


def swap_box(weight1, weight2, expert_index1, expert_index2):
    """
    不同box之间交换
    :param weight1: 权重大的box对应权重
    :param weight2: 权重小的box对应权重
    :param index1: 权重大的box对应专家index
    :param index2: 权重小的box对应专家index
    :return: _description_
    """
    sum1_original = np.sum(weight1)
    sum2_original = np.sum(weight2)
    raw_max_weight = max(sum1_original, sum2_original)
    sum_total = sum1_original + sum2_original
    a = sum_total / 2.0
    elements = np.array(list(weight1) + list(weight2))
    expert_index = np.array(list(expert_index1) + list(expert_index2))
    all_combinations = itertools.combinations(range(len(elements)), len(elements)//2)
    candidates = []
    all_list = list(all_combinations)
    all_sum = []
    # all_index 
    for i, combo in enumerate(all_list):
        try:
            values = [elements[i] for i in combo]
        except Exception as e:
            raise (f"Error processing {combo}: {e}") from e
        sum3 = np.sum(values)
        all_sum.append(sum3)
    
    all_sum_np = np.array(all_sum)
    all_sum_sorted_index = np.argsort(all_sum_np, kind='stable')

    new_index1 = [i for i in range(len(elements)//2)]
    new_index2 = [i for i in range(len(elements)//2, len(elements))]
    for i, index in enumerate(all_sum_sorted_index):
        cur_new_workload = all_sum_np[index]
        if cur_new_workload > a and cur_new_workload < raw_max_weight: # 有收益，小于最大值
            new_index1 = list(all_list[all_sum_sorted_index[i]])
            new_index2 = list(set(range(len(elements))) - set(new_index1))
            break
    
    new_weight1 = elements[new_index1]
    new_weight2 = elements[new_index2]
    new_expert_index1 = expert_index[new_index1]
    new_expert_index2 = expert_index[new_index2]
    equal_expert_flag = are_elements_equal(expert_index1, new_expert_index1) # 原始专家是否相等
    new_max_weight = max(np.sum(new_weight1), np.sum(new_weight2))
    if new_max_weight > raw_max_weight:
        raise ValueError("专家交换错误，检查参数输入是否正常。")
    return list(new_weight1), list(new_weight2), list(new_expert_index1), \
        list(new_expert_index2), equal_expert_flag is False


def deployment_iter_swap(global_deployment: np.ndarray, workload_data: np.ndarray):
    global_deployment = np.array(global_deployment)
    layer_num = len(global_deployment)
    all_layers_deployment = []
    for i in range(layer_num):
        logger.info(f"\nProcessing layer {i}...")
        boxes, boxes_weights, box_weights, box_counts = recover_layer_results(
            layer_deployment=global_deployment[i], 
            layer_raw_workload=workload_data[i]
        )
        new_boxes, _, _, _ = iter_swap_experts(
            boxes, boxes_weights, box_weights, box_counts
        )
        # break
        all_layers_deployment.append(new_boxes)
    all_layers_deployment = np.array(all_layers_deployment).astype(int) # 新的部署策略
    return all_layers_deployment


def get_layer_weight(layer_workloads, layer: int):
    """
    获取特定层上每个专家负载和总负载
    """
    _, expert_num = layer_workloads.shape
    weights = np.zeros((expert_num,), dtype='object')
    total_weight = 0
    for expert_id, workload_weight in enumerate(layer_workloads[layer]):
        total_weight += workload_weight
        weights[expert_id] = (expert_id, workload_weight)
    return total_weight, weights


# 冗余专家部署
def lb_and_intra_layer_affinity_redundancy_deploy_a3(
        layer_workloads,
        num_redundancy_expert,
        num_npus,
        num_original_expert,
        iter_swap_flag=True
        ):
    """
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """
    threshold = 0.05 # 新的专家冗余切分时，阈值固定为0.05

    # 计算负载均衡，部署冗余专家
    layer_num, expert_num = layer_workloads.shape
    # 校验专家数量、卡数量、冗余专家数量不能超过卡数量
    if num_original_expert != expert_num:
        raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")

    if num_npus <= 0:
        raise ValueError("NPUs 数量必须大于 0")

    if num_npus == 256 and num_redundancy_expert == 256:
        logger.info("调用新的冗余专家切分策略代码")
    else:
        logger.info("调用不加专家放置限制的C2LB代码,不做专家迭代交换操作")
        iter_swap_flag = False # 保证运行速度，不做迭代专家交换

    # 每个卡部署的专家数量 一个冗余专家
    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
    # 遍历获得每一层的放置策略，考虑计算均衡
    for layer in range(layer_num):
        # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
        total_weight, weights = get_layer_weight(layer_workloads, layer)

        average_weight = total_weight//num_npus
        # 获取每一层全局计算均衡的放置策略
        if num_npus == 256 and num_redundancy_expert == 256:
            # 调用新的冗余专家切分策略代码
            _, layer_deployment, _ = a3_new_compute_balanced_pack_redundancy(
                weights,
                num_npus,
                num_redundancy_expert,
                average_weight,
                threshold
            )
        else:
            _, layer_deployment, _ = new_compute_balanced_pack_redundancy(
                weights,
                num_npus,
                num_redundancy_expert,
                average_weight,
                is_only=0 # 默认切分多份冗余专家
            )

        global_deployment[layer] = layer_deployment

    global_deployment = np.array(global_deployment)
    if iter_swap_flag:
        logger.info(f"\nStart iter swap experts")
        global_deployment = deployment_iter_swap(
            global_deployment=global_deployment,
            workload_data=layer_workloads
        )
        
        logger.info(f"\nIter swap experts finished, global_deployment shape is {global_deployment.shape}")

    return global_deployment






