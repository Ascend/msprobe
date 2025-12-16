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

import json
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger("msit_logger")


# 热点专家拆分为冗余专家
def compute_balanced_pack_redundancy(origin_weights, card_num, num_redundancy_expert, is_only):
    # Step 1: Sort the items by weight in descending order (we are sorting by weight now)
    # Sort based on the second element (the second value of each tuple)
    route_expert_num = len(origin_weights)
    route_expert_redundancy = [[] for _ in range(route_expert_num)]
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

    # Step 2: Calculate the number of items per box
    expert_num = route_expert_num + num_redundancy_expert
    items_per_box = expert_num // card_num  # Number of items per box
    remaining_items = expert_num % card_num  # Number of items per box

    # Step 3: Initialize card_num boxes with empty lists to store item IDs
    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num  # To store the total weight of each box
    box_counts = [0] * card_num  # To store the number of items in each box
    index = 0
    for i in range(route_expert_num):
        redundancy_num = len(route_expert_redundancy[i])
        for _ in range(redundancy_num):
            cur_weight = 0
            for item, weight in origin_weights:
                if item == i:
                    cur_weight = weight
            if index >= card_num:
                logger.error("Index Out of Bounds")
                break
            boxes[index].append(i)
            boxes_weights[index].append(cur_weight)
            box_weights[index] += cur_weight
            box_counts[index] += 1
            index += 1

    sorted_indices = np.argsort([t[1] for t in origin_weights], kind='stable')[::-1]
    origin_weights = [origin_weights[idx] for idx in sorted_indices]
    # Step 4: Distribute items into boxes based on weight
    for item_id, weight in origin_weights:
        # Find the box with the least items but not full
        min_box_index = -1
        for i in range(card_num):
            # Only choose boxes that still have space (box_counts[i] < items_per_box)
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    min_box_index = i

        # Place the item (id) into the selected box
        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        # If there's an imbalance in the remaining items, reduce the "remaining_items" counter
        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    # Step 5: Output each box's contents and total weight
    result = []
    for i in range(card_num):
        result.append({
            "box_index": i + 1,
            "items": boxes[i],  # List of item IDs in the box
            "weight": boxes_weights[i],
            "total_weight": box_weights[i],  # Total weight in this box
            "item_count": box_counts[i]  # Number of items in the box
        })

    return result, boxes


# 无冗余专家方案
def compute_balanced_pack(origin_weights, card_num):
    # Step 1: Sort the items by weight in descending order (we are sorting by weight now)
    # Sort based on the second element (the second value of each tuple)
    sorted_indices = np.argsort([t[1] for t in origin_weights])[::-1]

    # Output the sorted array using the sorted indices
    weights = origin_weights[sorted_indices]

    # Step 2: Calculate the number of items per box
    expert_num = len(weights)
    items_per_box = expert_num // card_num  # Number of items per box
    remaining_items = expert_num % card_num  # Number of items per box

    # Step 3: Initialize card_num boxes with empty lists to store item IDs
    boxes = [[] for _ in range(card_num)]
    box_weights = [0] * card_num  # To store the total weight of each box
    box_counts = [0] * card_num  # To store the number of items in each box

    # Step 4: Distribute items into boxes based on weight
    for item_id, weight in weights:
        # Find the box with the least items but not full
        min_box_index = -1
        for i in range(card_num):
            # Only choose boxes that still have space (box_counts[i] < items_per_box)
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    min_box_index = i

        # Place the item (id) into the selected box
        boxes[min_box_index].append(item_id)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        # If there's an imbalance in the remaining items, reduce the "remaining_items" counter
        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    # Step 5: Output each box's contents and total weight
    result = []
    for i in range(card_num):
        result.append({
            "box_index": i + 1,
            "items": boxes[i],  # List of item IDs in the box
            "total_weight": box_weights[i],  # Total weight in this box
            "item_count": box_counts[i]  # Number of items in the box
        })

    return result, boxes


# 冗余专家部署
def lb_and_intra_layer_affinity_redundancy_deploy(
        layer_workloads,  
        num_redundancy_expert, 
        num_npus=64, 
        num_original_expert=256,):
    """
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """
    # 计算负载均衡，部署冗余专家
    layer_num = layer_workloads.shape[0]
    expert_num = layer_workloads.shape[1]
    # 校验专家数量、卡数量、冗余专家数量不能超过卡数量
    if num_original_expert != expert_num:
        raise ValueError(f"原始专家数量 {num_original_expert} 必须等于 expert_num {expert_num}")
    
    if num_npus <= 0:
        raise ValueError("NPUs 数量必须大于 0")
    
    if num_npus < num_redundancy_expert:
        raise ValueError(f"NPUs 数量 {num_npus} 必须大于或等于冗余专家数量 {num_redundancy_expert}")
        
    # 每个卡部署的专家数量 一个冗余专家
    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
    # 遍历获得每一层的放置策略，考虑计算均衡
    for layer in range(layer_num):
        # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
        weights = np.zeros((expert_num,), dtype='object')
        for expert_id, workload_weight in enumerate(layer_workloads[layer]):
            weights[expert_id] = (expert_id, workload_weight)

        # 获取每一层全局计算均衡的放置策略
        _, layer_deployment = compute_balanced_pack_redundancy(weights, num_npus, num_redundancy_expert, 0)
        global_deployment[layer] = layer_deployment

    return global_deployment