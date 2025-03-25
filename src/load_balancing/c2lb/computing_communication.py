
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import logging

import numpy as np
import pandas as pd


# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger()


def save_matrix_to_csv(output_path, file_name, matrix):
    """
    保存矩阵到 CSV 文件或 Excel 文件（用于处理三维矩阵的多个 sheet）
    :param output_path: 输出文件的路径
    :param file_name: 输出文件的名字
    :param matrix: 矩阵
    """
    if matrix.ndim == 2:
        # 二维矩阵保存为普通 CSV 文件
        df = pd.DataFrame(matrix)
        file_name = f"{output_path}/{file_name}.csv"
        df.to_csv(file_name, index=False)
    elif matrix.ndim == 3:
        # 三维矩阵保存到 Excel 文件的不同 sheet 中
        file_name = f"{output_path}/{file_name}.xlsx"
        with pd.ExcelWriter(file_name) as writer:
            for i in range(matrix.shape[0]):
                slice_2d = matrix[i]
                df = pd.DataFrame(slice_2d)
                df.to_excel(writer, sheet_name=f'slice_{i}', index=False)
    else:
        logger.error(f"矩阵的维度 {matrix.ndim} 不支持，仅支持二维和三维矩阵。")


def save_matrix_to_json(output_path, file_name, deployment):
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": list(deployment[i][j])}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = f"{output_path}/{file_name}.json"
    # 保存为 JSON 文件
    try:
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"写入文件 {deployment} 时出错: {e}")


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
        output_path, 
        file_name,
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
        result, layer_deployment = compute_balanced_pack_redundancy(weights, num_npus, num_redundancy_expert, 0)
        for box in result:
            logger.info(
                f"before: Box {box['box_index']}: "
                f"Items = {box['items']}, weight = {box['weight']}, "
                f"Total Weight = {box['total_weight']}, Item Count = {box['item_count']}"
            )
        global_deployment[layer] = layer_deployment

    save_matrix_to_json(output_path, file_name, global_deployment)