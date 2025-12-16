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

import logging

import numpy as np


logger = logging.getLogger("msit_logger")


def get_redundant_experts(
    origin_weights, num_redundancy_expert, expert_id_bias, route_expert_num, is_only
):
    """
    识别冗余专家并划分
    """
    route_expert_redundancy = [[] for _ in range(route_expert_num)]

    if is_only == 1:
        weights = sorted(origin_weights, key=lambda x: -x[1])
        for i in range(num_redundancy_expert):
            route_expert_redundancy[weights[i][0] - expert_id_bias].append(
                route_expert_num + i
            )
            avg_weight = weights[i][1] / (
                len(route_expert_redundancy[weights[0][0] - expert_id_bias]) + 1
            )
            weights[i] = (weights[i][0], avg_weight)
    else:
        weights = origin_weights
        for i in range(num_redundancy_expert):
            weights = sorted(weights, key=lambda x: -x[1])
            tmp_raw_weight = weights[0][1] * (
                len(route_expert_redundancy[weights[0][0] - expert_id_bias]) + 1
            )
            route_expert_redundancy[weights[0][0] - expert_id_bias].append(
                route_expert_num + i
            )
            avg_weight = tmp_raw_weight / (
                len(route_expert_redundancy[weights[0][0] - expert_id_bias]) + 1
            )
            weights[0] = (weights[0][0], avg_weight)

    return route_expert_redundancy, weights


def deploy_redundant_expert(route_expert_redundancy, origin_weights, expert_id_bias, card_num):
    """
    优先放置冗余专家
    """
    boxes = [[] for _ in range(card_num)]
    boxes_weights = [[] for _ in range(card_num)]
    box_weights = [0] * card_num
    box_counts = [0] * card_num

    index = 0
    for i, redundancy in enumerate(route_expert_redundancy):
        redundancy_num = len(redundancy)
        for _ in range(redundancy_num):
            expert_id = i + expert_id_bias
            weight = next(
                (w for (item, w) in origin_weights if item == expert_id), None
            )

            if weight is None:
                raise AssertionError("Expert weight not found")
            if index >= card_num:
                raise ValueError("Index Out of Bounds")

            boxes[index].append(expert_id)
            boxes_weights[index].append(weight)
            box_weights[index] += weight
            box_counts[index] += 1
            index += 1
    return boxes, boxes_weights, box_weights, box_counts


def deploy_route_expert(origin_weights, 
                        boxes_dict: dict, 
                        card_num: int = 64, 
                        items_per_box: int = 5, 
                        remaining_items: int = 0):
    """
    放入路由专家
 
    Args:
        origin_weights (_type_): 专家权重
        boxes_dict (dict): 装箱相关信息
        card_num (int, optional): NPU数量. Defaults to 64.
        items_per_box (int, optional): 每个卡上专家数. Defaults to 5.
        remaining_items (int, optional): _description_. Defaults to 0.
 
    Returns:
        boxes, boxes_weights, box_weights, box_counts: 装箱结果
    """
    boxes = boxes_dict["boxes"]
    boxes_weights = boxes_dict["boxes_weights"]
    box_weights = boxes_dict["box_weights"]
    box_counts = boxes_dict["box_counts"]
    origin_weights = sorted(origin_weights, key=lambda x: -x[1])
    for item_id, weight in origin_weights:
        # Find the box with the least items but not full
        min_box_index = -1
        for i in range(card_num):
            # Only choose boxes that still have space (box_counts[i] < items_per_box)
            item_count_flag = box_counts[i] < items_per_box or (
                box_counts[i] == items_per_box and remaining_items > 0
            )
            box_flag = (
                min_box_index == -1 or box_weights[i] < box_weights[min_box_index]
            )
            if item_count_flag is True and box_flag is True:
                min_box_index = i
        # Place the item (id) into the selected box
        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1

        # If there's an imbalance in the remaining items, reduce the "remaining_items" counter
        if box_counts[min_box_index] == (items_per_box + 1) and remaining_items > 0:
            remaining_items -= 1

    return boxes, boxes_weights, box_weights, box_counts


def compute_node_balanced_pack_redundancy(
    origin_weights, card_num, num_redundancy_expert, expert_id_bias, is_only
):
    """
    每一层，在单个节点层面做冗余专家以及专家排布负载均衡

    :param origin_weights: 当前层、节点内的专家数
    :param card_num: 节点内卡的数量
    :param num_redundancy_expert: 当前节点内冗余专家数量
    :param expert_id_bias: 当前节点专家id偏置，即起始专家id
    :param is_only: 是否使用>2个冗余专家策略，默认冗余专家数量不做限制
    """
    # Step 1: Sort the items by weight in descending order (we are sorting by weight now)
    # Sort based on the second element (the second value of each tuple)
    route_expert_num = len(origin_weights)
    route_expert_redundancy, origin_weights = get_redundant_experts(
        origin_weights, num_redundancy_expert, expert_id_bias, route_expert_num, is_only
    )
    # Step 2: Calculate the number of items per box
    expert_num = route_expert_num + num_redundancy_expert
    items_per_box = expert_num // card_num  # Number of items per box
    remaining_items = expert_num % card_num  # Number of items per box

    # Step 3: Initialize card_num boxes with empty lists to store item IDs
    boxes, boxes_weights, box_weights, box_counts = deploy_redundant_expert(
        route_expert_redundancy, origin_weights, expert_id_bias, card_num
    )

    boxes_dict = {
        "boxes": boxes,
        "boxes_weights": boxes_weights,
        "box_weights": box_weights,
        "box_counts": box_counts,
    }
    # Step 4: Distribute items into boxes based on weight
    boxes, boxes_weights, box_weights, box_counts = deploy_route_expert(
        origin_weights=origin_weights,
        boxes_dict=boxes_dict,
        card_num=card_num,
        items_per_box=items_per_box,
        remaining_items=remaining_items)

    if any(len(row) != items_per_box for row in boxes):
        return None, None
    # Step 5: Output each box's contents and total weight
    result = []
    for i in range(card_num):
        result.append(
            {
                "box_index": i + 1,
                "items": boxes[i],  # List of item IDs in the box
                "weight": boxes_weights[i],
                "total_weight": box_weights[i],  # Total weight in this box
                "item_count": box_counts[i],  # Number of items in the box
            }
        )

    return result, boxes


def check_layer_deployment(layer_deployment, num_npus, num_nodes, expert_num):
    """
    检验单层的部署是否是在节点内的冗余部署

    :param layer_deployment: 多个list组成的list 表示当前层上每个卡的专家部署
    :param num_npus: NPU卡数量
    :param num_nodes: 节点数量
    :param expert_num: 原始专家数量
    """
    npus_per_node = num_npus // num_nodes
    experts_per_node = expert_num // num_nodes
    for i in range(num_nodes):
        cur_nodes_deployment = layer_deployment[
            i * npus_per_node: (i + 1) * npus_per_node
        ]
        merged_list = sorted(sum(cur_nodes_deployment, []))
        try:
            # 确保专家id在合理范围内
            if not (
                np.max(merged_list) < (i + 1) * experts_per_node
                and np.min(merged_list) >= i * experts_per_node
            ):
                raise ValueError("专家ID超出合理范围")

            # 确保所有专家都被使用
            expected_experts = set(
                range(i * experts_per_node, (i + 1) * experts_per_node)
            )
            if set(merged_list) != expected_experts:
                raise ValueError("实际使用的专家不等于应使用的专家")

        except Exception as e:
            raise RuntimeError(f"发生错误: {e}") from e


def get_layer_deployment(layer_workload, 
                         expert_num: int = 256, 
                         num_nodes: int = 8, 
                         origin_experts_per_node: int = 64,
                         npus_per_node: int = 8, 
                         redundant_experts_per_node: int = 8):
    """
    获取单层的部署策略
    """
    # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
    weights = np.zeros((expert_num,), dtype="object")
    for expert_id, workload_weight in enumerate(layer_workload):
        weights[expert_id] = (expert_id, workload_weight)

    # 每一层内专家按照顺序放置到不同节点内，节点内做专家冗余和卡的负载均衡放置
    tmp_layer_depolyment = []
    for node_idx in range(num_nodes):
        # 每个节点内做冗余和负载均衡放置
        node_weights = weights[
            node_idx
            * origin_experts_per_node: (node_idx + 1)
            * origin_experts_per_node
        ]
        if len(node_weights) != origin_experts_per_node:
            raise ValueError("node_weights 的长度不等于 origin_experts_per_node")
        # 节点层面做负载均衡
        try:
            _, node_deployment = compute_node_balanced_pack_redundancy(
                origin_weights=node_weights,
                card_num=npus_per_node,
                num_redundancy_expert=redundant_experts_per_node,
                expert_id_bias=node_idx * origin_experts_per_node,
                is_only=0,
            )
            if node_deployment is not None:
                tmp_layer_depolyment += node_deployment
            else:
                raise ValueError("node_deployment is None.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}") from e
    return tmp_layer_depolyment


def lb_redundancy_deploy_for_dynamic(
    layer_workloads,
    num_redundancy_expert,
    num_nodes=8,
    num_npus=64,
):
    """
    静态冗余专家放置策略，主要用于动态冗余策略中生成专家部署，满足以下约束
    1. 只能做节点内冗余，冗余专家和路由专家在同一节点（仅在动态负载均衡时有这一限制）
    2. 冗余专家数量是总的数量，每个机器上均分
    3. 机器内，每张卡上冗余专家均匀放置，不同卡上冗余专家数量相同（冗余专家数量一定是物理卡数量的整数倍）

    :param layer_workloads: np.array [layer_num, expert_num] 读取的workload数据
    :param num_nodes: 节点（机器）数量
    :param num_npus: NPU卡数量
    :param num_redundancy_expert: 总的冗余专家数量
    :param layer_workloads[layer_num, expert_num] 58*256
    :return: optimized layer_deployment: [layer_num, card_num, card_expert_num] 58*64*4
    """

    # 计算负载均衡，部署冗余专家
    layer_num, expert_num = layer_workloads.shape
    # 校验专家数量、卡数量、冗余专家数量不能超过卡数量
    if not (num_npus > 0 and num_nodes > 0 and num_redundancy_expert >= 0):
        raise ValueError("异常参数值！")

    if not (
        num_npus >= num_redundancy_expert and num_redundancy_expert % num_npus == 0
    ):
        raise ValueError("冗余专家数量不满足要求！")

    if not (num_npus % num_nodes == 0):
        raise ValueError("每个机器上卡的数量必须相同！")

    npus_per_node = num_npus // num_nodes
    if not (num_redundancy_expert % num_nodes == 0):
        raise ValueError("每个机器上冗余专家的数量必须相同")

    redundant_experts_per_node = num_redundancy_expert // num_nodes

    all_expert_num = num_redundancy_expert + expert_num
    if not (all_expert_num % num_nodes == 0):
        raise ValueError("每个节点上专家数量必须相同")
    origin_experts_per_node = expert_num // num_nodes

    # 每个卡部署的专家数量 一个冗余专家
    global_deployment = [[[] for _ in range(num_npus)] for _ in range(layer_num)]
    # 遍历获得每一层的放置策略，考虑计算均衡
    for layer in range(layer_num):
        tmp_layer_depolyment = get_layer_deployment(
            layer_workload=layer_workloads[layer],
            expert_num=expert_num,
            num_nodes=num_nodes,
            origin_experts_per_node=origin_experts_per_node,
            npus_per_node=npus_per_node,
            redundant_experts_per_node=redundant_experts_per_node,
        )
        check_layer_deployment(
            tmp_layer_depolyment, num_npus, num_nodes, expert_num
        )  # 校验生成的部署文件是否符合要求
        global_deployment[layer] = tmp_layer_depolyment

    return global_deployment
