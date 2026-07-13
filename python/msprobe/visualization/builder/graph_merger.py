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

import re
import math

from msprobe.core.common.const import Const
from msprobe.visualization.graph.graph import Graph, BaseNode
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.core.common.log import logger
from msprobe.visualization.utils import GraphConst, update_shared_dict, update_pbar_info
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.parallel_state import get_tp_pp_default_groups

MAX_INFO = "The Max value merging method for "
MIN_INFO = "The Min value merging method for "
MEAN_INFO = "The Mean value merging method for "
NORM_INFO = "The Norm value merging method for "


class GraphMerger:
    def __init__(self, build_graph_results, parallel_param, is_bench=False, pbar_info=None):
        self.strategy = self._select_strategy(build_graph_results, parallel_param, is_bench, pbar_info)

    @staticmethod
    def _select_strategy(results, param, is_bench, pbar_info):
        if param.tp == param.pp == param.rank_size == 1:
            return NoParallelMerger(results, param, is_bench, pbar_info)
        elif param.tp == param.rank_size:
            return TPMerger(results, param, is_bench, pbar_info)
        elif param.pp == param.rank_size:
            return (
                PPMerger(results, param, is_bench, pbar_info)
                if param.vpp == 1
                else VPPMerger(results, param, is_bench, pbar_info)
            )
        elif param.pp == 1:
            return TPMerger(results, param, is_bench, pbar_info)
        elif param.tp == 1:
            return (
                PPMerger(results, param, is_bench, pbar_info)
                if param.vpp == 1
                else VPPMerger(results, param, is_bench, pbar_info)
            )
        elif param.tp * param.pp == param.rank_size:
            return TPPPMerger(results, param, is_bench, pbar_info)
        else:
            return FullMerger(results, param, is_bench, pbar_info)

    def merge_graph(self):
        return self.strategy.merge_graphs()


class BaseGraphMerger:
    def __init__(self, build_graph_results, parallel_param, is_bench, pbar_info=None):
        self.unmerged_module = [Const.CLIP_GRAD, Const.OPTIMIZER]
        self.dtype_list = (
            Const.TORCH_INT_DTYPE + Const.TORCH_FLOAT_DTYPE + [Const.FLOAT16, Const.FLOAT32, Const.BFLOAT16]
        )
        self.build_graph_results = build_graph_results
        self.parallel_param = parallel_param
        self.is_bench = is_bench
        self.log_prefix = "[Bench]" if self.is_bench else "[NPU]"
        self.pbar_info = pbar_info
        self._add_all_nodes_rank()

    @staticmethod
    def sort_merged_api_collection(graph):
        def extract_rank(node):
            match = re.search(r"_Rank(\d+)", node.id)
            return int(match.group(1)) if match else None

        for sub_node in graph.root.subnodes:
            if sub_node.op == NodeOp.api_collection and sub_node.id.startswith(
                GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS
            ):
                sub_node.subnodes = sorted(sub_node.subnodes, key=extract_rank)

    @staticmethod
    def _update_node_data_key(old_id, new_id, data_dict):
        new_dict = {}
        for key, value in data_dict.items():
            new_key = key.replace(old_id, new_id)
            if "full_op_name" in value:
                value["full_op_name"] = value.get("full_op_name").replace(old_id, new_id)
            new_dict[new_key] = value
        return new_dict

    @staticmethod
    def _compare_value_same(main_value, other_value, has_uncertainty=False):
        if not isinstance(main_value, (int, float)) or not isinstance(other_value, (int, float)):
            return True
        # 没开启确定性计算，各rank的mean和norm有细微差异，如果相对误差在阈值内则认为是相同的
        if has_uncertainty:
            diff = abs(main_value - other_value)
            if math.isnan(diff):
                return math.isnan(main_value) and math.isnan(other_value)
            elif math.isinf(diff):
                return math.isinf(main_value) and math.isinf(other_value)
            else:
                return (
                    diff < GraphConst.UNCERTAINTY_THRESHOLD
                    if main_value == 0
                    else abs(diff / main_value) < GraphConst.UNCERTAINTY_THRESHOLD
                )
        else:
            return main_value == other_value

    def merge_graphs(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def merge_graph_api_collection(self, results: list):
        """
        graph合并时，将各rank的游离api集合合并为一个总的游离api集合
        example:
            rank0: Apis_Between_Modules.0                   rank1: Apis_Between_Modules.0
                   Module.module.Float16Module.forward.0           Module.module.Float16Module.forward.0
                   Apis_Between_Modules.1                          Apis_Between_Modules.1

            merged: Apis_Between_Modules_All_Ranks.0
                      |_ Apis_Between_Modules_Rank0.0
                      |_ Apis_Between_Modules_Rank1.0
                    Module.module.Float16Module.forward.0
                    Apis_Between_Modules_All_Ranks.1
                      |_ Apis_Between_Modules_Rank0.1
                      |_ Apis_Between_Modules_Rank1.1
        """
        main_graph_result = results[0]
        main_root_sub_nodes = main_graph_result.graph.root.subnodes
        new_main_root_sub_nodes = []
        for main_node in main_root_sub_nodes:
            # 如果游离api集合已合并为一个总的游离api集合，总的游离api集合之间还要再合并
            if main_node.id.startswith(GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS):
                new_main_root_sub_nodes.append(main_node)
                for other_graph_result in results[1:]:
                    other_node = other_graph_result.graph.get_node(main_node.id)
                    if not other_node:
                        continue
                    for sub_node in other_node.subnodes:
                        sub_node.upnode = main_node
                        main_graph_result.graph.node_map[sub_node.id] = sub_node
                        for sub_sub_node in sub_node.subnodes:
                            main_graph_result.graph.node_map[sub_sub_node.id] = sub_sub_node
                    main_node.subnodes.extend(other_node.subnodes)
            # 游离api集合合并为一个总的游离api集合
            elif main_node.id.startswith(GraphConst.APIS_BETWEEN_MODULES):
                all_collection_node_id = main_graph_result.graph.add_node(
                    NodeOp.api_collection,
                    GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS,
                    id_accumulation=True,
                )
                all_collection_node = main_graph_result.graph.get_node(all_collection_node_id)
                all_collection_node.upnode = main_graph_result.graph.root
                new_main_root_sub_nodes.append(all_collection_node)
                # Apis_Between_Modules.0 --> Apis_Between_Modules_Rank0.0
                origin_main_node_id = main_node.id
                main_node.id = (
                    GraphConst.APIS_BETWEEN_MODULES
                    + f"_Rank{main_graph_result.rank}."
                    + main_node.id.split(Const.SEP)[-1]
                )
                all_collection_node.subnodes = [main_node]
                main_node.upnode = all_collection_node
                main_graph_result.graph.node_map[main_node.id] = main_node
                del main_graph_result.graph.node_map[origin_main_node_id]
                for other_graph_result in results[1:]:
                    other_node = other_graph_result.graph.get_node(origin_main_node_id)
                    if not other_node:
                        continue
                    # Apis_Between_Modules.0 --> Apis_Between_Modules_Rank1.0
                    other_node.id = (
                        GraphConst.APIS_BETWEEN_MODULES
                        + f"_Rank{other_graph_result.rank}."
                        + other_node.id.split(Const.SEP)[-1]
                    )
                    main_graph_result.graph.node_map[other_node.id] = other_node
                    for sub_node in other_node.subnodes:
                        # api节点，在api名称上添加rank信息
                        old_id = sub_node.id
                        parts = sub_node.id.split(Const.SEP)
                        parts[1] += f"_rank{other_graph_result.rank}"
                        sub_node.id = Const.SEP.join(parts)
                        sub_node.input_data = self._update_node_data_key(old_id, sub_node.id, sub_node.input_data)
                        sub_node.output_data = self._update_node_data_key(old_id, sub_node.id, sub_node.output_data)
                        main_graph_result.graph.node_map[sub_node.id] = sub_node
                    all_collection_node.subnodes.append(other_node)
                    other_node.upnode = all_collection_node
            else:
                new_main_root_sub_nodes.append(main_node)
        main_graph_result.graph.root.subnodes = new_main_root_sub_nodes

    def split_graph_results_by_groups(self, groups):
        """
        基于pp或tp域，划分待合并的graph
        """
        rank_results_mapping = {result.rank: result for result in self.build_graph_results}
        return [[rank_results_mapping.get(rank) for rank in ranks] for ranks in groups]

    def compare_node_param_data(self, main_node, other_nodes, compare_data=True):
        """
        当前节点与若干其他节点比较输入输出参数的数据是否一致，如果发现有不一致的参数，将参数暂存于列表中
        :param main_node: 当前节点
        :param other_nodes: 其他节点列表
        :param compare_data: 是否进行数据比对，如果compare_data=False则直接认为数据不一致
        :return: 输入不一致的参数dict，输出不一致的参数dict，两个dict都为空列表代表两个节点的输入输出完全一致
        """
        if not other_nodes:
            return {}, {}
        data_types = {"input_data": {}, "output_data": {}}
        for data_type, data_dict in data_types.items():
            main_data_dict = getattr(main_node, data_type)
            for key, main_param in main_data_dict.items():
                same_flag = compare_data
                if main_param.get(Const.DTYPE) not in self.dtype_list:
                    continue
                tp_need_merge_params = [main_param]
                for other_node in other_nodes:
                    param_key = key.replace(main_node.id, other_node.id) if main_node.id != other_node.id else key
                    other_param = getattr(other_node, data_type).get(param_key, {})
                    if other_param.get(Const.DTYPE) not in self.dtype_list:
                        break
                    tp_need_merge_params.append(other_param)
                    if compare_data and not self.compare_param_same(main_param, other_param, has_uncertainty=True):
                        same_flag = False
                if not same_flag:
                    data_dict[key.replace(main_node.id + Const.SEP, "")] = tp_need_merge_params
        return data_types.get("input_data"), data_types.get("output_data")

    def compare_param_same(self, main_param, other_param, has_uncertainty=False):
        if not self._compare_value_same(main_param.get(Const.MAX), other_param.get(Const.MAX)):
            return False
        if not self._compare_value_same(main_param.get(Const.MIN), other_param.get(Const.MIN)):
            return False
        if not self._compare_value_same(main_param.get(Const.MEAN), other_param.get(Const.MEAN), has_uncertainty):
            return False
        if not self._compare_value_same(main_param.get(Const.NORM), other_param.get(Const.NORM), has_uncertainty):
            return False
        return True

    def get_default_groups(self):
        """
        根据GPU总数、TP数、PP数初始化并行组

        return:
        tp_groups: 张量并行组列表，每个元素是一个包含组内rank的列表
        pp_groups: 流水线并行组列表，每个元素是一个包含组内rank的列表
        """
        tp_groups, pp_groups = get_tp_pp_default_groups(
            self.parallel_param.rank_size,
            self.parallel_param.tp,
            self.parallel_param.pp,
            order=self.parallel_param.order,
        )

        return tp_groups, pp_groups

    def _add_all_nodes_rank(self):
        for result in self.build_graph_results:
            for node in result.graph.node_map.values():
                node.rank = result.rank


class PPMerger(BaseGraphMerger):
    LAYERS_PATTERN = re.compile(r"(layers\.|layer\.)\d+(\.)")
    MARK_PATTERN = re.compile(r"%(\d+)%(\d+)$")
    MARK = "%"

    @staticmethod
    def _trace_p2p_mapping(p2p_mapping: dict):
        """
        将字典分组为独立的链，每个链都从未访问过的键开始，按照字典中的映射关系进行追踪
        p2p_mapping内容为p2p通信的send映射，追踪映射关系建立pp域
        example: p2p_mapping={0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 4, 7: 5}, return=[[0, 2, 4, 6], [1, 3, 5, 7]]
        """
        visited = set()
        result = []

        def collect_keys(start_key):
            """
            追踪从某一个键开始的所有“连续”键，直到无法再找到下一个键为止
            """
            current_key = start_key
            chain = []
            while current_key in p2p_mapping and current_key not in visited:
                chain.append(current_key)
                visited.add(current_key)
                current_key = p2p_mapping[current_key]
            return chain

        for key in p2p_mapping:
            if key not in visited:
                chain_result = collect_keys(key)
                if chain_result:
                    result.append(chain_result)
        return result

    @recursion_depth_decorator("msprobe.visualization.builder.graph_merger.PPMerger._merge_nodes", 1000)
    def _merge_nodes(self, main_graph, main_node, other_graphs):
        """
        其他rank graph中被pp切分的节点，需要合并到main graph
        """
        other_nodes = []
        for other_graph in other_graphs:
            other_node = other_graph.get_node(main_node.id)
            # 表明此节点只有main graph有
            if not other_node:
                other_nodes.clear()
                return
            other_nodes.append(other_node)
        if other_nodes:
            param_in, param_out = self.compare_node_param_data(main_node, other_nodes)
            # 各个rank都有的模块，且输入输出都不一致，且节点id符合正则，判定为被pp切分的模块，需要合并结构
            pp_merged_condition = param_in and param_out and self.LAYERS_PATTERN.search(main_node.id)
            # backward可能没有output，是否要pp合并从对应的forward节点判断
            if Const.SEP + Const.BACKWARD + Const.SEP in main_node.id:
                f_node = main_graph.node_map.get(
                    main_node.id.replace(
                        Const.SEP + Const.BACKWARD + Const.SEP,
                        Const.SEP + Const.FORWARD + Const.SEP,
                    )
                )
                if f_node and hasattr(f_node, "is_pp_merged"):
                    pp_merged_condition = True
            if pp_merged_condition:
                main_node.is_pp_merged = True
                main_up_node = main_node.upnode
                for other_node in other_nodes:
                    # pp切分中被切分的层在各rank的名称是一样的，这里给其他rank的同名层增加位置和rank标记
                    self._mark_node_id_position_rank(other_node, other_node.rank)
                    self._add_node_to_main_graph(main_graph, other_node)
                    # 其他rank被pp切分的模块节点添加到当前rank的graph
                    other_node.upnode = main_up_node
                    main_up_node.subnodes.append(other_node)
                # 已找到被pp切分的模块节点，不再递归其内部
                return
            # 各个rank都有的forward模块，且输入一致，输出不一致，判定为模块内部包含被pp切分的模块，此模块的输出要使用最后一个rank的输出
            elif not param_in and param_out and Const.SEP + Const.FORWARD + Const.SEP in main_node.id:
                main_node.output_data = other_nodes[-1].output_data
            # 各个rank都有的backward模块，且输出一致，输入不一致，判定为模块内部包含被pp切分的模块，此模块的输入要使用最后一个rank的输入
            elif param_in and not param_out and Const.SEP + Const.BACKWARD + Const.SEP in main_node.id:
                main_node.input_data = other_nodes[-1].input_data
            self._merge_other_unique_nodes(main_graph, main_node, other_nodes)
        for sub_node in main_node.subnodes:
            if sub_node.op == NodeOp.module:
                self._merge_nodes(main_graph, sub_node, other_graphs)

    def merge_graphs(self):
        results_groups = self.split_graph_results_by_groups(self.get_groups())
        results = []
        for result_groups in results_groups:
            self.merge_graph_api_collection(result_groups)
            results.extend(self.merge_pp_graphs(result_groups))
        return results

    def merge_pp_graphs(self, results):
        if self.pbar_info:
            update_shared_dict(
                self.pbar_info.current_stage_dict,
                self.pbar_info.task_id,
                1,
                update_all=True,
            )
        if not results or len(results) < 2:
            return results
        graphs = [x.graph for x in results]
        main_graph_result = results[0]
        total = len(main_graph_result.graph.root.subnodes)
        for i, main_node in enumerate(main_graph_result.graph.root.subnodes):
            if main_node.op == NodeOp.module and main_node.id not in self.unmerged_module:
                self._merge_nodes(main_graph_result.graph, main_node, graphs[1:])
                self._sort_nodes(main_graph_result.graph, main_node)
            if self.pbar_info:
                update_pbar_info(self.pbar_info, i + 1, total, update_all=True)
        self._merge_root_unique_subnodes(main_graph_result, graphs)
        return [main_graph_result]

    def _merge_root_unique_subnodes(self, main_graph_result, graphs):
        """合并其他rank根节点下独有的子节点到主graph

        场景：layers模块直接挂在根节点下，各PP stage各有不同的layers，
        导致各stage的独有子节点（如layers.14-layers.27）需要合并到主graph。
        """
        if len(graphs) < 2:
            return
        main_root = main_graph_result.graph.root
        # 守卫条件：只有根节点下直接包含layer模块时（而非包在decoder等wrapper模块内），
        # 才需要执行根级别的独有节点合并。Megatron等场景的layers在decoder wrapper下，
        # 由_merge_nodes递归处理。
        if not any(self.LAYERS_PATTERN.search(node.id) for node in main_root.subnodes):
            return
        main_subnode_ids = {n.id for n in main_root.subnodes}
        nodes_to_merge = []
        for other_root in [graph.root for graph in graphs[1:]]:
            for node in other_root.subnodes:
                if node.id not in main_subnode_ids and not node.id.startswith(GraphConst.APIS_BETWEEN_MODULES):
                    nodes_to_merge.append(node)
        if not nodes_to_merge:
            return
        for node in nodes_to_merge:
            self._mark_node_id_position_rank(node, node.rank)
            self._add_node_to_main_graph(main_graph_result.graph, node)
            node.upnode = main_root
            main_root.subnodes.append(node)
        # 将新增节点排序在已有节点之后，按标记中的position排序
        unmarked = [n for n in main_root.subnodes if not self.MARK_PATTERN.search(n.id)]
        marked = [n for n in main_root.subnodes if self.MARK_PATTERN.search(n.id)]
        marked.sort(
            key=lambda n: (int(self.MARK_PATTERN.search(n.id).group(2)), int(self.MARK_PATTERN.search(n.id).group(1)))
        )
        main_root.subnodes = unmarked + marked
        # 计算已有节点的最大layer index，用于给新增节点重编号
        layer_idx = -1
        for node in unmarked:
            if self.LAYERS_PATTERN.search(node.id):
                layer_idx += 1
        for node in marked:
            if self.LAYERS_PATTERN.search(node.id):
                layer_idx += 1
                node.pp_index = layer_idx
            self._update_node_id(main_graph_result.graph, node)

    def get_groups(self):
        """
        在各rank寻找p2p通信节点，建立各rank之间p2p的映射关系
        """
        p2p_mapping = {}
        for result in self.build_graph_results:
            rank = result.rank
            pp_rank = None
            for node in result.graph.node_map.values():
                if not node.id.startswith(Const.DISTRIBUTED + Const.SEP):
                    continue
                if ".batch_isend_irecv." in node.id:
                    for p2p_info in node.batch_p2p_info:
                        target_rank = p2p_info.get(GraphConst.PEER)
                        if target_rank is not None and target_rank != rank and p2p_info.get(GraphConst.OP) == "isend":
                            pp_rank = target_rank
                            break
                elif ".send." in node.id or ".isend." in node.id:
                    # example: Distributed.isend.0.forward --> Distributed.isend.0.forward.input.dst
                    dst_kwarg = f"{node.id}{Const.SEP}{Const.INPUT}{Const.SEP}{GraphConst.DST}"
                    dst = node.input_data.get(dst_kwarg, {}).get("value")
                    if dst is not None:
                        pp_rank = dst
                        break
                if pp_rank is not None:
                    break
            if pp_rank is not None:
                p2p_mapping[rank] = pp_rank
        pp_groups = self._trace_p2p_mapping(p2p_mapping)
        if not pp_groups:
            logger.info(
                "Unable to get pp groups based on Distributed Api (batch_isend_irecv, send, or isend), "
                'generate pp groups using parallel param "rank_size", "tp" and "pp".'
            )
            _, pp_groups = self.get_default_groups()
        elif len(pp_groups[0]) != self.parallel_param.pp:
            logger.warning(
                f"Based on Distributed Api (atch_isend_irecv, send, or isend), "
                f"the resulting pp groups={pp_groups}, "
                f'its length is not equal to the parallel param "pp"({self.parallel_param.pp}) you defined, '
                f'generate pp groups using parallel param "rank_size", "tp" and "pp".'
            )
            _, pp_groups = self.get_default_groups()
        logger.info(f"{self.log_prefix} All pp groups is {pp_groups}.")
        return pp_groups

    def _merge_other_unique_nodes(self, main_graph, main_node, other_nodes):
        """
        其他rank graph中other_node的子节点列表如果包含独有的节点，需要合并到main graph
        """
        lists = [main_node.subnodes]
        for other_node in other_nodes:
            lists.append(other_node.subnodes)
        dicts = [{node.id: node for node in lst} for lst in lists]
        unique_node_ids = {}
        # 计算每个集合的独有元素
        for i, current_dict in enumerate(dicts):
            other_ids = set()
            for j, other_dict in enumerate(dicts):
                if i != j:
                    # 更新并集，添加当前遍历到的集合的元素
                    other_ids.update(other_dict.keys())
            result = set(current_dict.keys()) - other_ids
            if i != 0 and result:
                # 计算当前集合与其他集合并集的差集，即独有元素，保持原始顺序
                unique_node_ids[i] = [node_id for node_id in current_dict if node_id in result]
        unique_nodes = []
        if unique_node_ids:
            for i, items in unique_node_ids.items():
                for item in items:
                    unique_nodes.append(dicts[i].get(item))
        if unique_nodes:
            for unique_node in unique_nodes:
                self._mark_node_id_position_rank(unique_node, unique_node.rank)
                self._add_node_to_main_graph(main_graph, unique_node)
                main_node.subnodes.append(unique_node)
                unique_node.upnode = main_node

    def _sort_nodes(self, main_graph, start_node):
        stack = [start_node]
        while stack:
            node = stack.pop()
            if self.MARK_PATTERN.search(node.id):
                is_forward = (
                    Const.SEP + Const.FORWARD + Const.SEP in node.id or Const.SEP + Const.FORWARD + self.MARK in node.id
                )
                new_sub_nodes1, new_sub_nodes2 = [], []
                for item in node.upnode.subnodes:
                    if self.MARK_PATTERN.search(item.id):
                        new_sub_nodes2.append(item)
                    else:
                        new_sub_nodes1.append(item)

                order = bool(is_forward)
                new_sub_nodes2.sort(key=lambda n: self._get_node_sort_rule(n, rank_ascending=order))
                new_sub_nodes = new_sub_nodes1 + new_sub_nodes2 if is_forward else new_sub_nodes2 + new_sub_nodes1

                index = -1
                node_iter = new_sub_nodes if is_forward else reversed(new_sub_nodes)
                for item in node_iter:
                    if self.LAYERS_PATTERN.search(item.id):
                        index += 1
                    if self.MARK_PATTERN.search(item.id):
                        item.pp_index = index

                for item in new_sub_nodes2:
                    self._update_node_id(main_graph, item)

                node.upnode.subnodes = new_sub_nodes

            stack.extend(node.subnodes)

    def _add_node_to_main_graph(self, main_graph: Graph, node: BaseNode):
        if node.id in main_graph.node_map:
            logger.warning(f"{node.id} is exist!")
        else:
            main_graph.node_map[node.id] = node
        for sub_node in node.subnodes:
            self._add_node_to_main_graph(main_graph, sub_node)

    def _get_node_sort_rule(self, node, rank_ascending=True):
        match = self.MARK_PATTERN.search(node.id)
        if match:
            # position代表当前节点在父节点中的位置序号
            position, rank = int(match.group(1)), int(match.group(2))
            if rank_ascending:
                return rank, position
            else:
                return -rank, position
        return (float("inf"), float("inf")) if rank_ascending else (-float("inf"), -float("inf"))

    def _mark_node_id_position_rank(self, node: BaseNode, rank):
        position = 0
        for index, item in enumerate(node.upnode.subnodes):
            if item.id == node.id:
                position = index
                break
        # 各rank重复节点添加所处层级位置排序信息position和rank号，用%分隔
        node.id = node.id + f"{self.MARK}{position}" + f"{self.MARK}{rank}"
        for sub_node in node.subnodes:
            self._mark_node_id_position_rank(sub_node, rank)

    def _update_node_id(self, graph, start_node: BaseNode, pp_index=""):
        stack = [(start_node, pp_index)]
        while stack:
            node, pp_index = stack.pop()
            # 修改节点id之前删除node_map的信息，修改完再添加回去
            if node.id not in graph.node_map:
                logger.warning(f"Update node id {node.id} fail!")
            else:
                del graph.node_map[node.id]
                old_id = self.MARK_PATTERN.sub("", node.id)
                if node.op == NodeOp.module:
                    # 被pp切分的模块节点，基于位置和rank信息修改模块名称计数信息
                    if self.LAYERS_PATTERN.search(node.id) and self.MARK_PATTERN.search(node.id):
                        if hasattr(node, "pp_index"):
                            pp_index = str(node.pp_index)
                        node.id = self.LAYERS_PATTERN.sub(r"\g<1>" + pp_index + r"\g<2>", node.id)
                else:
                    # api节点，在api名称上添加rank信息
                    parts = node.id.split(Const.SEP)
                    parts[1] += f"_rank{node.id.split(PPMerger.MARK)[-1]}"
                    node.id = Const.SEP.join(parts)
                # 把之前添加的位置和rank信息删掉
                node.id = self.MARK_PATTERN.sub("", node.id)
                # node id更新了，那么data的key中包含node id也要更新
                node.input_data = self._update_node_data_key(old_id, node.id, node.input_data)
                node.output_data = self._update_node_data_key(old_id, node.id, node.output_data)
                graph.node_map[node.id] = node
            # 将子节点加入栈中
            for sub_node in node.subnodes:
                stack.append((sub_node, pp_index))


class TPMerger(BaseGraphMerger):
    RANK_PATTERN = re.compile(r"_rank(\d+)\.")
    OPERATION_TABLE = {
        Const.MAX: {
            "initial": lambda p: p.get(Const.MAX),
            "merge": lambda current, other: max(current, other.get(Const.MAX)),
            "finalize": lambda current, count: current,
            "formula": lambda key, values: (f"{MAX_INFO}{key} is: max({', '.join(map(str, values))})"),
        },
        Const.MIN: {
            "initial": lambda p: p.get(Const.MIN),
            "merge": lambda current, other: min(current, other.get(Const.MIN)),
            "finalize": lambda current, count: current,
            "formula": lambda key, values: (f"{MIN_INFO}{key} is: min({', '.join(map(str, values))})"),
        },
        Const.MEAN: {
            "initial": lambda p: p.get(Const.MEAN),
            "merge": lambda current, other: current + other.get(Const.MEAN),
            "finalize": lambda current, count: current / count,
            "formula": lambda key, values: (f"{MEAN_INFO}{key} is: ({' + '.join(map(str, values))}) / {len(values)}"),
        },
        Const.NORM: {
            "initial": lambda p: pow(p.get(Const.NORM), 2.0),
            "merge": lambda current, other: current + pow(other.get(Const.NORM), 2.0),
            "finalize": lambda current, count: pow(current, 1 / 2.0),
            "formula": lambda key, values: (
                f"{NORM_INFO}{key} is: ({' + '.join([f'{v} ** 2' for v in values])}) ** 0.5"
            ),
        },
    }
    TP_MERGED_INFO = "This data is the merged data after tensor parallelism(TP), and the data is merged from rank "

    @staticmethod
    def _merge_params(tp_need_merge_param: dict):
        """
        合并tp切分的各rank参数统计值
        tp_need_merge_param: {input.0: [{"Max": 0, "Min": 0, ...}, {"Max": 0.1, "Min": 0, ...}, ...]}
        return: 计算详情
        """
        merge_info = []
        for key, param_list in tp_need_merge_param.items():
            if len(param_list) < 2:
                continue
            main_param = param_list[0]

            for stat, ops in TPMerger.OPERATION_TABLE.items():
                # 检查所有参数在当前统计量上是否有有效的数值，若存在None等非数值则跳过该统计量的合并
                if any(not isinstance(p.get(stat), (int, float)) for p in param_list):
                    continue

                current_value = ops["initial"](main_param)
                value_list = [current_value if stat != Const.NORM else main_param.get(Const.NORM)]

                for other_param in param_list[1:]:
                    current_value = ops["merge"](current_value, other_param)
                    value_list.append(other_param.get(stat) if stat != Const.NORM else other_param.get(Const.NORM))

                final_value = ops["finalize"](current_value, len(param_list))
                main_param[stat] = final_value
                formula_base = f"{ops['formula'](key, value_list)}" + f" = {final_value}"

                merge_info.append(formula_base)

        return merge_info

    @staticmethod
    def _get_need_merge_node(main_node, other_graphs, tp_merge_mapping):
        """
        获取需要TP合并的节点列表
        如果是TP+PP的混合并行，此时数据已经被PP合并过，一些node_id被标记上rank信息，此时需要基于rank映射才能获取到需要TP合并的节点列表，例如：
        main_node = Torch.matmul_rank4.32.forward  other_node = Torch.matmul_rank5.32.forward
        需要建立4->5的映射，才能基于Torch.matmul_rank4.32.forward找到Torch.matmul_rank5.32.forward
        """
        other_nodes = []
        match = TPMerger.RANK_PATTERN.search(main_node.id)
        # 节点名称被标记rank信息，且提供了映射
        if match and tp_merge_mapping:
            rank = int(match.group(1))
            tp_mapping_ranks = tp_merge_mapping.get(rank)
            if not tp_mapping_ranks:
                return other_nodes
            if len(tp_mapping_ranks) != len(other_graphs):
                return other_nodes
            for i, graph in enumerate(other_graphs):
                # 基于映射得到目标rank，替换node_id当前rank信息后去目标graph取node
                tp_mapping_id = TPMerger.RANK_PATTERN.sub(f"_rank{tp_mapping_ranks[i]}.", main_node.id)
                other_node = graph.node_map.get(tp_mapping_id)
                if not other_node or main_node.get_ancestors() != other_node.get_ancestors():
                    other_nodes.clear()
                    break
                other_nodes.append(other_node)
        else:
            for graph in other_graphs:
                other_node = graph.node_map.get(main_node.id)
                if not other_node or main_node.get_ancestors() != other_node.get_ancestors():
                    other_nodes.clear()
                    break
                other_nodes.append(other_node)

        return other_nodes

    @staticmethod
    def _slice_list_at_id(node_list, target_id1, target_id2):
        start_index, end_index = -1, -1
        for index, node in enumerate(node_list):
            if target_id1 in node.id:
                start_index = index
            elif target_id2 in node.id:
                end_index = index
        return [] if start_index == -1 or end_index == -1 else node_list[start_index : end_index + 1]

    def merge_graphs(self):
        results_groups = self.split_graph_results_by_groups(self.get_groups())
        results = []
        for result_groups in results_groups:
            self.merge_graph_api_collection(result_groups)
            results.extend(self.merge_tp_graphs(result_groups))
        return results

    def merge_tp_graphs(self, results, tp_merge_mapping=None):
        if self.pbar_info:
            update_shared_dict(
                self.pbar_info.current_stage_dict,
                self.pbar_info.task_id,
                1,
                update_all=True,
            )
        if not results or len(results) < 2:
            return results
        graphs = [x.graph for x in results]
        main_graph_result = results[0]
        total = len(main_graph_result.graph.node_map.values())
        for i, main_node in enumerate(main_graph_result.graph.node_map.values()):
            should_continue = (
                not main_node.upnode
                or main_node.upnode.op != NodeOp.module
                or main_node.upnode.id in self.unmerged_module
                or main_node.id.startswith(Const.DISTRIBUTED)
                or main_node.parallel_merge_info != []
            )
            if should_continue:
                continue
            self._handle_tp_matmul_reduce(main_node, graphs[1:], tp_merge_mapping)
            other_nodes = self._get_need_merge_node(main_node, graphs[1:], tp_merge_mapping)
            tp_need_merge_param_in, tp_need_merge_param_out = self.compare_node_param_data(main_node, other_nodes)
            if tp_need_merge_param_in or tp_need_merge_param_out:
                ranks = [main_node.rank]
                for other_node in other_nodes:
                    ranks.append(other_node.rank)
                main_node.parallel_merge_info.append(f"{self.TP_MERGED_INFO}{ranks}.")
                merge_info_in = self._merge_params(tp_need_merge_param_in)
                merge_info_out = self._merge_params(tp_need_merge_param_out)
                main_node.parallel_merge_info.extend(merge_info_in + merge_info_out)
            if self.pbar_info:
                update_pbar_info(self.pbar_info, i + 1, total, update_all=True)
        for main_node in main_graph_result.graph.node_map.values():
            self._merge_tp_megatron_column_row_parallel(main_node, graphs[1:], tp_merge_mapping)
        return [main_graph_result]

    def get_groups(self):
        tp_groups = []
        for result in self.build_graph_results:
            for node in result.graph.node_map.values():
                if any(op in node.id for op in GraphConst.REDUCE_OPERATIONS):
                    group_ranks = node.input_data.get(f"{node.id}.input.group", {}).get("group_ranks")
                    if group_ranks and group_ranks not in tp_groups:
                        tp_groups.append(group_ranks)
                    break
        if not tp_groups:
            logger.info(
                "Unable to get tp groups based on Distributed Api (reduce_scatter or all_reduce), "
                'generate tp groups using parallel param "rank_size", "tp" and "pp".'
            )
            tp_groups, _ = self.get_default_groups()
        elif len(tp_groups[0]) != self.parallel_param.tp:
            logger.warning(
                f"Based on Distributed Api (reduce_scatter or all_reduce), "
                f"the resulting tp groups={tp_groups}, "
                f'its length is not equal to the parallel param "tp"({self.parallel_param.tp}) you defined, '
                f'generate tp groups using parallel param "rank_size", "tp" and "pp".'
            )
            tp_groups, _ = self.get_default_groups()
        logger.info(f"{self.log_prefix} All tp groups is {tp_groups}.")
        return tp_groups

    def _handle_tp_matmul_reduce(self, node, other_graphs, tp_merge_mapping):
        """
        前向RowParallel和反向ColumnParallel层的matmul输出需要替换成matmul计算完成后all_reduce/reduce_scatter的输出
        """
        if node.op != NodeOp.module:
            return
        splits = node.id.split(Const.SEP)
        if len(splits) < 4:
            return
        is_forward_with_row_parallel = splits[-2] == Const.FORWARD and "RowParallelLinear" in splits[-3]
        is_backward_with_column_parallel = splits[-2] == Const.BACKWARD and "ColumnParallelLinear" in splits[-3]
        if not is_forward_with_row_parallel and not is_backward_with_column_parallel:
            return
        matmul_list = []
        reduce_list = []
        for sub_node in node.subnodes:
            if "matmul" in sub_node.id:
                matmul_list.append(sub_node)
            if (
                "_reduce_scatter_base" in sub_node.id
                or "reduce_scatter_tensor" in sub_node.id
                or "all_reduce" in sub_node.id
            ):
                reduce_list.append(sub_node)
        if not matmul_list or not reduce_list:
            return
        for matmul_node in matmul_list:
            if not matmul_node.output_data:
                continue
            # matmul的output0，将传递给all_reduce/reduce_scatter，作为all_reduce的input0，或作为reduce_scatter的input1
            matmul_node_output_param = list(matmul_node.output_data.values())[0]
            for reduce_node in list(reduce_list):
                if not reduce_node.output_data:
                    continue
                if "all_reduce" in reduce_node.id:
                    if not reduce_node.input_data:
                        continue
                    reduce_node_input_param = list(reduce_node.input_data.values())[0]
                else:
                    if len(reduce_node.input_data) < 2:
                        continue
                    reduce_node_input_param = list(reduce_node.input_data.values())[1]
                if not self.compare_param_same(matmul_node_output_param, reduce_node_input_param):
                    continue
                # matmul的input统计值与其他rank的数据进行合并
                other_nodes = self._get_need_merge_node(matmul_node, other_graphs, tp_merge_mapping)
                tp_need_merge_param_in, _ = self.compare_node_param_data(matmul_node, other_nodes)
                if tp_need_merge_param_in:
                    ranks = [matmul_node.rank]
                    for other_node in other_nodes:
                        ranks.append(other_node.rank)
                    matmul_node.parallel_merge_info.append(f"{self.TP_MERGED_INFO}{ranks}.")
                    merge_info_in = self._merge_params(tp_need_merge_param_in)
                    matmul_node.parallel_merge_info.extend(merge_info_in)
                # matmul的output0替换为all_reduce/reduce_scatter的output0
                reduce_node_output_param = list(reduce_node.output_data.values())[0]
                keys = [Const.MAX, Const.MIN, Const.MEAN, Const.NORM]
                matmul_node_output_param.update({k: reduce_node_output_param.get(k) for k in keys})
                full_op_name = reduce_node_output_param.get("full_op_name")
                param_name = full_op_name if full_op_name else reduce_node.id
                matmul_node.parallel_merge_info.append(f"The output of this data is merged from {param_name}")
                reduce_list.remove(reduce_node)
                break

    def _merge_tp_megatron_column_row_parallel(self, node, other_graphs, tp_merge_mapping):
        if node.op != NodeOp.module or node.parallel_merge_info:
            return
        splits = node.id.split(Const.SEP)
        if len(splits) < 4:
            return
        is_forward_with_column_parallel = splits[-2] == Const.FORWARD and "ColumnParallelLinear" in splits[-3]
        if not is_forward_with_column_parallel:
            return
        if not node.upnode:
            return
        # 获取[ColumnParallelLinear, RowParallelLinear]结构
        nodes = self._slice_list_at_id(node.upnode.subnodes, node.id, "RowParallelLinear")
        if len(nodes) < 2:
            return
        stack = nodes[:]
        while stack:
            current_node = stack.pop()
            stack.extend(reversed(current_node.subnodes))

            if current_node.parallel_merge_info or current_node.id.startswith(Const.DISTRIBUTED):
                continue

            other_nodes = self._get_need_merge_node(current_node, other_graphs, tp_merge_mapping)
            param_in, param_out = self.compare_node_param_data(current_node, other_nodes, False)

            if param_in or param_out:
                ranks = [current_node.rank]
                for other_node in other_nodes:
                    ranks.append(other_node.rank)
                current_node.parallel_merge_info.append(f"{self.TP_MERGED_INFO}{ranks}.")
                # ColumnParallelLinear层的输入、其中的matmul输入不需要合并
                if current_node == nodes[0] or ("matmul" in current_node.id and current_node.upnode == nodes[0]):
                    param_in.pop("input.0", None)
                # RowParallelLinear层的输出、其中的matmul输出不需要合并, bias不需要合并
                elif current_node == nodes[-1] or ("matmul" in current_node.id and current_node.upnode == nodes[-1]):
                    param_out = {}
                    param_in.pop("parameters.bias", None)

                merge_info_in = self._merge_params(param_in)
                merge_info_out = self._merge_params(param_out)
                current_node.parallel_merge_info.extend(merge_info_in + merge_info_out)


class NoParallelMerger(BaseGraphMerger):
    def merge_graphs(self):
        self.merge_graph_api_collection(self.build_graph_results)
        return self.build_graph_results


class TPPPMerger(BaseGraphMerger):
    def merge_graphs(self):
        tp_merger = TPMerger(self.build_graph_results, self.parallel_param, self.is_bench, self.pbar_info)
        pp_merger = (
            PPMerger(
                self.build_graph_results,
                self.parallel_param,
                self.is_bench,
                self.pbar_info,
            )
            if self.parallel_param.vpp == 1
            else VPPMerger(
                self.build_graph_results,
                self.parallel_param,
                self.is_bench,
                self.pbar_info,
            )
        )
        pp_groups = pp_merger.get_groups()
        tp_groups = tp_merger.get_groups()
        # 进入TP+PP混合处理器，PP和TP必然大于1
        tp_merge_mapping = {}
        for tp_group in tp_groups[1:]:
            tp_merge_mapping[tp_group[0]] = tp_group[1:]
        self.merge_graph_api_collection(self.build_graph_results)
        # 先合并pp，需要知道pp域，在各自pp域中合并
        results_groups_pp = self.split_graph_results_by_groups(pp_groups)
        pp_results = []
        for results in results_groups_pp:
            pp_results.extend(pp_merger.merge_pp_graphs(results))
        # pp合并完成后，直接进行tp合并，最终得到一个graph
        tp_result = tp_merger.merge_tp_graphs(pp_results, tp_merge_mapping)
        self.sort_merged_api_collection(tp_result[0].graph)
        return tp_result


class FullMerger(BaseGraphMerger):
    def merge_graphs(self):
        tp_merger = TPMerger(self.build_graph_results, self.parallel_param, self.is_bench, self.pbar_info)
        pp_merger = (
            PPMerger(
                self.build_graph_results,
                self.parallel_param,
                self.is_bench,
                self.pbar_info,
            )
            if self.parallel_param.vpp == 1
            else VPPMerger(
                self.build_graph_results,
                self.parallel_param,
                self.is_bench,
                self.pbar_info,
            )
        )
        pp_groups = pp_merger.get_groups()
        tp_groups = tp_merger.get_groups()
        tp_merge_mapping = {}
        if len(tp_groups) < 1:
            raise RuntimeError(f"Graph merged error, and tp_groups is {tp_groups}.")
        for tp_group in tp_groups[1:]:
            if len(tp_group) < 1:
                raise RuntimeError(f"Graph merged error, and tp_group is {tp_group}.")
            tp_merge_mapping[tp_group[0]] = tp_group[1:]
        # 先合并pp，需要知道pp域，在各自pp域中合并
        results_groups_pp = self.split_graph_results_by_groups(pp_groups)
        pp_results = {}
        for pp_result in results_groups_pp:
            self.merge_graph_api_collection(pp_result)
            pp_result = pp_merger.merge_pp_graphs(pp_result)[0]
            pp_results[pp_result.rank] = pp_result
        # pp合并完成后，基于tp域划分pp合并结果
        lists_to_be_tp_merged = []
        for tp_group in tp_groups:
            list_to_be_tp_merged = []
            for rank in tp_group:
                pp_result = pp_results.get(rank)
                if pp_result:
                    list_to_be_tp_merged.append(pp_result)
            if list_to_be_tp_merged:
                lists_to_be_tp_merged.append(list_to_be_tp_merged)
        tp_results = []
        for list_to_be_tp_merged in lists_to_be_tp_merged:
            self.merge_graph_api_collection(list_to_be_tp_merged)
            tp_merged_result = tp_merger.merge_tp_graphs(list_to_be_tp_merged, tp_merge_mapping)
            self.sort_merged_api_collection(tp_merged_result[0].graph)
            tp_results.extend(tp_merged_result)
        return tp_results


class VPPMerger(PPMerger):
    LAYERS_NUM_PATTERN = re.compile(r"(layers\.|layer\.)(\d+)(\.)")
    FORWARD_PATTERN = re.compile(r"\.forward\.\d+$")

    @staticmethod
    def _replace_vpp_id(s, vpp_id):
        parts = s.split(Const.SEP)
        if len(parts) < 2 or not parts[1].isdigit():
            return s
        parts[1] = str(vpp_id)
        return Const.SEP.join(parts)

    def merge_pp_graphs(self, results):
        if self.pbar_info:
            update_shared_dict(
                self.pbar_info.current_stage_dict,
                self.pbar_info.task_id,
                1,
                update_all=True,
            )
        if not results or len(results) < 2:
            return results
        graphs = [x.graph for x in results]
        main_graph_result = results[0]
        total = len(main_graph_result.graph.root.subnodes)
        for i, main_node in enumerate(main_graph_result.graph.root.subnodes):
            if main_node.op == NodeOp.module and main_node.id not in self.unmerged_module:
                self._merge_nodes(main_graph_result.graph, main_node, graphs[1:])
                self._sort_nodes(main_graph_result.graph, main_node)
            if self.pbar_info:
                update_pbar_info(self.pbar_info, i + 1, total, update_all=True)
        self._merge_root_unique_subnodes(main_graph_result, graphs)
        self._merge_vpp_data(main_graph_result.graph)
        self._merge_vpp_chunks(main_graph_result.graph)
        return [main_graph_result]

    def _merge_vpp_data(self, graph):
        """
        所有chunk的数据都合并到chunk0，前向chunk0的输出使用最后一个chunk的输出，反向chunk0的输入使用最后一个chunk的输入
        """
        module_list = []
        for node in reversed(graph.root.subnodes):
            parts = node.id.split(Const.SEP)
            if len(parts) < 2:
                continue
            if parts[1] in [GraphConst.VPP_CHUNK_0, str(self.parallel_param.vpp - 1)]:
                module_list.append(node)
        if not module_list:
            return
        stack = module_list[:]
        while stack:
            current_node = stack.pop()
            if (
                hasattr(current_node, "is_pp_merged")
                or hasattr(current_node, "pp_index")
                or current_node.op != NodeOp.module
            ):
                continue
            is_forward = self.FORWARD_PATTERN.search(current_node.id)
            stack.extend(reversed(current_node.subnodes))
            target_id = self._replace_vpp_id(current_node.id, self.parallel_param.vpp - 1)
            target_node = graph.node_map.get(target_id)
            if not target_node:
                continue
            if is_forward:
                current_node.output_data = self._update_node_data_key(
                    target_node.id, current_node.id, target_node.output_data
                )
            else:
                current_node.input_data = self._update_node_data_key(
                    target_node.id, current_node.id, target_node.input_data
                )

    def _merge_vpp_chunks(self, graph):
        """
        所有chunk都合并到chunk0，layers层搬到chunk0并重排序号
        """
        chunk_id_list = list(range(1, self.parallel_param.vpp))
        chunk_0_list = []
        for node in reversed(graph.root.subnodes):
            parts = node.id.split(Const.SEP)
            if len(parts) < 2:
                continue
            if parts[1] == GraphConst.VPP_CHUNK_0:
                chunk_0_list.append(node)
        if not chunk_0_list:
            return
        stack = chunk_0_list[:]
        layers_need_merge_dict = {}
        while stack:
            current_node = stack.pop()
            if (
                hasattr(current_node, "is_pp_merged")
                or hasattr(current_node, "pp_index")
                and current_node.upnode.id not in layers_need_merge_dict
            ):
                layers_need_merge_dict[current_node.upnode.id] = current_node.upnode
                continue
            stack.extend(reversed(current_node.subnodes))
        for node in layers_need_merge_dict.values():
            is_forward = self.FORWARD_PATTERN.search(node.id)
            for vpp_id in chunk_id_list:
                target_node = graph.node_map.get(self._replace_vpp_id(node.id, vpp_id))
                if not target_node:
                    continue
                # 其他chunk的layers都搬到chunk0，forward追加到后面，backward追加到前面
                if is_forward:
                    node.subnodes.extend(target_node.subnodes)
                else:
                    node.subnodes = target_node.subnodes + node.subnodes
                for sub_node in target_node.subnodes:
                    sub_node.upnode = node
                # 获取其他chunk的层级链路，删除所有父节点，不在前端展示已合并的其他chunk节点
                ancestors = target_node.get_ancestors()
                if len(ancestors) < 2:
                    continue
                for module_id in ancestors[1:]:
                    graph.node_map.pop(module_id, None)
                graph.root.subnodes = [node for node in graph.root.subnodes if node.id != ancestors[1]]
            # layers层重排序号
            self._sort_layers(node.subnodes, graph, is_forward)

    def _sort_layers(self, node_list, graph, is_forward):
        if not is_forward:
            node_list = list(reversed(node_list))
        index = -1
        for node in node_list:
            match = self.LAYERS_NUM_PATTERN.search(node.id)
            if match:
                index += 1
            parts = node.id.split(Const.SEP)
            # Module.0.xxx代表第一个chunk，不必重排序
            if len(parts) < 2 or parts[1] == GraphConst.VPP_CHUNK_0:
                continue
            # layers层修改chunk号和layers序号，非layers层修改chunk号
            new_node_id_prefix = ""
            if match:
                prefix, number, dot = match.groups()
                new_string = prefix + str(index) + dot
                start, end = match.span()
                new_node_id_prefix = node.id[:start] + new_string
                new_node_id_prefix = self._replace_vpp_id(new_node_id_prefix, GraphConst.VPP_CHUNK_0)
                new_node_id = new_node_id_prefix + node.id[end:]
            else:
                new_node_id = self._replace_vpp_id(node.id, GraphConst.VPP_CHUNK_0)
            graph.node_map.pop(node.id, None)
            node.input_data = self._update_node_data_key(node.id, new_node_id, node.input_data)
            node.output_data = self._update_node_data_key(node.id, new_node_id, node.output_data)
            node.id = new_node_id
            graph.node_map[new_node_id] = node
            stack = node.subnodes[:]
            while stack:
                current_node = stack.pop()
                if current_node.op != NodeOp.module:
                    continue
                stack.extend(reversed(current_node.subnodes))
                match = self.LAYERS_NUM_PATTERN.search(current_node.id)
                if match:
                    _, e = match.span()
                    new_current_node_id = new_node_id_prefix + current_node.id[e:]
                else:
                    new_current_node_id = self._replace_vpp_id(current_node.id, GraphConst.VPP_CHUNK_0)
                current_node.input_data = self._update_node_data_key(
                    current_node.id, new_current_node_id, current_node.input_data
                )
                current_node.output_data = self._update_node_data_key(
                    current_node.id, new_current_node_id, current_node.output_data
                )
                graph.node_map.pop(current_node.id, None)
                current_node.id = new_current_node_id
                graph.node_map[new_current_node_id] = current_node
