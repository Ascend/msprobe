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
import copy
from dataclasses import dataclass

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_json, load_construct_json
from msprobe.core.common.utils import load_stack_json
from msprobe.core.common.log import logger
from msprobe.visualization.builder.msprobe_adapter import get_input_output
from msprobe.visualization.graph.graph import Graph
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import GraphConst, update_shared_dict, update_pbar_info
from msprobe.visualization.db_utils import node_to_db, config_to_db


class GraphBuilder:
    backward_pattern = re.compile(r"(\.backward\.)(\d+)$")
    forward_pattern = re.compile(r"(\.forward\.)(\d+)$")
    # 匹配以大写字母开头，后接任意字母，并以Template(结尾，或包含api_template(的字符串
    template_pattern = re.compile(r'\b([A-Z][a-zA-Z]*Template|api_template|api_instance)\(')
    micro_step_dict = {}
    framework = Const.PT_FRAMEWORK

    @staticmethod
    def build(construct_path, data_path, stack_path, model_name='DefaultModel', pbar_info=None):
        """
        GraphBuilder的对外提供的构图方法
        Args:
            construct_path: construct.json路径
            data_path: dump.json路径
            stack_path: stack.json路径
            model_name: 模型名字，依赖外部输入
            pbar_info: 进度条对象
        Returns: Graph，代表图的数据结构
        """
        construct_dict, micro_step_dict = load_construct_json(construct_path)
        if not construct_dict:
            logger.error("The content of 'construct.json' is empty, failed to build graph. "
                         "When dumping data, it is necessary to select level L0 or mix in order to "
                         "collect model structure data, that is, the content of 'construct.json' is not empty.")
            raise RuntimeError
        GraphBuilder.micro_step_dict = micro_step_dict
        dump_dict = load_json(data_path)
        GraphBuilder.framework = dump_dict.get("framework", Const.PT_FRAMEWORK)
        stack_dict = load_stack_json(stack_path)
        data_dict = dump_dict.get(GraphConst.DATA_KEY, {})
        graph = Graph(model_name, data_path=dump_dict.get('dump_data_dir', ''), dump_data=data_dict,
                      micro_step_num=micro_step_dict.get(Const.MEGATRON_MICRO_STEP_NUMBER))
        GraphBuilder._init_nodes(graph, construct_dict, data_dict, stack_dict, pbar_info=pbar_info)
        GraphBuilder._handle_recompute(graph, stack_dict)
        GraphBuilder._collect_apis_between_modules(graph)
        GraphBuilder._add_parameters_grad(graph, data_dict)
        return graph

    @staticmethod
    def to_db(filename, config, pbar_info=None):
        config.graph_n.step = config.step
        config.graph_n.rank = config.rank
        config.graph_n.compare_mode = config.compare_mode
        node_to_db(config.graph_n, filename, pbar_info)
        if config.graph_b:
            config.graph_b.data_source = GraphConst.JSON_BENCH_KEY
            config.graph_b.step = config.step
            config.graph_b.rank = config.rank
            config.graph_b.compare_mode = config.compare_mode
            node_to_db(config.graph_b, filename, pbar_info)
        config_to_db(config, filename)

    @staticmethod
    def _handle_backward_upnode_missing(construct_dict, subnode_id, upnode_id):
        """
        如果backward节点的父级节点是null，则尝试从同名的forward节点寻找父级节点
        """
        # 匹配以.backward.后跟一个或多个数字结尾的模式
        if GraphBuilder.backward_pattern.search(subnode_id) and not upnode_id:
            forward_upnode_id = construct_dict.get(GraphBuilder.backward_pattern.sub(r".forward.\2", subnode_id))
            if forward_upnode_id:
                new_upnode_id = GraphBuilder.forward_pattern.sub(r".backward.\2", forward_upnode_id)
                if new_upnode_id in construct_dict:
                    return new_upnode_id
        # 匹配以.backward结尾的节点
        if subnode_id.endswith(Const.SEP + Const.BACKWARD) and not upnode_id:
            forward_upnode_id = construct_dict.get(subnode_id.replace(Const.BACKWARD, Const.FORWARD))
            if forward_upnode_id:
                new_upnode_id = forward_upnode_id.replace(Const.FORWARD, Const.BACKWARD)
                if new_upnode_id in construct_dict:
                    return new_upnode_id
        return upnode_id

    @staticmethod
    def _handle_backward_inplace(construct_dict, sub_node_id, up_node_id):
        """
        如果当前backward节点的父层级信息不等于其父级节点的层级信息，则尝试从同名的forward节点寻找父级节点
        主要针对的场景：inplace层会无法触发backward hook导致反向层级错误

        example:
            正确的层级关系：
                父层：Module.layer4.1.BasicBlock.backward.0的层级信息为Module.layer4.1
                子层：Module.layer4.1.conv2.Conv2d.backward.0的父层级信息为Module.layer4.1

            错误的层级关系：
                父层：Module.layer4.1.relu.ReLU.backward.1的层级信息为Module.layer4.1.relu
                子层：Module.layer4.1.conv2.Conv2d.backward.0的父层级信息为Module.layer4.1
        """
        if GraphBuilder.backward_pattern.search(sub_node_id) and up_node_id:
            sub_split = sub_node_id.split(Const.SEP)
            if len(sub_split) < 5:
                return up_node_id
            up_split = up_node_id.split(Const.SEP)
            if len(up_split) < 4:
                return up_node_id
            sub_node_prefix = Const.SEP.join(sub_split[:-4])
            up_node_prefix = Const.SEP.join(up_split[:-3])
            if sub_node_prefix != up_node_prefix:
                forward_sub_node_id = GraphBuilder.backward_pattern.sub(r".forward.\2", sub_node_id)
                if forward_sub_node_id in construct_dict:
                    forward_up_node_id = construct_dict.get(forward_sub_node_id)
                    # forward_up_node_id ---> null
                    if not forward_up_node_id:
                        return forward_up_node_id
                    new_up_node_id = GraphBuilder.forward_pattern.sub(r".backward.\2", forward_up_node_id)
                    if new_up_node_id in construct_dict:
                        return new_up_node_id
        return up_node_id

    @staticmethod
    def _init_nodes(graph, construct_dict, data_dict, stack_dict, pbar_info=None):
        if pbar_info:
            update_shared_dict(pbar_info.current_stage_dict, pbar_info.task_id, 1)
        for i, (subnode_id, upnode_id) in enumerate(construct_dict.items()):
            upnode_id = GraphBuilder._handle_backward_inplace(construct_dict, subnode_id, upnode_id) if upnode_id \
                else GraphBuilder._handle_backward_upnode_missing(construct_dict, subnode_id, upnode_id)
            if upnode_id:
                upnode_op = NodeOp.get_node_op(upnode_id)
                upnode = GraphBuilder._create_or_get_node(graph, [data_dict, stack_dict], upnode_op, upnode_id)
            else:
                upnode = graph.root
            node_op = NodeOp.get_node_op(subnode_id)
            GraphBuilder._create_or_get_node(graph, [data_dict, stack_dict], node_op, subnode_id, upnode)

            if pbar_info:
                update_pbar_info(pbar_info, i + 1, len(construct_dict))

    @staticmethod
    def _create_or_get_node(graph, data_stack_list, op, name, upnode=None):
        if name in graph.node_map:
            node = graph.get_node(name)
        else:
            graph.add_node(op, name, upnode)
            node = graph.get_node(name)
            node_data = data_stack_list[0].get(name, {})
            node_stack_info = data_stack_list[1].get(name, [])
            # 添加输入输出数据
            input_data, output_data = get_input_output(node_data, node.id)
            # 更新数据
            node.set_input_output(input_data, output_data)
            if GraphConst.BATCH_P2P in name:
                GraphBuilder._extract_batch_p2p_info(node, node_data)
            # 反向节点使用对应前向节点的堆栈信息
            # 模块命名举例：Module.module.module.GPTModel.backward.0; API命名举例：Tensor.permute.1.backward
            if (not node_stack_info and
                    (GraphBuilder.backward_pattern.search(name) or name.endswith(f'{Const.SEP}{Const.BACKWARD}'))):
                forward_node = graph.get_node(
                    # 同名模块全局唯一，无论调用几次堆栈信息都一致，直接使用编号0的同名模块堆栈信息，避免遗漏
                    GraphBuilder.backward_pattern.sub(f'{Const.SEP}{Const.FORWARD}{Const.SEP}0', name)) \
                    if GraphBuilder.backward_pattern.search(name) \
                    else graph.get_node(name.replace(Const.BACKWARD, Const.FORWARD))
                node_stack_info = forward_node.stack_info if forward_node \
                    else ['This backward node cannot find the forward node and cannot retrieve stack information.']
            node.stack_info = node_stack_info
            if GraphBuilder.micro_step_dict:
                node.micro_step_id = GraphBuilder.micro_step_dict.get(node.id, 0)
        # 添加节点
        node.add_upnode(upnode)
        return node

    @staticmethod
    def _is_valid_batch_p2p_output(param_list):
        if not isinstance(param_list, list) or not param_list:
            return False
        if not isinstance(param_list[0], list) or not param_list[0]:
            return False
        return True

    @staticmethod
    def _extract_batch_p2p_info(node, node_data):
        param_list = node_data.get(Const.OUTPUT, [])
        # 数据格式："output": [[{param1}, {param2}, ...]]
        if GraphBuilder._is_valid_batch_p2p_output(param_list):
            for param in param_list[0]:
                if not isinstance(param, dict):
                    continue
                info = {GraphConst.OP: param.get(GraphConst.OP), GraphConst.PEER: param.get(GraphConst.PEER),
                        GraphConst.GROUP_ID: param.get(GraphConst.GROUP_ID)}
                node.batch_p2p_info.append(info)

    @staticmethod
    def _collect_apis_between_modules(graph):
        """
        图首次展开，这些首层节点包含许多module和api，api数量很多导致图被拉得很长严重影响查阅，因此将module之间的apis收集起来成为节点
        Args:
            graph: 模型结构

        Returns: None
        """
        i = 0
        output = []
        node_list = graph.root.subnodes
        while i < len(node_list):
            current_node = node_list[i]

            # 当前节点为api，检查后续是否还有api
            if current_node.op == NodeOp.function_api:
                temp_nodes = [current_node]
                i += 1
                while i < len(node_list) and node_list[i].op == NodeOp.function_api:
                    temp_nodes.append(node_list[i])
                    i += 1

                # 检查api节点是否大于等于2个
                if len(temp_nodes) >= 2:
                    # 创建新节点，将这些api节点放入新节点的subnodes属性
                    node_id = graph.add_node(NodeOp.api_collection, GraphConst.APIS_BETWEEN_MODULES,
                                             id_accumulation=True)
                    api_collection_node = graph.get_node(node_id)
                    api_collection_node.subnodes = temp_nodes
                    # 重新确立父子关系
                    for node in temp_nodes:
                        node.upnode = api_collection_node
                    api_collection_node.upnode = graph.root
                    output.append(api_collection_node)
                    if temp_nodes[0].micro_step_id is not None:
                        api_collection_node.micro_step_id = temp_nodes[0].micro_step_id
                else:
                    # 如果连续的api节点不足2个，将它们原样添加到输出列表
                    output.extend(temp_nodes)
            else:
                # 如果当前节点为module，直接添加到输出列表
                output.append(current_node)
                i += 1

        graph.root.subnodes = output

    @staticmethod
    def _add_parameters_grad(graph, data_dict):
        """
        将parameters_grad信息添加到graph中，
        对应模块的parameters_grad节点添加到对应模块的最后一次backward节点（backward计数最大）内作为子节点

        例如，graph有节点Module.a.backward.0, Module.a.backward.1, Module.a.backward.2
        则Module.a.parameters_grad添加在Module.a.backward.2内作为子节点
        """
        prefix_suffix_map = {}
        base_suffix = Const.SEP + Const.PARAMS_GRAD
        pattern = re.compile(rf"{re.escape(base_suffix)}(?:\.\d+)?$")
        for node_id in data_dict.keys():
            if node_id not in graph.node_map:
                m = pattern.search(node_id)
                if not m:
                    continue
                prefix = node_id[:m.start()]
                suffix = node_id[m.start():]
                prefix_suffix_map.setdefault(prefix, []).append(suffix)

        max_info = {prefix: 0 for prefix in prefix_suffix_map}

        for key in graph.node_map.keys():
            parts = key.split(Const.SEP)
            if len(parts) > 2 and parts[-2] == Const.BACKWARD:
                num = int(parts[-1])
                prefix = Const.SEP.join(parts[:-2])
                if prefix in max_info and num > max_info[prefix]:
                    max_info[prefix] = num

        for prefix, num in max_info.items():
            node_id = prefix + Const.SEP + Const.BACKWARD + Const.SEP + str(num)
            node = graph.get_node(node_id)
            if node:
                suffix_list = prefix_suffix_map[prefix]
                for suffix in suffix_list:
                    parameters_grad_node_id = graph.add_node(NodeOp.module, prefix + suffix, up_node=node)
                    # 添加输入输出数据
                    node_data = data_dict.get(parameters_grad_node_id, {})
                    input_data, output_data = get_input_output(node_data, parameters_grad_node_id)
                    # 更新数据
                    graph.get_node(parameters_grad_node_id).set_input_output(input_data, output_data)

    @staticmethod
    def _handle_recompute(graph, stack_dict):
        """
        1. 通过_get_recompute_map获得重计算节点映射recompute_map: dict(node_id: node_id_prefix)
        2. 通过_get_no_recompute_map获得非重计算节点映射no_recompute_map: dict(node_id_prefix: list(node_id))
        3. 遍历recompute_map，通过node_id_prefix与no_recompute_map建立连接，通过非重计算节点找到自身的父节点
        """
        recompute_map, recompute_id_map = GraphBuilder._get_recompute_map(graph.root.subnodes, stack_dict)
        if not recompute_map:
            return
        id_prefixes = set(recompute_map.values())
        no_recompute_map = GraphBuilder._get_no_recompute_map(graph, id_prefixes)
        if not no_recompute_map:
            return
        # 拷贝非重计算节点字典用于反向模式
        no_recompute_ids_b = {node_id: list(node_list) for node_id, node_list in no_recompute_map.items()}

        del_indexes = []
        for node_id, id_prefix in recompute_map.items():
            if id_prefix not in no_recompute_map:
                continue
            node_list = no_recompute_map.get(id_prefix) if GraphBuilder.forward_pattern.search(node_id) else \
                no_recompute_ids_b.get(id_prefix)
            if not node_list:
                continue
            no_recompute_node = node_list.pop()
            recompute_node = graph.node_map.get(node_id)
            if not recompute_node:
                continue
            # 通过非重计算forward节点的父节点，找到对应的backward父节点
            new_up_node = graph.node_map.get(
                GraphBuilder.forward_pattern.sub(r".backward.\2", no_recompute_node.upnode.id))
            if not new_up_node:
                continue

            # 更新节点连接关系，同时避免可能出现的父节点链接到子节点的情况
            new_up_node_ancestors = new_up_node.get_ancestors()
            if recompute_node.id not in new_up_node_ancestors:
                recompute_node.upnode = new_up_node
                new_up_node.subnodes.append(recompute_node)

                del_indexes.append(recompute_id_map.get(node_id))

        # 从后往前删除graph首层中已更新父节点的重计算节点
        del_indexes.sort(reverse=True)
        for index in del_indexes:
            if 0 <= index <= len(graph.root.subnodes):
                del graph.root.subnodes[index]

    @staticmethod
    def _get_recompute_map(node_list: list, stack_dict: dict):
        """
        找到graph首层的重计算层

        return: dict(node_id: node_id_prefix), dict(node_id: index)

        example:
        {Module.0.module.decoder.layers.0.TransformerLayer.forward.4: Module.0.module.decoder.layers.0.TransformerLayer}
        """
        recompute_map = {}
        recompute_id_map = {}
        node_id_set = set([node.id for node in node_list])
        node_id_cache = set()
        for i, node in enumerate(node_list):
            if NodeOp.get_node_op(node.id) != NodeOp.module:
                continue
            if GraphBuilder.framework == Const.PT_FRAMEWORK:
                node_id = node.id
                # 反向节点没有堆栈，要转换为对应的前向节点id去获取
                if GraphBuilder.backward_pattern.search(node.id):
                    node_id = GraphBuilder.backward_pattern.sub(r".forward.\2", node.id)
                if not GraphBuilder._is_recompute_by_stack_torch(stack_dict.get(node_id)):
                    continue
            id_segments = node.id.split(Const.SEP)
            prefix = Const.SEP.join(id_segments[:-2])
            if node.id in node_id_cache:
                recompute_map[node.id] = prefix
                recompute_id_map[node.id] = i
                continue
            is_recompute = GraphBuilder._is_recompute_node_id(id_segments)
            if not is_recompute:
                continue
            # 重计算层必然是一组对应的前反向节点
            id_segments[-2] = Const.BACKWARD if id_segments[-2] == Const.FORWARD else Const.FORWARD
            relative_node_id = Const.SEP.join(id_segments)
            if relative_node_id in node_id_set:
                recompute_map[node.id] = prefix
                recompute_id_map[node.id] = i
                # 对应节点id放入缓存避免后续重复判断
                node_id_cache.add(relative_node_id)
        return recompute_map, recompute_id_map

    @staticmethod
    def _is_recompute_by_stack_torch(stack_list: list):
        """
        通过stack.json堆栈信息判断重计算情况
        """
        if not stack_list:
            return False

        parsed_stack = []
        for stack_str in stack_list:
            try:
                stack_str = stack_str.strip().replace("File ", "")
                file_path = stack_str.split(", line ")[0].strip()
                func_name = stack_str.split(", in ")[1].split(",")[0].strip()
                parsed_stack.append({"filename": file_path, "function": func_name})
            except Exception:
                continue

        if not parsed_stack:
            return False

        # 1. 最高优先级：官方重计算函数 recompute_fn
        for frame in parsed_stack:
            if frame["function"] == "recompute_fn" and frame["filename"].endswith("torch/utils/checkpoint.py"):
                return True

        # 2. 匹配 Megatron/MindSpeed 自定义反向函数
        backward_indices = []
        for idx, frame in enumerate(parsed_stack):
            if (frame["function"] == 'backward' or
                    (frame["function"] == 'checkpoint_function_backward' and "megatron" in frame["filename"])):
                backward_indices.append(idx)

        # 3. 校验调用来源为 PyTorch 自动求导核心
        for idx in backward_indices:
            if idx + 1 < len(parsed_stack) and parsed_stack[idx + 1]["filename"].endswith("torch/autograd/function.py"):
                return True
            if idx + 2 < len(parsed_stack) and parsed_stack[idx + 2]["filename"].endswith("torch/autograd/function.py"):
                return True

        return False

    @staticmethod
    def _is_recompute_node_id(id_segments):
        """
        非重计算首层节点命名必然是：Module/Cell.{number(可选)}.module_name.{number(可选)}.class_name.forward/backward.number
        如果不符合，则判断为重计算节点
        """
        if len(id_segments) > 7:
            return True
        if len(id_segments) == 7 and not (id_segments[1].isdigit() and id_segments[3].isdigit()):
            return True
        if len(id_segments) == 6 and not id_segments[1].isdigit():
            return True
        return False

    @staticmethod
    def _get_no_recompute_map(graph, recompute_id_prefixes):
        """
        寻找与重计算层id前缀相同的非重计算forward层，按顺序排列，重计算层按照顺序使用非重计算forward层的父节点对应的backward节点

        return: dict(node_id_prefix: list(node_id))
        """
        no_recompute_map = {}
        for node_id, node in graph.node_map.items():
            if NodeOp.get_node_op(node_id) == NodeOp.module and GraphBuilder.forward_pattern.search(node_id):
                if not node.upnode or node.upnode.id == graph.root.id:
                    continue
                id_prefix = GraphBuilder.forward_pattern.sub('', node_id)
                if id_prefix not in recompute_id_prefixes:
                    continue
                no_recompute_map.setdefault(id_prefix, []).append(node)
        for node_list in no_recompute_map.values():
            # 方便按顺序pop弹出
            node_list.reverse()
        return no_recompute_map


class GraphExportConfig:
    def __init__(self, graph_n, graph_b=None, tool_tip=None, node_colors=None, micro_steps=None, task='',
                 overflow_check=False, compare_mode=None, step=0, rank=0, step_list=None, rank_list=None):
        self.graph_n = graph_n
        self.graph_b = graph_b
        self.tool_tip = tool_tip
        self.node_colors = node_colors
        self.micro_steps = micro_steps
        self.task = task
        self.overflow_check = overflow_check
        self.compare_mode = compare_mode
        self.step = step
        self.rank = rank
        self.step_list = step_list
        self.rank_list = rank_list


@dataclass
class GraphInfo:
    graph: Graph
    construct_path: str
    data_path: str
    stack_path: str


@dataclass
class BuildGraphTaskInfo:
    graph_info_n: GraphInfo
    graph_info_b: GraphInfo
    npu_rank: str
    bench_rank: str
    time_str: str
