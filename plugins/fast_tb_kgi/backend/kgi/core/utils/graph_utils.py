# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#         http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# --------------------------------------------------------------------------------------------#

from tensorboard.util import tb_logging
import networkx as nx
from typing import Optional, cast
from backend.kgi.core.structs.node import Node

logger = tb_logging.get_logger()

def get_nodes_degree(graph: nx.DiGraph) -> dict[Node, int]:
    """获取图中所有节点的度

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    dict[Node, int]
        所有节点的度
    """
    nodes_degree = {cast(Node, node): cast(int, (graph.in_degree[node] + graph.out_degree[node]))
                    for node in graph.nodes()}

    return nodes_degree

def get_nodes_out_degree(graph: nx.DiGraph) -> dict[Node, int]:
    """获取图中所有节点的出度

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    dict[Node, int]
        所有节点的出度
    """
    nodes_out_degree = {cast(Node, node): cast(int, graph.out_degree[node])
                        for node in graph.nodes()}

    return nodes_out_degree

def get_nodes_in_degree(graph: nx.DiGraph) -> dict[Node, int]:
    """获取图中所有节点的入度

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    dict[Node, int]
        所有节点的入度
    """
    nodes_in_degree = {cast(Node, node): cast(int, graph.in_degree[node])
                       for node in graph.nodes()}

    return nodes_in_degree

def get_isolated_nodes(graph: nx.DiGraph) -> set[Node]:
    """获取图中所有孤立节点

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    set[Node]
        所有孤立节点
    """
    nodes_degree = get_nodes_degree(graph)

    isolated_nodes = {node for node, degree in nodes_degree.items() if degree == 0}

    return isolated_nodes

def _get_subgraph_nodes(graph: nx.DiGraph, anchor: Node, grow_up: bool) -> set[Node]:
    """获取子图节点

    Parameters
    ----------
    graph : nx.DiGraph
        图
    anchor : Node
        锚点
    grow_up : bool
        基于锚点生成子图的方向,
        True表示基于锚点向上生成子图, 即锚点及其依赖节点生成的子图,
        False表示基于锚点向下生成子图, 即锚点及其影响节点生成的子图

    Returns
    -------
    set[Node]
        子图节点
    """
    get_around = graph.successors
    if grow_up:
        get_around = graph.predecessors

    sub_graph_nodes = {anchor}
    cur_layer_nodes = {anchor}
    while True:
        around_nodes = set()
        for cur_layer_node in cur_layer_nodes:
            around_nodes.update(get_around(cur_layer_node))
        around_nodes = around_nodes - sub_graph_nodes
        if len(around_nodes) == 0:
            logger.info(f"Get subgraph, node num is {len(sub_graph_nodes)}.")
            break
        sub_graph_nodes.update(around_nodes)
        cur_layer_nodes = around_nodes

    return sub_graph_nodes

def get_subgraph_without_anchor(graph: nx.DiGraph, anchor: Optional[Node], grow_up: bool) -> nx.DiGraph:
    """获取子图, 结果不包含锚点

    Parameters
    ----------
    graph : nx.DiGraph
        图
    anchor : Optional[Node]
        锚点
    grow_up : bool
        基于锚点生成子图的方向,
        True表示基于锚点向上生成子图, 即锚点及其依赖节点生成的子图,
        False表示基于锚点向下生成子图, 即锚点及其影响节点生成的子图

    Returns
    -------
    nx.DiGraph
        子图, 不包含锚点
    """
    if not anchor:
        return graph

    sub_graph_nodes = _get_subgraph_nodes(graph, anchor, grow_up)
    sub_graph_nodes.remove(anchor)

    return cast(nx.DiGraph, graph.subgraph(sub_graph_nodes))

def get_subgraph_with_anchor(graph: nx.DiGraph, anchor: Optional[Node], grow_up: bool) -> nx.DiGraph:
    """获取子图, 结果包含锚点

    Parameters
    ----------
    graph : nx.DiGraph
        图
    anchor : Optional[Node]
        锚点
    grow_up : bool
        基于锚点生成子图的方向,
        True表示基于锚点向上生成子图, 即锚点及其依赖节点生成的子图,
        False表示基于锚点向下生成子图, 即锚点及其影响节点生成的子图

    Returns
    -------
    nx.DiGraph
        子图, 包含锚点
    """
    if not anchor:
        return graph

    sub_graph_nodes = _get_subgraph_nodes(graph, anchor, grow_up)

    return cast(nx.DiGraph, graph.subgraph(sub_graph_nodes))

def get_topo_layers(graph: nx.DiGraph) -> tuple[list[set[Node]], dict[Node, int]]:
    """获取拓扑分层

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list[set[Node]]
        拓扑分层
    dict[Node, int]
        节点所在层号
    """
    nodes_out_degree = get_nodes_out_degree(graph)
    cur_layer = {node for node, out_degree in nodes_out_degree.items() if out_degree == 0}
    cur_layer_index = 0
    topo_layers = [cur_layer]
    nodes_layer_id: dict[Node, int] = {node: cur_layer_index for node in cur_layer}

    while True:
        next_layer = set()
        for node in cur_layer:
            for predecessor in graph.predecessors(node):
                nodes_out_degree[predecessor] -= 1
                if nodes_out_degree[predecessor] == 0:
                    next_layer.add(predecessor)
                    nodes_layer_id[predecessor] = cur_layer_index + 1
        if len(next_layer) == 0:
            logger.info(f"Get topo layers, max layer index is {cur_layer_index}.")
            break
        topo_layers.append(next_layer)
        cur_layer = next_layer
        cur_layer_index += 1

    return topo_layers, nodes_layer_id

def _opt_topo_layers(
        graph: nx.DiGraph,
        topo_layers: list[set[Node]],
        nodes_layer_id: dict[Node, int]
) -> list[set[Node]]:
    """优化拓扑分层

    Parameters
    ----------
    graph : nx.DiGraph
        图
    topo_layers : list[set[Node]]
        拓扑分层
    nodes_layer_id : dict[Node, int]
        节点所在层号

    Returns
    -------
    list[set[Node]]
        优化的拓扑分层
    """
    topo_layer_num = len(topo_layers)
    for layer_id in range(topo_layer_num - 1, -1, -1):
        topo_layer = topo_layers[layer_id]
        for node in topo_layer.copy():
            if graph.in_degree(node) == 0:
                continue
            min_layer_id = min(nodes_layer_id[predecessor] for predecessor in graph.predecessors(node))
            if min_layer_id - 1 > layer_id:
                topo_layer.remove(node)
                topo_layers[min_layer_id - 1].add(node)
                nodes_layer_id[node] = min_layer_id - 1

    return topo_layers

def get_opt_topo_layers(graph: nx.DiGraph) -> list[set[Node]]:
    """获取优化的拓扑分层

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list[set[Node]]
        优化的拓扑分层
    """
    topo_layers, nodes_layer_id = get_topo_layers(graph)
    topo_layers = _opt_topo_layers(graph, topo_layers, nodes_layer_id)

    return topo_layers

def get_bfs_layers(graph: nx.DiGraph) -> list[set[Node]]:
    """获取bfs分层

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list[set[Node]]
        bfs分层
    """
    virtual_node = Node()

    edges_to_add = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            edges_to_add.append((virtual_node, node))
    graph.add_edges_from(edges_to_add)

    bfs_layers = list(nx.bfs_layers(graph, sources=[virtual_node]))
    bfs_layers = bfs_layers[1:]  # 移除虚拟节点层
    bfs_layers = [set(layer) for layer in bfs_layers]

    graph.remove_node(virtual_node)  # 还原图

    return bfs_layers

def get_nodes_pos(graph: nx.DiGraph) -> dict[Node, tuple[int, int]]:
    """根据优化的拓扑分层结果, 计算节点位置, 用于显示图

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    dict[Node, tuple[int, int]]
        图中所有节点的位置
    """
    nodes_pos = dict()

    topo_layers = get_opt_topo_layers(graph)
    for layer_index, layer in enumerate(topo_layers):
        layer = sorted(layer, key=lambda node: node.get_node_id())
        for index, node in enumerate(layer):
            nodes_pos[node] = (index, layer_index)

    return nodes_pos

def get_nodes_pos_with_connected_subgraphs(graph: nx.DiGraph) -> dict[Node, tuple[int, int]]:
    """先拆分连通子图, 在根据优化的拓扑分层结果, 计算节点位置, 用于显示图

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    dict[Node, tuple[int, int]]
        图中所有节点的位置
    """
    nodes_pos = dict()

    connected_subgraphs = [graph.subgraph(nodes) for nodes in nx.weakly_connected_components(graph)]
    connected_subgraphs.sort(key=lambda subgraph: subgraph.number_of_nodes())
    subgraph_start_x_index = 0
    for subgraph in connected_subgraphs:
        topo_layers = get_opt_topo_layers(subgraph)
        max_layer_node_num = 0
        for layer_index, layer in enumerate(topo_layers):
            max_layer_node_num = max(len(layer), max_layer_node_num)
            layer = sorted(layer, key=lambda node: node.get_node_id())
            for index, node in enumerate(layer):
                nodes_pos[node] = (index + subgraph_start_x_index, layer_index)
        subgraph_start_x_index += max_layer_node_num + 1

    return nodes_pos