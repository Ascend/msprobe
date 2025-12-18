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

from typing import cast
import networkx as nx

def get_cycle_number(graph: nx.DiGraph) -> int:
    """获取图中环的数量

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    int
        环的数量
    """
    if nx.is_directed_acyclic_graph(graph):
        return 0
    cycles = list(nx.simple_cycles(graph))
    return len(cycles)

def print_cycle_number(graph: nx.DiGraph):
    """打印图中环的数量

    Parameters
    ----------
    graph : nx.DiGraph
        图
    """
    cycle_num = get_cycle_number(graph)
    if cycle_num == 0:
        print("No cycle.")
        return
    print(f"Cycle number is {cycle_num}.")

def get_cycles(graph: nx.DiGraph) -> list:
    """获取图中所有环

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list
        图中环
    """
    if nx.is_directed_acyclic_graph(graph):
        return []
    cycles = [sorted(cycle) for cycle in nx.simple_cycles(graph)]
    return sorted(cycles, key=lambda x: (len(x), x))

def print_cycles(graph: nx.DiGraph):
    """打印图中所有环

    Parameters
    ----------
    graph : nx.DiGraph
        图
    """
    cycles = get_cycles(graph)
    if len(cycles) == 0:
        print("No cycle.")
        return
    print(
        "\n\n".join(
            "\n".join(
                [str(node) for node in cycle]
            ) for cycle in cycles
        )
    )

def get_subgraph_number(graph: nx.DiGraph) -> int:
    """获取连通子图数量

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    int
        连通子图数量
    """
    subgraph_list = list(nx.weakly_connected_components(graph))
    return len(subgraph_list)

def print_subgraph_number(graph: nx.DiGraph):
    """打印连通子图数量

    Parameters
    ----------
    graph : nx.DiGraph
        图
    """
    subgraph_num = get_subgraph_number(graph)
    print(f"Subgraph number is {subgraph_num}.")

def get_subgraphs_node_number(graph: nx.DiGraph) -> list[int]:
    """获取所有连通子图的节点数量

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list[int]
        所有连通子图的节点数量
    """
    subgraphs_node_number = [len(cast(set, subgraph_nodes))
                             for subgraph_nodes in nx.weakly_connected_components(graph)]
    subgraphs_node_number.sort()
    return subgraphs_node_number

def print_subgraphs_node_number(graph: nx.DiGraph):
    """打印所有连通子图的节点数量

    Parameters
    ----------
    graph : nx.DiGraph
        图
    """
    subgraphs_node_number = get_subgraphs_node_number(graph)
    print(", ".join([str(subgraph_node_number) for subgraph_node_number in subgraphs_node_number]))

def get_subgraphs(graph: nx.DiGraph) -> list:
    """获取所有连通子图

    Parameters
    ----------
    graph : nx.DiGraph
        图

    Returns
    -------
    list
        所有连通子图
    """
    subgraph_list = [sorted(subgraph_nodes) for subgraph_nodes in nx.weakly_connected_components(graph)]
    subgraph_list = sorted(subgraph_list, key=lambda subgraph_nodes: (len(subgraph_nodes), subgraph_nodes))
    return subgraph_list

def print_subgraphs(graph: nx.DiGraph):
    """打印所有连通子图

    Parameters
    ----------
    graph : nx.DiGraph
        图
    """
    subgraph_list = get_subgraphs(graph)
    print("\n".join(", ".join([str(node) for node in subgraph_nodes]) for subgraph_nodes in subgraph_list))