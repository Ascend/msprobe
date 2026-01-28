# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# ==============================================================================
from .hierarchy import Hierarchy
from ..utils.global_state import GraphState
from ..utils.constant import NPU, BENCH, SINGLE


class LayoutHierarchyModel:
    npu_hierarchy = None
    bench_hierarchy = None
    single_hierarchy = None
      
    hierarchy = {
        NPU: npu_hierarchy,
        BENCH: bench_hierarchy,
        SINGLE: single_hierarchy
    }    

    @staticmethod
    def change_expand_state(node_name, graph_type, repo, micro_step, rank_step):
        if node_name == 'root':
            rank = rank_step.get('rank')
            step = rank_step.get('step')
            GraphState.set_global_value("update_precision_cache", {})  # 切换图清缓存
            LayoutHierarchyModel.hierarchy[graph_type] = Hierarchy(graph_type, repo, micro_step, rank, step)
            
        elif LayoutHierarchyModel.hierarchy.get(graph_type, None):
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_data(node_name)
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_shape()
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_position()
        else:
            return {}
        return LayoutHierarchyModel.hierarchy[graph_type].get_hierarchy()

    @staticmethod
    def update_hierarchy_data(graph_type):
        if LayoutHierarchyModel.hierarchy.get(graph_type, None):
            return LayoutHierarchyModel.hierarchy[graph_type].update_hierarchy_data()
        else:
            return {}
        
    @staticmethod
    def update_current_hierarchy_data(data):
        npu_update_data = []
        bench_update_data = []
        for node in data:
            if node['graph_type'] == NPU:
                npu_update_data.append(node)
            elif node['graph_type'] == BENCH:
                bench_update_data.append(node)
        if LayoutHierarchyModel.hierarchy.get(NPU, None) and npu_update_data:
            LayoutHierarchyModel.hierarchy[NPU].update_current_hierarchy_data(npu_update_data)
        if LayoutHierarchyModel.hierarchy.get(BENCH, None) and bench_update_data:
            LayoutHierarchyModel.hierarchy[BENCH].update_current_hierarchy_data(bench_update_data)
     
