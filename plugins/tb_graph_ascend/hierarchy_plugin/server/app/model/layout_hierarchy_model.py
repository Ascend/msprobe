# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
     
