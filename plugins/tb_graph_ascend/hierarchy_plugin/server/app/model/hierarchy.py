
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
from tensorboard.util import tb_logging
from ..utils.global_state import GraphState
from ..utils.constant import NPU_PREFIX, BENCH_PREFIX, NPU, SINGLE, UNEXPAND_NODE, MODULE, DataType

logger = tb_logging.get_logger()

INNER_WIDTH = 50  # 内部节点宽度
INNER_HIGHT = 15  # 内部节点高度
HORIZONTAL_SPACING = 5  # 横向排列间距
VERTICAL_SPACING = 10  # 纵向排列间距
MAX_PER_ROW = 5  # 横向每行最大数
DB_TYPE = DataType.DB.value
JSON_TYPE = DataType.JSON.value


class Hierarchy:
    
    def __init__(self, graph_type, repo, micro_step, rank, step):
        # DB：查询根节点信息
        node_info = repo.query_root_nodes(graph_type, rank, step)
        if not node_info:
            logger.error("No root node info found in database.")
            return
        root_node_name = node_info.get('node_name')
        name_prefix = NPU_PREFIX if graph_type == NPU else BENCH_PREFIX
        name_prefix = '' if graph_type == SINGLE else name_prefix
        self.repo = repo
        self.name_prefix = name_prefix
        self.root_name = root_node_name
        self.graph_type = graph_type
        self.rank = rank
        self.step = step
        self.micro_step_id = micro_step
        self.current_hierarchy = {
            root_node_name: self.get_basic_rende_info(root_node_name, node_info)
        }
        # 默认展开根节点
        self.update_graph_data(self.root_name)
        self.update_graph_shape()
        self.update_graph_position()

    @classmethod
    def measure_text_width(cls, text):
        return len(text) * 6  # 假设每个字符宽度为6

    @classmethod
    def extract_label_name(cls, node_name, node_type):
        if not node_name:
            return ''
        splited_subnode_name = node_name.split('.')
        splited_label = []
        # 在展开层级时，将父级层级名称相关去除，仅保留子节点本身名称信息
        # 如Module.layer1.1.relu.ReLU.forward.0中的父级名称Module.layer1.1去除，仅保留子级的relu.ReLU.forward.1
        # 如Module.layer4.0.BasicBlock.forward.0中的父级名称Module.1去除，仅保留子级的layer4.0.BasicBlock.forward.0
        if int(node_type) == MODULE:
            if len(splited_subnode_name) < 4:
                return node_name
            elif not splited_subnode_name[-4].isdigit():
                splited_label = splited_subnode_name[-4:]
            elif len(splited_subnode_name) > 4:
                splited_label = splited_subnode_name[-5:]
            else:
                return node_name
        # 在展开层级时，将父级层级名称相关去除，仅保留API子节点本身名称信息，
        # 如 Module.layer1.1.ApiList.1 中的父级名称Module.layer1.1去除，仅保留子级的ApiList.1
        # 如 Module.layer1.1.ApiList.0.1 中的父级名称Module.layer1.1去除，仅保留子级的ApiList.0.1
        else:
            if len(splited_subnode_name) < 2:
                return node_name
            elif not splited_subnode_name[-2].isdigit():
                splited_label = splited_subnode_name[-2:]
            elif len(splited_subnode_name) > 2:
                splited_label = splited_subnode_name[-3:]
            else:
                return node_name
        return ('.').join(splited_label)

    def update_graph_data(self, node_name):
        target_node = self.current_hierarchy.get(node_name, {})
        if node_name == self.root_name or target_node:
            self.process_click_expand(node_name)
        else:  # 如果图中不存在该节点，说明是选中节点，需要递归展开该节点的所有父节点
            self.process_select_expand(node_name)

    def process_click_expand(self, node_name):
        target_node = self.current_hierarchy.get(node_name, {})
        target_node_children = target_node.get("children", [])
        if not target_node or not target_node_children:
            return
        if not target_node.get('expand', False):
            # 1.将target_node的expand置为true
            # 2.将node_name的子节点信息初始化，并添加到current_hierarchy中
            # DB: 查询所有以当前为父节点的子节点
            sub_nodes = self.repo.query_sub_nodes(node_name, self.graph_type, self.rank, self.step)
            for subnode_name, node_info in sub_nodes.items():
                if self.current_hierarchy.get(subnode_name):
                    continue
                render_info = self.get_basic_rende_info(subnode_name, node_info)
                self.current_hierarchy[subnode_name] = render_info
            target_node['expand'] = True
        else:
            target_node['expand'] = False if node_name != self.root_name else True  # 根节点默认展开

    def process_select_expand(self, node_name):
        # DB：逻辑
        # 1.查询当前节点及其的所有父节点，一直到没有父节点位置{}
        
        # DB：查询当前节点的父节点信息
        up_nodes = self.repo.query_up_nodes(node_name, self.graph_type, self.rank, self.step)
        parent_node_name = up_nodes.get(node_name, {}).get('upnode')
        parent_node = self.current_hierarchy.get(parent_node_name, '')
        # 递归展开父节点
        while not parent_node or not parent_node.get('expand', False):
            if not parent_node:  # 如果父节点不存在图中，则初始化父节点
              
                render_info = self.get_basic_rende_info(parent_node_name, up_nodes.get(parent_node_name))
                self.current_hierarchy[parent_node_name] = render_info
            try:
                self.process_click_expand(parent_node_name)  # 展开父节点
            except Exception as e:
                logger.error(f"Failed to expand parent node {parent_node_name}: {e}")
                break
            if parent_node_name == self.root_name:
                break
            parent_node_name = up_nodes.get(parent_node_name, {}).get('upnode')
            parent_node = self.current_hierarchy.get(parent_node_name, '')

    def update_graph_shape(self):
        self.resize_hierarchy(self.root_name)

    def update_graph_position(self):
        self.layout_hierarchy(self.root_name)

    def resize_hierarchy(self, current_name):
        node = self.current_hierarchy.get(current_name, {})
        if not node:
            return
        if node.get('nodeType') == UNEXPAND_NODE:
            # 不可展开叶子节点固定尺寸
            node['width'] = INNER_WIDTH
            node['height'] = INNER_HIGHT
            return
        if not node.get('expand', False):
            # 未展开的父节点按文字宽度
            node['width'] = self.measure_text_width(node.get('label', '')) + HORIZONTAL_SPACING * 2  # 文字宽度 + 边距
            node['height'] = INNER_HIGHT
            return
        for child_name in node.get('children', []):
            self.resize_hierarchy(child_name)
        max_child_width = 0
        total_height = INNER_HIGHT
        for group_type, children in self.group_children(node.get('children')):
            if group_type == UNEXPAND_NODE:
                # 横向布局：分批处理每行最多MAX_PER_ROW个
                rows = [children[i:i + MAX_PER_ROW] for i in range(0, len(children), MAX_PER_ROW)]
                if (len(rows) > 1):
                    max_child_width = max(max_child_width,
                                          INNER_WIDTH * MAX_PER_ROW + (HORIZONTAL_SPACING * (MAX_PER_ROW - 1)))
                    total_height += ((INNER_HIGHT + VERTICAL_SPACING) * len(rows))
                else:
                    child_total_width = sum(self.current_hierarchy.get(child, {}).get('width', 0) for child in children)
                    spacing = HORIZONTAL_SPACING * (len(children) - 1)
                    max_child_width = max(max_child_width, child_total_width + spacing)
                    total_height += (INNER_HIGHT + VERTICAL_SPACING)

            else:
                # 纵向布局，计算子节点最大宽度
                # 更新 max_child_width：取所有子节点中最宽的一个
                child_max_width = max(self.current_hierarchy.get(child, {}).get('width', 0) for child in children)
                max_child_width = max(max_child_width, child_max_width)
                # 累加 total_height：所有子节点高度 + 垂直间距（每个子节点之间都有间距，所以是 len(children) 个间距）
                children_heights = sum(self.current_hierarchy.get(child, {}).get('height', 0) for child in children)
                vertical_spacing = VERTICAL_SPACING * len(children)
                total_height += (children_heights + vertical_spacing)

        # 最终尺寸计算
        node['width'] = max(
            max_child_width + HORIZONTAL_SPACING * 2,  # 子节点最大宽度 + 边距
            self.measure_text_width(node.get('label', '')) + HORIZONTAL_SPACING * 2  # 保证文字可见
        )
        node['height'] = total_height + VERTICAL_SPACING  # 总高度 + 边距

    def layout_hierarchy(self, current_name):
        node = self.current_hierarchy.get(current_name, {})
        if not node or not node.get('children'):
            return
        parent_x = node.get('x')
        parent_width = node.get('width')
        current_y = node.get('y') + INNER_HIGHT + VERTICAL_SPACING  # 初始Y坐标

        # 按类型分组处理子节点
        for group_type, children in self.group_children(node.get('children')):
            if group_type == UNEXPAND_NODE:
                # 横向布局：分批处理每行最多MAX_PER_ROW个
                rows = [children[i:i + MAX_PER_ROW] for i in range(0, len(children), MAX_PER_ROW)]

                for row_children in rows:
                    # 计算该行总宽度
                    row_width = sum(
                        self.current_hierarchy.get(child_name, {}).get('width') for child_name in row_children)
                    row_width += HORIZONTAL_SPACING * (len(row_children) - 1)

                    # 居中对齐计算
                    start_x = parent_x + (parent_width - row_width) // 2
                    max_height = 0
                    current_x = start_x

                    # 布局该行所有子节点
                    for child_name in row_children:
                        child = self.current_hierarchy.get(child_name, {})
                        child['x'] = current_x
                        child['y'] = current_y
                        current_x += child.get('width') + HORIZONTAL_SPACING
                        max_height = max(max_height, child.get('height'))

                    # 更新Y坐标
                    current_y += max_height + VERTICAL_SPACING
            else:
                # 纵向布局：每个子节点单独占一行且居中
                for child_name in children:
                    child = self.current_hierarchy.get(child_name, {})
                    child['x'] = parent_x + (parent_width - child.get('width')) // 2
                    child['y'] = current_y
                    current_y += child.get('height') + VERTICAL_SPACING
                    if (child.get('expand', False)):
                        self.layout_hierarchy(child_name)  # 递归处理子节点

    def group_children(self, children):
        """将子节点按类型分组，连续同类型节点为一组"""
        groups = []
        current_group = []
        current_type = None

        for name in children:
            child = self.current_hierarchy.get(name, {})
            if not child:
                continue
            # 类型变化时创建新组
            if current_type is not None and child.get('nodeType') != current_type:
                groups.append((current_type, current_group))
                current_group = []
            current_type = child.get('nodeType')
            current_group.append(name)
        # 添加最后一组
        if current_group:
            groups.append((current_type, current_group))
        return groups

    def get_basic_rende_info(self, node_name, node_info):
        if not node_info:
            return {}
        label = node_name
        children = []
        precision_index = None,
        if node_name == self.root_name:  # 根节点，根据micro_step获取子节点
            # DB：查询以根节点为父节点的所有子节点
            sub_nodes = self.repo.query_sub_nodes(node_name, self.graph_type, self.rank, self.step)
            for subnode_name, child_node in sub_nodes.items():
                child_micro_step_id = child_node.get('micro_step_id', -1)  # 如果子节点不包含micro_step_id，则默认为-1，直接添加
                is_append_all_node = int(self.micro_step_id) == -1 or child_micro_step_id == -1
                is_append_split_node = int(self.micro_step_id) != -1 and int(child_micro_step_id) == int(
                    self.micro_step_id)
                if is_append_all_node or is_append_split_node:
                    children.append(subnode_name)
        else:
            children = node_info.get('subnodes', [])
        # precisionIndex 先从缓存中获取(更新误差可能会更新缓存)，如果不存在，则从ode_info中获取
        update_precision_cache = GraphState.get_global_value('update_precision_cache', {})
        if update_precision_cache.get(node_name) is not None:
            precision_index = update_precision_cache.get(node_name, {}).get('precision_index')
        else:
            precision_index = node_info.get('data', {}).get('precision_index', "NaN")
        if node_info.get('upnode', '') != self.root_name:  # 首层节点不处理显示内容
            label = self.extract_label_name(node_name, node_info.get('node_type'))
        render_info = {
            'x': 0,
            'y': 0,
            'width': self.measure_text_width(node_name) + HORIZONTAL_SPACING * 2,
            'height': INNER_HIGHT,
            'expand': False,
            'isRoot': node_name == self.root_name,
            'parentNode': node_info.get('upnode', ''),
            'label': label,
            'name': self.name_prefix + node_name,
            'children': children,
            'nodeType': node_info.get('node_type') if node_info.get("subnodes") else UNEXPAND_NODE,
            'matchedNodeLink': node_info.get('matched_node_link', []),
            'precisionIndex': precision_index,  # 精度
            'overflowLevel': node_info.get('data', {}).get('overflow_level', "NaN"),  # 溢出
            'matchedDistributed': node_info.get('matched_distributed', {}),
        }
        return render_info

    # 获取连通图
    def get_connected_graph(self, name, result, new_hierarchy):
        node = self.current_hierarchy.get(name, {})
        if (not node): 
            return
        filtered_value = {k: v for k, v in node.items() if k != "children"}
        result[name] = filtered_value
        new_hierarchy[name] = node
        # 递归处理子节点
        if (node.get('expand')):
            for child_name in node.get('children', []):
                self.get_connected_graph(child_name, result, new_hierarchy)

    def update_hierarchy_data(self):
        if self.repo.repo_type == DB_TYPE:
            return self.current_hierarchy
        # 处理JSON
        for node_name, node_info in self.current_hierarchy.items():
            graph_node_info = self.repo.query_node_info(node_name, self.graph_type)
            node_info['matchedNodeLink'] = graph_node_info.get('matched_node_link', [])
            node_info['precisionIndex'] = graph_node_info.get('data', {}).get('precision_index', "NaN")
        return self.current_hierarchy

    def update_current_hierarchy_data(self, data):
        for node_info in data:
            node_name = node_info.get('node_name')
            if node_name in self.current_hierarchy:
                current_node_info = self.current_hierarchy[node_name]
                current_node_info['matchedNodeLink'] = node_info.get('matched_node_link')
                current_node_info['precisionIndex'] = node_info.get('precision_index')
        return True
            
    def get_hierarchy(self):
        result = {}
        new_hierarchy = {}
        # 遍历当前层级结构
        self.get_connected_graph(self.root_name, result, new_hierarchy)
        self.current_hierarchy = new_hierarchy  # 折叠之后的需要重新展开
        return result
