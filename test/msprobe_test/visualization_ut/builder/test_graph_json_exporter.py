# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphJsonExporter
from msprobe.visualization.graph.graph import Graph
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import GraphConst


class TestGraphJsonExporter(unittest.TestCase):
    def setUp(self):
        self.model_name = 'TestNet'
        self.graph = Graph(self.model_name)
        self.graph.dump_task = 'test_task'
        self.graph.dump_level = 'test_level'
        self.graph.data_path = '/fake/data/path'

        # 创建临时输出目录
        self.tmp_dir = tempfile.mkdtemp()

        # ---- 节点1：在 raw_data 中的正常节点 ----
        self.graph.add_node(NodeOp.module, 'Layer1', self.graph.root)
        self.graph.node_map['Layer1'].input_data = {
            'key0': {'full_op_name': 'Layer1.input.0', 'shape': (4, 4), 'dtype': 'float32'},
            'key1': {'full_op_name': 'Layer1.input.1', 'shape': (4, 4), 'dtype': 'float32'},
        }
        self.graph.node_map['Layer1'].output_data = {
            'key2': {'full_op_name': 'Layer1.output.0', 'shape': (4, 4), 'dtype': 'float32'},
        }

        # ---- 节点2：仅在 raw_data 中带 kwargs/params 的节点 ----
        self.graph.add_node(NodeOp.module, 'Layer2', self.graph.root)
        self.graph.node_map['Layer2'].input_data = {
            'k0': {'full_op_name': 'Layer2.input.0', 'shape': (8, 8), 'dtype': 'float16'},
            'k1': {'full_op_name': 'Layer2.input.attention_mask', 'shape': (1, 8), 'dtype': 'bool'},
        }
        self.graph.node_map['Layer2'].output_data = {
            'k2': {'full_op_name': 'Layer2.output.0', 'shape': (8, 8), 'dtype': 'float16'},
        }
        # params 存放在 dump_data 的原始数据中
        raw_params_item = {'full_op_name': 'Layer2.parameters.0', 'shape': (256, 256), 'dtype': 'float32'}

        # ---- 节点3：不在 raw_data 中的节点（模拟 PP 重命名） ----
        self.graph.add_node(NodeOp.module, 'PP_Renamed_Layer', self.graph.root)
        self.graph.node_map['PP_Renamed_Layer'].input_data = {
            'pk0': {'full_op_name': 'PP_Renamed_Layer.input.0', 'shape': (2, 2), 'dtype': 'float32'},
        }
        self.graph.node_map['PP_Renamed_Layer'].output_data = {
            'pk1': {'full_op_name': 'PP_Renamed_Layer.output.0', 'shape': (2, 2), 'dtype': 'float32'},
        }

        # ---- 节点4：有 stack_info 的节点 ----
        self.graph.add_node(NodeOp.module, 'LayerWithStack', self.graph.root)
        self.graph.node_map['LayerWithStack'].input_data = {
            'sk0': {'full_op_name': 'LayerWithStack.input.0', 'shape': (1,), 'dtype': 'float32'},
        }
        self.graph.node_map['LayerWithStack'].output_data = {
            'sk1': {'full_op_name': 'LayerWithStack.output.0', 'shape': (1,), 'dtype': 'float32'},
        }
        self.graph.node_map['LayerWithStack'].stack_info = ['trace_line1', 'trace_line2']

        # ---- 节点5：用于测试 stack_info 分组 ----
        self.graph.add_node(NodeOp.module, 'LayerWithStack2', self.graph.root)
        self.graph.node_map['LayerWithStack2'].input_data = {
            'sk2': {'full_op_name': 'LayerWithStack2.input.0', 'shape': (2,), 'dtype': 'float32'},
        }
        self.graph.node_map['LayerWithStack2'].output_data = {
            'sk3': {'full_op_name': 'LayerWithStack2.output.0', 'shape': (2,), 'dtype': 'float32'},
        }
        self.graph.node_map['LayerWithStack2'].stack_info = ['trace_line1', 'trace_line2']  # 与 node4 相同 stack

        # ---- 节点6：无 input/output 的节点（应被跳过） ----
        self.graph.add_node(NodeOp.module, 'EmptyNode', self.graph.root)

        # ---- 构建 dump_data（原始 JSON 数据） ----
        self.graph.dump_data = {
            'Layer1': {
                Const.INPUT_ARGS: [
                    {'shape': (4, 4), 'dtype': 'float32'},
                    {'shape': (4, 4), 'dtype': 'float32'},
                ],
                Const.OUTPUT: [
                    {'shape': (4, 4), 'dtype': 'float32'},
                ],
            },
            'Layer2': {
                Const.INPUT_ARGS: [
                    {'shape': (8, 8), 'dtype': 'float16'},
                ],
                Const.INPUT_KWARGS: {
                    'attention_mask': {'shape': (1, 8), 'dtype': 'bool', 'type': 'Tensor'},
                },
                Const.OUTPUT: [
                    {'shape': (8, 8), 'dtype': 'float16'},
                ],
                Const.PARAMS: [
                    raw_params_item,
                ],
            },
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ======================== to_json ========================

    @patch('msprobe.visualization.builder.graph_builder.logger')
    def test_to_json_creates_files(self, mock_logger):
        """验证 to_json 正确创建三个 JSON 文件。"""
        GraphJsonExporter.to_json(self.tmp_dir, self.graph, rank=0, step=0)
        out_dir = os.path.join(self.tmp_dir, 'step0', 'rank0')
        self.assertTrue(os.path.isdir(out_dir))
        for filename in [GraphConst.CONSTRUCT_FILE, GraphConst.DUMP_FILE, GraphConst.STACK_FILE]:
            self.assertTrue(os.path.isfile(os.path.join(out_dir, filename)), f'{filename} should exist')

        # 验证文件内容是合法 JSON
        for filename in [GraphConst.CONSTRUCT_FILE, GraphConst.DUMP_FILE, GraphConst.STACK_FILE]:
            with open(os.path.join(out_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

        mock_logger.info.assert_called()

    # ======================== _build_construct_dict ========================

    def test_build_construct_dict(self):
        """验证父子节点关系 dict 正确构建。"""
        # 在根节点下添加子节点的子节点
        self.graph.add_node(NodeOp.module, 'SubLayer', self.graph.root)
        self.graph.add_node(NodeOp.module, 'SubSubLayer', self.graph.node_map['SubLayer'])

        result = GraphJsonExporter._build_construct_dict(self.graph)

        # root 本身不应在 construct 中（没有 upnode）
        self.assertNotIn(self.model_name, result)

        # 直系子节点
        self.assertEqual(result.get('SubLayer'), self.model_name)
        # 孙节点
        self.assertEqual(result.get('SubSubLayer'), 'SubLayer')

    def test_build_construct_dict_empty(self):
        """验证单 root 图返回空 dict。"""
        empty_graph = Graph('Empty')
        result = GraphJsonExporter._build_construct_dict(empty_graph)
        self.assertEqual(result, {})

    # ======================== _build_dump_dict ========================

    def test_build_dump_dict_metadata(self):
        """验证 dump_dict 的 metadata 字段正确。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        self.assertEqual(result['task'], 'test_task')
        self.assertEqual(result['level'], 'test_level')
        self.assertEqual(result['framework'], GraphBuilder.framework)
        self.assertEqual(result['dump_data_dir'], '/fake/data/path')

    def test_build_dump_dict_root_skipped(self):
        """验证 root 节点不在 data 中。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})
        self.assertNotIn(self.model_name, data)

    def test_build_dump_dict_empty_node_skipped(self):
        """验证无 input/output 的节点被跳过。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})
        self.assertNotIn('EmptyNode', data)

    def test_build_dump_dict_normal_node_from_raw(self):
        """验证正常节点从 raw_data 重建，包含 input_args/output。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})

        layer1 = data.get('Layer1', {})
        self.assertIn(Const.INPUT_ARGS, layer1)
        self.assertIsInstance(layer1[Const.INPUT_ARGS], list)
        self.assertEqual(len(layer1[Const.INPUT_ARGS]), 2)
        self.assertIn(Const.OUTPUT, layer1)
        self.assertEqual(len(layer1[Const.OUTPUT]), 1)

    def test_build_dump_dict_kwargs_and_params(self):
        """验证 input_kwargs 和 params 字段被正确重建。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})

        layer2 = data.get('Layer2', {})
        self.assertIn(Const.INPUT_KWARGS, layer2)
        self.assertIn('attention_mask', layer2[Const.INPUT_KWARGS])
        self.assertEqual(layer2[Const.INPUT_KWARGS]['attention_mask']['type'], 'Tensor')
        self.assertIn(Const.PARAMS, layer2)
        self.assertEqual(len(layer2[Const.PARAMS]), 1)

    def test_build_dump_dict_stats_merged(self):
        """验证合并后的张量统计信息（Max/Min/Mean/Norm）覆盖原始数据。"""
        # 在 node 的 output_data 中添加统计信息（模拟合并后的数据）
        layer1_node = self.graph.node_map['Layer1']
        layer1_node.output_data['key2']['Max'] = 0.99
        layer1_node.output_data['key2']['Min'] = 0.01
        layer1_node.output_data['key2']['Mean'] = 0.5
        layer1_node.output_data['key2']['Norm'] = 1.0

        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})
        output_entry = data['Layer1'][Const.OUTPUT][0]

        self.assertEqual(output_entry.get('Max'), 0.99)
        self.assertEqual(output_entry.get('Min'), 0.01)
        self.assertEqual(output_entry.get('Mean'), 0.5)
        self.assertEqual(output_entry.get('Norm'), 1.0)
        # 原始数据中的字段仍然保留
        self.assertEqual(output_entry.get('dtype'), 'float32')

    def test_build_dump_dict_partial_stats_merged(self):
        """验证部分统计信息覆盖（只覆盖合并数据中存在的字段）。"""
        layer1_node = self.graph.node_map['Layer1']
        # 只在 output.0 的合并数据中有 Max，output.1 没有
        layer1_node.output_data['key2']['Max'] = 0.88

        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})
        output_entry = data['Layer1'][Const.OUTPUT][0]
        self.assertEqual(output_entry.get('Max'), 0.88)
        # Min/Mean/Norm 不应出现在输出中（原始数据中也没有）
        # _update_stats 不会写入原始不存在的字段
        # (但如果 merged 中有且原始中没有，会写入)

    def test_build_dump_dict_missing_node_reconstructed(self):
        """验证不在 raw_data 中的节点（PP重命名）通过 _reconstruct_node_entry 重建。"""
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})

        # node3 (PP_Renamed_Layer) 不在 dump_data 中，应被重建
        self.assertIn('PP_Renamed_Layer', data, 'PP-renamed node should be reconstructed')
        pp_node = data['PP_Renamed_Layer']
        self.assertIn(Const.INPUT_ARGS, pp_node)
        self.assertEqual(len(pp_node[Const.INPUT_ARGS]), 1)
        self.assertEqual(pp_node[Const.INPUT_ARGS][0]['shape'], (2, 2))
        self.assertIn(Const.OUTPUT, pp_node)
        self.assertEqual(len(pp_node[Const.OUTPUT]), 1)

    def test_build_dump_dict_parallel_merge_info(self):
        """验证 parallel_merge_info 被正确补充。"""
        node = self.graph.node_map['Layer1']
        node.parallel_merge_info = ['tp_group_0']

        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})
        self.assertIn('parallel_merge_info', data['Layer1'])
        self.assertEqual(data['Layer1']['parallel_merge_info'], ['tp_group_0'])

    def test_build_dump_dict_empty_dump_data(self):
        """验证 dump_data 为 None 时仍能正确工作（全量节点从 input_data/output_data 重建）。"""
        self.graph.dump_data = None
        result = GraphJsonExporter._build_dump_dict(self.graph)
        data = result.get(GraphConst.DATA_KEY, {})

        # 有 input/output 的节点都应存在
        self.assertIn('Layer1', data)
        self.assertIn('Layer2', data)
        self.assertIn('PP_Renamed_Layer', data)
        # EmptyNode 没有 input/output，不应出现
        self.assertNotIn('EmptyNode', data)

    # ======================== _build_stack_dict ========================

    def test_build_stack_dict_grouping(self):
        """验证相同 stack_info 的节点被分到同一组。"""
        result = GraphJsonExporter._build_stack_dict(self.graph)

        # node4 和 node5 有相同 stack，应分到同一组
        groups = list(result.values())
        all_node_ids = []
        for group in groups:
            all_node_ids.extend(group[0])

        self.assertIn('LayerWithStack', all_node_ids)
        self.assertIn('LayerWithStack2', all_node_ids)

        # 找到包含这两个节点的组
        for group in groups:
            if 'LayerWithStack' in group[0] and 'LayerWithStack2' in group[0]:
                self.assertEqual(group[1], ['trace_line1', 'trace_line2'])
                break
        else:
            self.fail('LayerWithStack and LayerWithStack2 should be in the same stack group')

    def test_build_stack_dict_nodes_without_stack_skipped(self):
        """验证没有 stack_info 的节点不在 stack_dict 中。"""
        # 给 node1 加上 stack_info，这样 node1 也会出现在 stack_dict 中
        # node2/node3/node6 没有 stack_info
        result = GraphJsonExporter._build_stack_dict(self.graph)

        all_node_ids = []
        for group in result.values():
            all_node_ids.extend(group[0])

        # 有 stack_info 的节点应在其中
        self.assertIn('LayerWithStack', all_node_ids)
        self.assertIn('LayerWithStack2', all_node_ids)
        # 没有 stack_info 的节点不应出现
        self.assertNotIn('Layer1', all_node_ids)
        self.assertNotIn('Layer2', all_node_ids)
        self.assertNotIn('PP_Renamed_Layer', all_node_ids)
        self.assertNotIn('EmptyNode', all_node_ids)

    def test_build_stack_dict_format(self):
        """验证 stack_dict 输出格式：{str_idx: [[node_ids], [stack_trace]]}。"""
        result = GraphJsonExporter._build_stack_dict(self.graph)

        for key, value in result.items():
            # key 是数字字符串
            self.assertTrue(key.isdigit())
            # value 是长度为 2 的 list
            self.assertIsInstance(value, list)
            self.assertEqual(len(value), 2)
            # value[0] 是 node_id 列表
            self.assertIsInstance(value[0], list)
            self.assertTrue(all(isinstance(nid, str) for nid in value[0]))
            # value[1] 是 stack trace 列表
            self.assertIsInstance(value[1], list)
            self.assertTrue(all(isinstance(line, str) for line in value[1]))

    # ======================== _reconstruct_node_from_raw ========================

    def test_reconstruct_node_from_raw_no_stats(self):
        """验证原始节点重建（无统计信息覆盖）。"""
        raw_node = {
            Const.INPUT_ARGS: [{'shape': (4, 4), 'dtype': 'float32'}],
            Const.OUTPUT: [{'shape': (4, 4), 'dtype': 'float32'}],
        }

        def noop_update(item, _fname, _nid):
            return dict(item) if isinstance(item, dict) else item

        result = GraphJsonExporter._reconstruct_node_from_raw(raw_node, 'NodeX', noop_update)
        self.assertEqual(len(result[Const.INPUT_ARGS]), 1)
        self.assertEqual(result[Const.INPUT_ARGS][0]['shape'], (4, 4))

    def test_reconstruct_node_from_raw_with_stats(self):
        """验证原始节点重建时叠加统计信息。"""
        raw_node = {
            Const.INPUT_ARGS: [{'shape': (4, 4)}],
            Const.OUTPUT: [{'shape': (4, 4)}],
        }
        merged_stats = {'NodeX.input.0': {'Max': 1.0, 'Min': 0.0}}

        def stats_update(item, fname, _nid):
            result = dict(item)
            merged = merged_stats.get(fname)
            if merged:
                result.update(merged)
            return result

        result = GraphJsonExporter._reconstruct_node_from_raw(raw_node, 'NodeX', stats_update)
        self.assertEqual(result[Const.INPUT_ARGS][0]['Max'], 1.0)
        self.assertEqual(result[Const.INPUT_ARGS][0]['Min'], 0.0)
        # output 没有合并统计，不应有
        self.assertNotIn('Max', result[Const.OUTPUT][0])

    def test_reconstruct_node_from_raw_unknown_fields_preserved(self):
        """验证原始节点中的未知字段被原样保留。"""
        raw_node = {
            Const.INPUT_ARGS: [{'shape': (4, 4)}],
            'custom_field': 'custom_value',
            'extra_dict': {'nested': True},
        }

        def noop_update(item, _fname, _nid):
            return dict(item) if isinstance(item, dict) else item

        result = GraphJsonExporter._reconstruct_node_from_raw(raw_node, 'NodeX', noop_update)
        self.assertEqual(result['custom_field'], 'custom_value')
        self.assertEqual(result['extra_dict'], {'nested': True})

    # ======================== _reconstruct_node_entry ========================

    def test_reconstruct_node_entry_basic(self):
        """验证从 input_data/output_data 重建节点的基本功能。"""
        node = self.graph.node_map['PP_Renamed_Layer']
        result = GraphJsonExporter._reconstruct_node_entry(node, 'PP_Renamed_Layer')

        self.assertIn(Const.INPUT_ARGS, result)
        self.assertEqual(len(result[Const.INPUT_ARGS]), 1)
        self.assertEqual(result[Const.INPUT_ARGS][0]['shape'], (2, 2))
        self.assertIn(Const.OUTPUT, result)
        self.assertEqual(len(result[Const.OUTPUT]), 1)

    def test_reconstruct_node_entry_empty(self):
        """验证节点无 input/output 时返回空 dict。"""
        node = BaseNode(NodeOp.module, 'EmptyNode')
        result = GraphJsonExporter._reconstruct_node_entry(node, 'EmptyNode')
        self.assertEqual(result, {})

    def test_reconstruct_node_entry_input_kwargs(self):
        """验证 input_kwargs 被正确分类重建。"""
        node = BaseNode(NodeOp.module, 'KwargsLayer')
        node.input_data = {
            'k0': {'full_op_name': 'KwargsLayer.input.0', 'shape': (4, 4)},
            'k1': {'full_op_name': 'KwargsLayer.input.attention_mask', 'shape': (1, 8)},
        }
        node.output_data = {
            'k2': {'full_op_name': 'KwargsLayer.output.0', 'shape': (4, 4)},
        }

        result = GraphJsonExporter._reconstruct_node_entry(node, 'KwargsLayer')
        self.assertEqual(len(result[Const.INPUT_ARGS]), 1)
        self.assertEqual(result[Const.INPUT_ARGS][0]['shape'], (4, 4))
        self.assertIn(Const.INPUT_KWARGS, result)
        self.assertIn('attention_mask', result[Const.INPUT_KWARGS])
        self.assertEqual(len(result[Const.OUTPUT]), 1)

    def test_reconstruct_node_entry_params(self):
        """验证 parameters 被正确分类重建。"""
        node = BaseNode(NodeOp.module, 'ParamsLayer')
        node.input_data = {
            'p0': {'full_op_name': 'ParamsLayer.input.0', 'shape': (4, 4)},
            'p1': {'full_op_name': 'ParamsLayer.parameters.0', 'shape': (256, 256)},
        }
        node.output_data = {
            'p2': {'full_op_name': 'ParamsLayer.output.0', 'shape': (4, 4)},
        }

        result = GraphJsonExporter._reconstruct_node_entry(node, 'ParamsLayer')
        self.assertEqual(len(result[Const.INPUT_ARGS]), 1)
        self.assertIn(Const.PARAMS, result)
        self.assertEqual(len(result[Const.PARAMS]), 1)
        self.assertEqual(result[Const.PARAMS][0]['shape'], (256, 256))

    def test_reconstruct_node_entry_deduplication(self):
        """验证相同位置的数据不会被重复添加。"""
        node = BaseNode(NodeOp.module, 'DedupLayer')
        # 两个 key 指向相同位置
        node.input_data = {
            'dup1': {'full_op_name': 'DedupLayer.input.0', 'shape': (4, 4)},
            'dup2': {'full_op_name': 'DedupLayer.input.0', 'shape': (8, 8)},
        }
        node.output_data = {
            'dout1': {'full_op_name': 'DedupLayer.output.0', 'shape': (4, 4)},
        }

        result = GraphJsonExporter._reconstruct_node_entry(node, 'DedupLayer')
        # 只应有一个 input_args[0]
        self.assertEqual(len(result[Const.INPUT_ARGS]), 1)
        # 去重后保留的是先遇到的那一个
        self.assertEqual(result[Const.INPUT_ARGS][0]['shape'], (4, 4))

    def test_reconstruct_node_entry_malformed_suffix(self):
        """验证 full_op_name 格式异常的条目被跳过。"""
        node = BaseNode(NodeOp.module, 'MalformedLayer')
        node.input_data = {
            'good': {'full_op_name': 'MalformedLayer.input.0', 'shape': (4, 4)},
            # full_op_name 中没有 '.'
            'bad1': {'full_op_name': 'MalformedLayer_no_dot', 'shape': (8, 8)},
            # full_op_name 不以 node_id 开头，也不含 '.'
            'bad2': {'full_op_name': 'SomeOtherNode.input.0', 'shape': (16, 16)},
        }
        node.output_data = {
            'out': {'full_op_name': 'MalformedLayer.output.0', 'shape': (4, 4)},
        }

        result = GraphJsonExporter._reconstruct_node_entry(node, 'MalformedLayer')
        # 只有 good 和 out 被正确解析
        self.assertEqual(result.get(Const.INPUT_ARGS, [])[0]['shape'], (4, 4))
        self.assertEqual(len(result.get(Const.OUTPUT, [])), 1)
