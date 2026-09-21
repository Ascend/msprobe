# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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
import unittest
from typing import Any
from dataclasses import dataclass
from unittest.mock import patch
from unittest.mock import MagicMock
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.graph.graph import Graph, BaseNode, NodeOp
from msprobe.visualization.utils import GraphConst


@dataclass
class Args:
    input_path: str = None
    output_path: str = None
    layer_mapping: Any = None
    framework: str = None
    overflow_check: bool = False
    fuzzy_match: bool = False


class TestGraphComparator(unittest.TestCase):

    def setUp(self):
        self.current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.input = os.path.join(self.current_path, "input_format_correct")
        self.output = os.path.join(self.current_path, 'output')
        self.dump_path_param = {
            'npu_path': os.path.join(self.input, 'step0', 'rank0', 'dump.json'),
            'bench_path': os.path.join(self.input, 'step0', 'rank0', 'dump.json'),
            'stack_json_path': os.path.join(self.input, 'step0', 'rank0', 'stack.json'),
            'is_print_compare_log': True
        }
        self.graphs = [Graph("model1"), Graph("model2")]
        self.output_path = "output/output.vis"

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test_compare(self, mock_get_compare_mode):
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        comparator._compare_nodes = MagicMock()
        comparator._postcompare = MagicMock()

        comparator.compare()

        comparator._compare_nodes.assert_called_once()
        comparator._postcompare.assert_called_once()

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test_add_compare_result_to_node(self, mock_get_compare_mode):
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        node = MagicMock()
        compare_result_list = [("output1", "data1"), ("input1", "data2")]

        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        comparator.ma = MagicMock()
        comparator.ma.prepare_real_data.return_value = True

        comparator.add_compare_result_to_node(node, compare_result_list)
        comparator.ma.prepare_real_data.assert_called_once_with(node)
        node.data.update.assert_not_called()

    @patch('msprobe.visualization.graph.node_colors.NodeColors.get_node_error_status')
    @patch('msprobe.visualization.compare.graph_comparator.get_csv_df')
    @patch('msprobe.visualization.compare.graph_comparator.run_real_data')
    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test__postcompare(self, mock_get_compare_mode, mock_run_real_data, mock_get_csv_df, mock_get_node_error_status):
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        mock_df = MagicMock()
        mock_df.iterrows = MagicMock(return_value=[(None, MagicMock())])
        mock_run_real_data.return_value = mock_df
        mock_get_csv_df.return_value = mock_df
        mock_get_node_error_status.return_value = True
        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        comparator.ma = MagicMock()
        comparator.ma.compare_mode = GraphConst.REAL_DATA_COMPARE
        comparator._handle_api_collection_index = MagicMock()
        comparator.ma.compare_nodes = [MagicMock()]
        comparator.ma.parse_result = MagicMock(return_value=(0.9, None))

        comparator._postcompare()

        comparator._handle_api_collection_index.assert_called_once()

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test__handle_api_collection_index(self, mock_get_compare_mode):
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        apis = BaseNode(NodeOp.api_collection, 'Apis_Between_Modules.0')
        api1 = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        api1.data = {GraphConst.JSON_INDEX_KEY: 0.9}
        api2 = BaseNode(NodeOp.function_api, 'Tensor.b.0')
        api2.data = {GraphConst.JSON_INDEX_KEY: 0.6}
        apis.subnodes = [api1, api2]
        sub_nodes = [BaseNode(NodeOp.module, 'Module.a.0'), apis, BaseNode(NodeOp.module, 'Module.a.1')]
        comparator.graph_n.root.subnodes = sub_nodes
        comparator._handle_api_collection_index()
        self.assertEqual(comparator.graph_n.root.subnodes[1].data.get(GraphConst.JSON_INDEX_KEY), 0.9)

    @patch('msprobe.visualization.builder.msprobe_adapter.compare_node')
    @patch('msprobe.visualization.graph.graph.Graph.match')
    @patch('msprobe.visualization.graph.graph.Graph.mapping_match')
    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test__compare_nodes(self, mock_get_compare_mode, mock_mapping_match, mock_match, mock_compare_node):
        node_n = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        node_b = BaseNode(NodeOp.function_api, 'Tensor.b.0')
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        mock_mapping_match.return_value = (node_b, [], [])
        mock_compare_node.return_value = ['result']
        comparator = GraphComparator(self.graphs, self.dump_path_param,
                                     Args(output_path=self.output_path, layer_mapping=True), True)
        comparator.mapping_dict = True
        comparator._compare_nodes(node_n)
        self.assertEqual(node_n.matched_node_link, ['Tensor.b.0'])
        self.assertEqual(node_b.matched_node_link, ['Tensor.a.0'])
        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        comparator.mapping_dict = False
        node_n = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        node_b = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        mock_match.return_value = (node_b, [])
        comparator._compare_nodes(node_n)
        self.assertEqual(node_n.matched_node_link, ['Tensor.a.0'])
        self.assertEqual(node_b.matched_node_link, ['Tensor.a.0'])

    @patch('msprobe.visualization.builder.msprobe_adapter.compare_node')
    @patch('msprobe.visualization.graph.graph.Graph.match')
    @patch('msprobe.visualization.graph.graph.Graph.fuzzy_match')
    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    def test_compare_nodes_fuzzy(self, mock_get_compare_mode, mock_fuzzy_match, mock_match, mock_compare_node):
        node_n = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        node_b = BaseNode(NodeOp.function_api, 'Tensor.b.0')
        node_module_n = BaseNode(NodeOp.module, 'Module.a.0')
        node_module_n.subnodes = [node_n]
        node_n.upnode = node_module_n
        node_module_b = BaseNode(NodeOp.module, 'Module.b.0')
        node_module_b.subnodes = [node_b]
        node_b.upnode = node_module_b
        self.graphs[0].node_map[node_n.id] = node_n
        self.graphs[1].node_map[node_b.id] = node_b
        self.graphs[0].node_map[node_module_n.id] = node_module_n
        self.graphs[1].node_map[node_module_b.id] = node_module_b
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        mock_fuzzy_match.return_value = (node_module_b, [], [])
        mock_compare_node.return_value = ['result']
        comparator = GraphComparator(self.graphs, self.dump_path_param,
                                     Args(output_path=self.output_path, layer_mapping=True), True)
        comparator.mapping_dict = True
        comparator._compare_nodes_fuzzy(node_module_n)
        self.assertEqual(node_n.matched_node_link, ['Module.b.0', 'Module.b.0'])
        self.assertEqual(node_b.matched_node_link, [])


    def test_add_compare_result_node(self):
        compare_result_list = [
            ['Tensor.__truediv__.139.backward.input.0', 'Tensor.__truediv__.139.backward.input.0', 'torch.float32',
             'torch.float32', [], [], 'False', 'False', 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, True, '', '', 'None'],
            ['Tensor.__truediv__.139.backward.output.0', 'Tensor.__truediv__.139.backward.output.0', 'torch.float32',
             'torch.float32', [], [], 'False', 'False', 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%',
             0.25, 0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625,
             0.000244140625, 0.000244140625, True, '', '', 'None']
        ]
        node = BaseNode(NodeOp.module, 'Module.module.Float16Module.forward.0')
        comparator = GraphComparator(self.graphs, self.dump_path_param, Args(output_path=self.output_path), False)
        comparator.add_compare_result_to_node(node, compare_result_list)
        self.assertEqual(node.data, {'precision_index': 0})
