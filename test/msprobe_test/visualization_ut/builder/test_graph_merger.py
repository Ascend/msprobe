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

import re
import unittest
from unittest.mock import patch, MagicMock, call, Mock
from msprobe.visualization.builder.graph_merger import (
    GraphMerger, BaseGraphMerger, PPMerger, TPMerger,
    NoParallelMerger, TPPPMerger, FullMerger, VPPMerger
)
from msprobe.core.common.const import Const
from msprobe.visualization.utils import GraphConst, ParallelParam
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.graph.graph import Graph, BaseNode
from msprobe.core.common.exceptions import MsprobeException


class TestGraphMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = MagicMock()
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=1)
        self.is_bench = False

    def test_select_strategy_no_parallel(self):
        self.parallel_param.tp = self.parallel_param.pp = self.parallel_param.rank_size = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, NoParallelMerger)

    def test_select_strategy_tp(self):
        self.parallel_param.tp = self.parallel_param.rank_size = 2
        self.parallel_param.pp = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, TPMerger)

    def test_select_strategy_pp(self):
        self.parallel_param.pp = self.parallel_param.rank_size = 2
        self.parallel_param.tp = 1
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, PPMerger)

    def test_select_strategy_tp_pp(self):
        self.parallel_param.tp = self.parallel_param.pp = 2
        self.parallel_param.rank_size = 4
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, TPPPMerger)

    def test_select_strategy_full(self):
        self.parallel_param.tp = 2
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        self.assertIsInstance(merger.strategy, FullMerger)

    def test_merge_graph(self):
        merger = GraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        merger.strategy.merge_graphs = MagicMock()
        merger.merge_graph()
        merger.strategy.merge_graphs.assert_called_once()


class TestBaseGraphMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(2)]
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=2)
        self.is_bench = False
        self.merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_sort_merged_api_collection(self):
        graph = MagicMock()
        root = MagicMock()
        graph.root = root
        subnode1 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS}.0", op=NodeOp.api_collection)
        subnode1.subnodes = [MagicMock(id="op_Rank1.0"), MagicMock(id="op_Rank0.0")]
        root.subnodes = [subnode1]
        self.merger.sort_merged_api_collection(graph)
        self.assertEqual([n.id for n in subnode1.subnodes], ["op_Rank0.0", "op_Rank1.0"])

    def test_update_node_data_key(self):
        data_dict = {
            "old_id.input.0": {"full_op_name": "old_id.op"},
            "other_key": {"value": "test"}
        }
        new_dict = self.merger._update_node_data_key("old_id", "new_id", data_dict)
        self.assertEqual(new_dict, {
            "new_id.input.0": {"full_op_name": "new_id.op"},
            "other_key": {"value": "test"}
        })

    def test_compare_value_same(self):
        self.assertTrue(self.merger._compare_value_same(1, 1))
        self.assertFalse(self.merger._compare_value_same(1, 2))
        self.assertTrue(self.merger._compare_value_same("a", "a"))
        self.assertTrue(self.merger._compare_value_same(1, 1.00000001, has_uncertainty=True))
        self.assertFalse(self.merger._compare_value_same(1, 1.1, has_uncertainty=True))

    def test_merge_graph_api_collection(self):
        results = [MagicMock() for _ in range(2)]
        graph0, graph1 = Graph("name1"), Graph("name2")
        results[0].graph, results[1].graph = graph0, graph1
        root0, root1 = MagicMock(), MagicMock()
        graph0.root, graph1.root = root0, root1
        node0 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES}.0")
        node0_sub1 = MagicMock(id="sub_op.0")
        node0.subnodes = [node0_sub1]
        node1 = MagicMock(id=f"{GraphConst.APIS_BETWEEN_MODULES}.0")
        node1_sub1 = MagicMock(id="sub_op.0")
        graph0.node_map = {f"{GraphConst.APIS_BETWEEN_MODULES}.0": node0}
        node1.subnodes = [node1_sub1]
        root0.subnodes = [node0]
        root1.subnodes = [node1]

        self.merger.merge_graph_api_collection(results)

        self.assertEqual(len(root0.subnodes), 1)
        self.assertTrue(root0.subnodes[0].id.startswith(GraphConst.APIS_BETWEEN_MODULES_ALL_RANKS))
        self.assertEqual(len(root0.subnodes[0].subnodes), 1)

    def test_split_graph_results_by_groups(self):
        groups = [[0, 1], [2, 3]]
        results = [MagicMock(rank=i) for i in range(4)]
        self.merger.build_graph_results = results
        split = self.merger.split_graph_results_by_groups(groups)
        self.assertEqual(len(split), 2)
        self.assertEqual([r.rank for r in split[0]], [0, 1])
        self.assertEqual([r.rank for r in split[1]], [2, 3])

    def test_compare_node_param_data(self):
        main_node = MagicMock()
        other_nodes = [MagicMock()]
        main_node.id = "id"
        other_nodes[0].id = "id"
        main_node.input_data = {"input.0": {Const.DTYPE: "torch.float16", Const.MAX: 1}}
        other_nodes[0].input_data = {"input.0": {Const.DTYPE: "torch.float16", Const.MAX: 2}}
        in_diff, out_diff = self.merger.compare_node_param_data(main_node, other_nodes)
        self.assertEqual(list(in_diff.keys()), ["input.0"])

    def test_compare_param_same(self):
        param1 = {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1}
        param2 = {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1}
        self.assertTrue(self.merger.compare_param_same(param1, param2))

        param2[Const.MAX] = 2
        self.assertFalse(self.merger.compare_param_same(param1, param2))

    def test_add_all_nodes_rank(self):
        graph0, graph1 = MagicMock(), MagicMock()
        node0, node1 = MagicMock(), MagicMock()
        graph0.node_map.values.return_value = [node0]
        graph1.node_map.values.return_value = [node1]
        self.build_graph_results[0].graph = graph0
        self.build_graph_results[1].graph = graph1

        self.merger._add_all_nodes_rank()

        self.assertEqual(node0.rank, 0)
        self.assertEqual(node1.rank, 1)

    def test_get_default_groups(self):
        self.parallel_param.tp = 4
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        tp_groups, pp_groups = merger.get_default_groups()
        self.assertEqual(tp_groups, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(pp_groups, [[0, 4], [1, 5], [2, 6], [3, 7]])

        self.parallel_param.tp = 2
        self.parallel_param.pp = 2
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        tp_groups, pp_groups = merger.get_default_groups()
        self.assertEqual(tp_groups, [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(pp_groups, [[0, 4], [1, 5], [2, 6], [3, 7]])

        self.parallel_param.tp = 2
        self.parallel_param.pp = 3
        self.parallel_param.rank_size = 8
        merger = BaseGraphMerger(self.build_graph_results, self.parallel_param, self.is_bench)
        with self.assertRaises(MsprobeException):
            merger.get_default_groups()


class TestPPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=1, pp=4, rank_size=4)
        self.is_bench = False
        self.merger = PPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_trace_p2p_mapping(self):
        p2p_mapping = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 4, 7: 5}
        chains = self.merger._trace_p2p_mapping(p2p_mapping)
        self.assertEqual(len(chains), 2)
        self.assertIn([0, 2, 4, 6], chains)
        self.assertIn([1, 3, 5, 7], chains)

    @patch('msprobe.visualization.builder.graph_merger.PPMerger._merge_nodes')
    def test_merge_nodes(self, mock_merge):
        main_graph = MagicMock()
        main_node = MagicMock(id="module.layers.0.forward")
        other_graphs = [MagicMock() for _ in range(3)]
        for i, g in enumerate(other_graphs):
            g.get_node.return_value = MagicMock(id=f"module.layers.{i}.forward")

        self.merger._merge_nodes(main_graph, main_node, other_graphs)
        mock_merge.assert_called()

    def test_merge_graphs(self):
        self.merger.get_groups = MagicMock(return_value=[[0, 1, 2, 3]])
        self.merger.merge_pp_graphs = MagicMock(return_value=self.build_graph_results[:1])
        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)

    def test_get_groups(self):
        for i, result in enumerate(self.build_graph_results):
            graph = MagicMock()
            node = MagicMock(id=f"Distributed.send.{i}.forward")
            node.input_data = {f"Distributed.send.{i}.forward.input.dst": {"value": (i + 1) % 4}}
            graph.node_map.values.return_value = [node]
            result.graph = graph

        groups = self.merger.get_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [0, 1, 2, 3])

    def test_merge_other_unique_nodes(self):
        main_graph = MagicMock()
        main_node = MagicMock()
        other_nodes = [MagicMock()]
        main_node.subnodes = [MagicMock(id="main_sub.0")]
        other_nodes[0].subnodes = [MagicMock(id="other_sub.0")]

        self.merger._merge_other_unique_nodes(main_graph, main_node, other_nodes)
        self.assertEqual(len(main_node.subnodes), 2)

    def test_sort_nodes(self):
        graph = MagicMock()
        start_node = MagicMock(id="module.layers.0.forward%0%0")
        start_node.op = NodeOp.module
        api_node = MagicMock(id="Torch.mul.forward.0%0%0")
        graph.node_map = {"module.layers.0.forward%0%0": start_node, "Torch.mul.forward.0%0%0": api_node}
        parent_node = MagicMock()
        parent_node.subnodes = [start_node, api_node]
        start_node.upnode = parent_node

        self.merger._sort_nodes(graph, start_node)
        self.assertEqual(parent_node.subnodes[0].id, "module.layers.0.forward")
        self.assertEqual(parent_node.subnodes[1].id, "Torch.mul_rank0.forward.0")

    def test_add_node_to_main_graph(self):
        graph = MagicMock()
        node = MagicMock()
        subnode = MagicMock()
        node.subnodes = [subnode]

        self.merger._add_node_to_main_graph(graph, node)
        graph.node_map.__setitem__.assert_has_calls([call(node.id, node), call(subnode.id, subnode)])

    def test_get_node_sort_rule(self):
        node = MagicMock(id="module.layers.0.forward%1%2")
        self.assertEqual(self.merger._get_node_sort_rule(node), (2, 1))
        self.assertEqual(self.merger._get_node_sort_rule(node, rank_ascending=False), (-2, 1))

    def test_mark_node_id_position_rank(self):
        node = MagicMock()
        parent_node = MagicMock()
        parent_node.subnodes = [MagicMock(), node, MagicMock()]
        node.upnode = parent_node
        node.id = "module.layers.0.forward"

        self.merger._mark_node_id_position_rank(node, 2)
        self.assertEqual(node.id, "module.layers.0.forward%1%2")

    def test_update_node_id(self):
        graph = MagicMock()
        start_node = MagicMock(id="module.layers.0.forward%1%2")
        start_node.op = NodeOp.module
        start_node.pp_index = 1
        graph.node_map = {start_node.id: start_node}

        self.merger._update_node_id(graph, start_node)
        self.assertEqual(start_node.id, "module.layers.1.forward")


class TestTPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=4, pp=1, rank_size=4)
        self.is_bench = False
        self.merger = TPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_merge_params(self):
        params = {
            "input.0": [
                {Const.MAX: 1, Const.MIN: 0, Const.MEAN: 0.5, Const.NORM: 1},
                {Const.MAX: 2, Const.MIN: 0, Const.MEAN: 0.7, Const.NORM: 1.2}
            ]
        }
        merge_info = self.merger._merge_params(params)
        self.assertIn("The Max value merging method for input.0 is: max(1, 2) = 2", merge_info)
        self.assertIn("The Mean value merging method for input.0 is: (0.5 + 0.7) / 2 = 0.6", merge_info)

    def test_get_need_merge_node(self):
        main_node = MagicMock(id="module.matmul_rank0.forward")
        other_graphs = [MagicMock() for _ in range(3)]
        tp_merge_mapping = {0: [1, 2, 3]}

        for i, g in enumerate(other_graphs):
            g.node_map = {f"module.matmul_rank{i + 1}.forward": MagicMock()}

        nodes = self.merger._get_need_merge_node(main_node, other_graphs, tp_merge_mapping)
        self.assertEqual(len(nodes), 0)

    def test_merge_graphs(self):
        self.merger.get_groups = MagicMock(return_value=[[0, 1, 2, 3]])
        self.merger.merge_tp_graphs = MagicMock(return_value=self.build_graph_results[:1])
        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)

    def test_get_groups(self):
        for i, result in enumerate(self.build_graph_results):
            graph = MagicMock()
            node = MagicMock(id=f"all_reduce.{i}")
            node.input_data = {f"all_reduce.{i}.input.group": {"group_ranks": [0, 1, 2, 3]}}
            graph.node_map.values.return_value = [node]
            result.graph = graph

        groups = self.merger.get_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [0, 1, 2, 3])

    def test_handle_tp_matmul_reduce(self):
        node = MagicMock(id=f"module.RowParallelLinear.forward.0")
        node.op = NodeOp.module
        matmul_node = MagicMock(id="matmul.0")
        matmul_node.output_data = {"output.0": {Const.MAX: 1}}
        reduce_node = MagicMock(id="all_reduce.0")
        reduce_node.input_data = {"input.0": {Const.MAX: 1}}
        reduce_node.output_data = {"output.0": {Const.MAX: 2}}
        node.subnodes = [matmul_node, reduce_node]
        other_graphs = [MagicMock()]

        self.merger._handle_tp_matmul_reduce(node, other_graphs, {})
        self.assertEqual(matmul_node.output_data["output.0"][Const.MAX], 2)


class TestNoParallelMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock()]
        self.parallel_param = ParallelParam(tp=1, pp=1, rank_size=1)
        self.is_bench = False
        self.merger = NoParallelMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    def test_merge_graphs(self):
        self.merger.merge_graph_api_collection = MagicMock()
        results = self.merger.merge_graphs()
        self.assertEqual(results, self.build_graph_results)
        self.merger.merge_graph_api_collection.assert_called_once_with(self.build_graph_results)


class TestTPPPMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(4)]
        self.parallel_param = ParallelParam(tp=2, pp=2, rank_size=4)
        self.is_bench = False
        self.merger = TPPPMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    @patch('msprobe.visualization.builder.graph_merger.TPMerger')
    @patch('msprobe.visualization.builder.graph_merger.PPMerger')
    def test_merge_graphs(self, mock_pp, mock_tp):
        tp_merger = MagicMock()
        pp_merger = MagicMock()
        mock_tp.return_value = tp_merger
        mock_pp.return_value = pp_merger

        pp_merger.get_groups.return_value = [[0, 1], [2, 3]]
        tp_merger.get_groups.return_value = [[0, 2], [1, 3]]
        tp_merger.merge_tp_graphs.return_value = [MagicMock()]

        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)


class TestFullMerger(unittest.TestCase):
    def setUp(self):
        self.build_graph_results = [MagicMock(rank=i) for i in range(8)]
        self.parallel_param = ParallelParam(tp=2, pp=4, rank_size=8, vpp=1)
        self.is_bench = False
        self.merger = FullMerger(self.build_graph_results, self.parallel_param, self.is_bench)

    @patch('msprobe.visualization.builder.graph_merger.TPMerger')
    @patch('msprobe.visualization.builder.graph_merger.PPMerger')
    def test_merge_graphs(self, mock_pp, mock_tp):
        tp_merger = MagicMock()
        pp_merger = MagicMock()
        mock_tp.return_value = tp_merger
        mock_pp.return_value = pp_merger

        pp_merger.get_groups.return_value = [[0, 1, 2, 3], [4, 5, 6, 7]]
        tp_merger.get_groups.return_value = [[0, 4], [1, 5], [2, 6], [3, 7]]

        pp_result0 = MagicMock(rank=0)
        pp_result1 = MagicMock(rank=4)
        pp_merger.merge_pp_graphs.side_effect = [[pp_result0], [pp_result1]]

        tp_merger.merge_tp_graphs.side_effect = [[MagicMock()], [MagicMock()]]

        results = self.merger.merge_graphs()
        self.assertEqual(len(results), 1)


class TestVPPMerger(unittest.TestCase):
    def setUp(self):
        """初始化测试环境，完整模拟父类初始化逻辑"""
        # 1. 模拟并行参数
        self.parallel_param = Mock()
        self.parallel_param.vpp = 3  # 3个VPP chunk: 0,1,2

        # 2. 模拟build_graph_results
        self.build_graph_results = [
            Mock(graph=Graph(model_name="test_model_0")),
            Mock(graph=Graph(model_name="test_model_1"))
        ]

        # 3. 创建VPPMerger实例（完整初始化父类）
        with patch.object(BaseGraphMerger, '_add_all_nodes_rank') as mock_add_rank:
            self.merger = VPPMerger(
                build_graph_results=self.build_graph_results,
                parallel_param=self.parallel_param,
                is_bench=False
            )
            # 验证父类初始化逻辑
            mock_add_rank.assert_called_once()

        # 4. 验证父类属性初始化
        self.assertEqual(self.merger.unmerged_module, [Const.CLIP_GRAD, Const.OPTIMIZER])
        self.assertEqual(self.merger.dtype_list,
                         Const.TORCH_INT_DTYPE + Const.TORCH_FLOAT_DTYPE + [Const.FLOAT16, Const.FLOAT32,
                                                                            Const.BFLOAT16])
        self.assertEqual(self.merger.build_graph_results, self.build_graph_results)
        self.assertEqual(self.merger.parallel_param, self.parallel_param)
        self.assertEqual(self.merger.is_bench, False)
        self.assertEqual(self.merger.log_prefix, '[NPU]')

        # 5. 验证VPPMerger类属性
        self.assertEqual(self.merger.LAYERS_NUM_PATTERN, re.compile(r"(layers\.|layer\.)(\d+)(\.)"))
        self.assertEqual(self.merger.FORWARD_PATTERN, re.compile(r'\.forward\.\d+$'))

        # 6. 构建测试用Graph和Node
        self.graph = self._create_test_graph()
        self.test_results = self._create_test_results()

    @staticmethod
    def _create_base_node(node_id, op=NodeOp.module, up_node=None):
        """创建BaseNode实例，完整初始化属性"""
        node = BaseNode(op, node_id, up_node)
        node.input_data = {}
        node.output_data = {}
        node.subnodes = []
        node.pp_index = 0
        node.rank = 0  # 父类_add_all_nodes_rank会添加的属性
        return node

    def _create_test_graph(self):
        """创建包含多VPP chunk的测试用Graph实例"""
        graph = Graph(model_name="test_model")
        graph.rank = 0

        # 创建不同chunk的模块节点
        # Chunk 0
        chunk0_forward = self._create_base_node("Module.0.forward.0", up_node=graph.root)
        chunk0_backward = self._create_base_node("Module.0.backward.0", up_node=graph.root)
        # Chunk 1
        chunk1_forward = self._create_base_node("Module.1.forward.0", up_node=graph.root)
        chunk1_backward = self._create_base_node("Module.1.backward.0", up_node=graph.root)
        # Chunk 2
        chunk2_forward = self._create_base_node("Module.2.forward.0", up_node=graph.root)
        chunk2_backward = self._create_base_node("Module.2.backward.0", up_node=graph.root)

        # 为前向节点添加layers子节点
        chunk0_forward.subnodes = [
            self._create_base_node("Module.0.forward.0.layers.0.conv", up_node=chunk0_forward),
            self._create_base_node("Module.0.forward.0.layers.1.relu", up_node=chunk0_forward)
        ]
        chunk1_forward.subnodes = [
            self._create_base_node("Module.1.forward.0.layers.0.fc", up_node=chunk1_forward),
            self._create_base_node("Module.1.forward.0.layers.1.bn", up_node=chunk1_forward)
        ]
        chunk2_forward.subnodes = [
            self._create_base_node("Module.2.forward.0.layers.0.pool", up_node=chunk2_forward),
            self._create_base_node("Module.2.forward.0.layers.1.dropout", up_node=chunk2_forward)
        ]

        # 构建图节点映射
        graph.node_map = {
            chunk0_forward.id: chunk0_forward,
            chunk0_backward.id: chunk0_backward,
            chunk1_forward.id: chunk1_forward,
            chunk1_backward.id: chunk1_backward,
            chunk2_forward.id: chunk2_forward,
            chunk2_backward.id: chunk2_backward,
            **{n.id: n for n in chunk0_forward.subnodes},
            **{n.id: n for n in chunk1_forward.subnodes},
            **{n.id: n for n in chunk2_forward.subnodes}
        }
        graph.root.subnodes = [chunk0_forward, chunk0_backward, chunk1_forward, chunk1_backward, chunk2_forward,
                               chunk2_backward]

        return graph

    def _create_test_results(self):
        """创建模拟的build_graph_results格式测试数据"""
        result1 = Mock()
        result1.graph = self.graph
        result2 = Mock()
        result2.graph = self.graph  # 模拟多个VPP chunk的图
        result3 = Mock()
        result3.graph = self.graph
        return [result1, result2, result3]

    # ------------------------ 测试父类初始化 ------------------------
    def test_base_init(self):
        """测试BaseGraphMerger初始化逻辑"""
        # 测试bench模式
        with patch.object(BaseGraphMerger, '_add_all_nodes_rank') as mock_add_rank:
            bench_merger = VPPMerger(
                build_graph_results=self.build_graph_results,
                parallel_param=self.parallel_param,
                is_bench=True
            )
            self.assertEqual(bench_merger.log_prefix, '[Bench]')
            self.assertEqual(bench_merger.is_bench, True)
            mock_add_rank.assert_called_once()

        # 测试NPU模式
        self.assertEqual(self.merger.log_prefix, '[NPU]')
        self.assertEqual(self.merger.is_bench, False)

        # 验证unmerged_module初始化
        self.assertEqual(
            self.merger.unmerged_module,
            [Const.CLIP_GRAD, Const.OPTIMIZER]
        )

        # 验证dtype_list初始化
        expected_dtype = Const.TORCH_INT_DTYPE + Const.TORCH_FLOAT_DTYPE + [Const.FLOAT16, Const.FLOAT32,
                                                                            Const.BFLOAT16]
        self.assertEqual(self.merger.dtype_list, expected_dtype)

    # ------------------------ 测试静态方法 ------------------------
    def test_replace_vpp_id(self):
        """测试VPP ID替换逻辑"""
        # 正常替换场景
        self.assertEqual(
            VPPMerger._replace_vpp_id("Module.1.forward.0", 2),
            "Module.2.forward.0"
        )
        self.assertEqual(
            VPPMerger._replace_vpp_id("Layer.5.backward.1", 0),
            "Layer.0.backward.1"
        )
        self.assertEqual(
            VPPMerger._replace_vpp_id("Module.99.forward.100", 50),
            "Module.50.forward.100"
        )

        # 边界场景 - 无效格式
        # 无分隔符
        self.assertEqual(
            VPPMerger._replace_vpp_id("InvalidNode", 2),
            "InvalidNode"
        )
        # 第二部分非数字
        self.assertEqual(
            VPPMerger._replace_vpp_id("Module.abc.forward.0", 2),
            "Module.abc.forward.0"
        )
        # 格式过短（仅两部分）
        self.assertEqual(
            VPPMerger._replace_vpp_id("Module.1", 2),
            "Module.2"
        )
        # 空字符串
        self.assertEqual(
            VPPMerger._replace_vpp_id("", 2),
            ""
        )

    # ------------------------ 测试merge_pp_graphs ------------------------
    def test_merge_pp_graphs_empty(self):
        """测试空/单元素results场景"""
        # 空列表
        self.assertEqual(self.merger.merge_pp_graphs([]), [])

        # 单元素列表
        single_result = [Mock(graph=self.graph)]
        self.assertEqual(self.merger.merge_pp_graphs(single_result), single_result)

    def test_merge_pp_graphs_normal(self):
        """测试正常合并流程"""
        # Mock父类/子类的依赖方法
        with patch.object(self.merger, '_merge_nodes') as mock_merge, \
                patch.object(self.merger, '_sort_nodes') as mock_sort, \
                patch.object(self.merger, '_merge_vpp_data') as mock_data, \
                patch.object(self.merger, '_merge_vpp_chunks') as mock_chunks:
            # 执行合并
            result = self.merger.merge_pp_graphs(self.test_results)

            # 验证返回结果
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], self.test_results[0])

            # 验证方法调用次数和参数
            main_nodes = [n for n in self.test_results[0].graph.root.subnodes if n.op == NodeOp.module]
            self.assertEqual(mock_merge.call_count, len(main_nodes))
            self.assertEqual(mock_sort.call_count, len(main_nodes))

            # 验证第一个节点的调用参数
            first_main_node = main_nodes[0]
            mock_merge.assert_any_call(self.graph, first_main_node, [self.graph, self.graph])
            mock_sort.assert_any_call(self.graph, first_main_node)

            # 验证数据合并和chunk合并方法调用
            mock_data.assert_called_once_with(self.graph)
            mock_chunks.assert_called_once_with(self.graph)

    # ------------------------ 测试_merge_vpp_data ------------------------
    def test_merge_vpp_data_empty(self):
        """测试空模块列表场景"""
        # 清空graph.root.subnodes
        empty_graph = Graph(model_name="empty")
        empty_graph.root.subnodes = []
        # 执行无异常
        self.merger._merge_vpp_data(empty_graph)

    def test_merge_vpp_data_forward(self):
        """测试前向节点数据合并逻辑"""
        # 准备目标节点（最后一个chunk）
        target_node_id = "Module.2.forward.0"
        target_node = self._create_base_node(target_node_id)
        target_node.output_data = {
            f"{target_node_id}.output.0": {"shape": [1, 2], "dtype": "float32"},
            f"{target_node_id}.output.1": {"shape": [3, 4]}
        }
        self.graph.node_map[target_node_id] = target_node

        # 准备当前节点（chunk0）
        current_node_id = "Module.0.forward.0"
        current_node = self.graph.node_map[current_node_id]
        current_node.output_data = {}

        del target_node.pp_index
        del current_node.pp_index

        # Mock数据更新方法
        with patch.object(self.merger, '_update_node_data_key') as mock_update:
            mock_update.side_effect = lambda old_id, new_id, data: {
                k.replace(old_id, new_id): v for k, v in data.items()
            }

            # 执行数据合并
            self.merger._merge_vpp_data(self.graph)

            # 验证输出数据更新
            self.assertEqual(
                current_node.output_data,
                {
                    f"{current_node_id}.output.0": {"shape": [1, 2], "dtype": "float32"},
                    f"{current_node_id}.output.1": {"shape": [3, 4]}
                }
            )

    def test_merge_vpp_data_backward(self):
        """测试反向节点数据合并逻辑"""
        # 准备目标节点（最后一个chunk）
        target_node_id = "Module.2.backward.0"
        target_node = self._create_base_node(target_node_id)
        target_node.input_data = {
            f"{target_node_id}.input.0": {"dtype": "float32"},
            f"{target_node_id}.input.1": {"shape": [10, 20]}
        }
        self.graph.node_map[target_node_id] = target_node

        # 准备当前节点（chunk0）
        current_node_id = "Module.0.backward.0"
        current_node = self.graph.node_map[current_node_id]
        current_node.input_data = {}

        del target_node.pp_index
        del current_node.pp_index

        # Mock数据更新方法
        with patch.object(self.merger, '_update_node_data_key') as mock_update:
            mock_update.side_effect = lambda old_id, new_id, data: {
                k.replace(old_id, new_id): v for k, v in data.items()
            }

            # 执行数据合并
            self.merger._merge_vpp_data(self.graph)

            # 验证输入数据更新
            self.assertEqual(
                current_node.input_data,
                {
                    f"{current_node_id}.input.0": {"dtype": "float32"},
                    f"{current_node_id}.input.1": {"shape": [10, 20]}
                }
            )

    # ------------------------ 测试_merge_vpp_chunks ------------------------
    def test_merge_vpp_chunks_empty(self):
        """测试空chunk0列表场景"""
        empty_graph = Graph(model_name="empty")
        empty_graph.root.subnodes = []
        # 执行无异常
        self.merger._merge_vpp_chunks(empty_graph)

    def test_merge_vpp_chunks_forward(self):
        """测试前向节点chunk合并（子节点追加）"""
        # 获取测试节点
        chunk0_forward = self.graph.node_map["Module.0.forward.0"]
        chunk1_forward = self.graph.node_map["Module.1.forward.0"]
        chunk2_forward = self.graph.node_map["Module.2.forward.0"]

        # 记录初始状态
        initial_subnodes = chunk0_forward.subnodes.copy()
        initial_subnode_count = len(initial_subnodes)

        # 执行chunk合并
        self.merger._merge_vpp_chunks(self.graph)

        # # 验证子节点合并结果
        self.assertEqual(
            len(chunk0_forward.subnodes),
            initial_subnode_count
        )
        self.assertEqual(
            chunk0_forward.subnodes[:initial_subnode_count],
            initial_subnodes
        )

        # 验证父节点更新
        for sub_node in chunk1_forward.subnodes + chunk2_forward.subnodes:
            self.assertEqual(sub_node.upnode, chunk0_forward)

    def test_merge_vpp_chunks_backward(self):
        """测试反向节点chunk合并（子节点前置）"""
        # 准备反向节点数据
        chunk0_backward = self.graph.node_map["Module.0.backward.0"]
        chunk0_backward.subnodes = [
            self._create_base_node("Module.0.backward.0.layers.0.grad", up_node=chunk0_backward)]

        chunk1_backward = self._create_base_node("Module.1.backward.0")
        chunk1_backward.subnodes = [
            self._create_base_node("Module.1.backward.0.layers.0.grad", up_node=chunk1_backward)]
        self.graph.node_map["Module.1.backward.0"] = chunk1_backward
        self.graph.root.subnodes.append(chunk1_backward)

        chunk2_backward = self._create_base_node("Module.2.backward.0")
        chunk2_backward.subnodes = [
            self._create_base_node("Module.2.backward.0.layers.0.grad", up_node=chunk2_backward)]
        self.graph.node_map["Module.2.backward.0"] = chunk2_backward
        self.graph.root.subnodes.append(chunk2_backward)

        # 执行chunk合并
        self.merger._merge_vpp_chunks(self.graph)


    def test_sort_layers_forward(self):
        """测试前向layers重排序"""
        # 准备待排序节点列表
        node_list = (
                self.graph.node_map["Module.0.forward.0"].subnodes +
                self.graph.node_map["Module.1.forward.0"].subnodes +
                self.graph.node_map["Module.2.forward.0"].subnodes
        )

        # 执行排序
        self.merger._sort_layers(node_list, self.graph, is_forward=True)

        # 验证layers序号重排
        layer_index = 0
        for node in node_list:
            if "layers." in node.id:
                # 验证layers序号递增
                self.assertIn(f"layers.{layer_index}.", node.id)
                # 验证chunk号改为0
                self.assertIn(".0.", node.id)
                # 验证节点在图映射中
                self.assertIn(node.id, self.graph.node_map)
                layer_index += 1

    def test_sort_layers_backward(self):
        """测试反向layers重排序（先反转再排序）"""
        # 准备反向layers节点
        backward_nodes = [
            self._create_base_node("Module.1.backward.0.layers.0.grad"),
            self._create_base_node("Module.2.backward.0.layers.1.grad"),
            self._create_base_node("Module.0.backward.0.layers.2.grad")
        ]

        # 执行反向排序
        self.merger._sort_layers(backward_nodes, self.graph, is_forward=False)

        # 验证节点顺序反转且序号重排
        self.assertIn("layers.2.grad", backward_nodes[0].id)
        self.assertIn("layers.1.grad", backward_nodes[1].id)
        # 验证chunk号改为0
        for node in backward_nodes:
            self.assertIn(".0.", node.id)

    def test_sort_layers_subnodes_recursive(self):
        """测试子节点递归重命名"""
        # 创建嵌套子节点
        parent_node = self._create_base_node("Module.1.forward.0.layers.0.conv")
        child_node = self._create_base_node("Module.1.forward.0.layers.0.conv.weight", up_node=parent_node)
        grandchild_node = self._create_base_node("Module.1.forward.0.layers.0.conv.bias", up_node=child_node)
        parent_node.subnodes = [child_node]
        child_node.subnodes = [grandchild_node]
        node_list = [parent_node]

        # 执行排序
        self.merger._sort_layers(node_list, self.graph, is_forward=True)

        # 验证递归重命名
        self.assertEqual(parent_node.id, "Module.0.forward.0.layers.0.conv")
        self.assertEqual(child_node.id, "Module.0.forward.0.layers.0.conv.weight")
        self.assertEqual(grandchild_node.id, "Module.0.forward.0.layers.0.conv.bias")
        # 验证图映射更新
        self.assertIn(parent_node.id, self.graph.node_map)
        self.assertIn(child_node.id, self.graph.node_map)
        self.assertIn(grandchild_node.id, self.graph.node_map)

    def test_sort_layers_no_layers_pattern(self):
        """测试非layers层节点重命名"""
        # 创建无layers模式的节点
        no_layer_node = self._create_base_node("Module.1.forward.0.conv.0")
        node_list = [no_layer_node]

        # 执行排序
        self.merger._sort_layers(node_list, self.graph, is_forward=True)

        # 验证仅替换chunk号，不修改layers序号
        self.assertEqual(no_layer_node.id, "Module.0.forward.0.conv.0")
        self.assertNotIn("layers.", no_layer_node.id)


if __name__ == '__main__':
    unittest.main()
