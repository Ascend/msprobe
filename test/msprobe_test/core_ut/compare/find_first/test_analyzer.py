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

import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from msprobe.core.compare.find_first.analyzer import DiffAnalyzer
from msprobe.core.compare.find_first.utils import RankPath, FileCache, DiffAnalyseConst
from msprobe.core.compare.find_first.graph import DataNode, CommunicationNode
from msprobe.core.common.const import Const


class TestDiffAnalyzer(unittest.TestCase):
    def setUp(self):
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.npu_path = os.path.join(self.temp_dir, "npu")
        self.bench_path = os.path.join(self.temp_dir, "bench")
        self.output_path = os.path.join(self.temp_dir, "output")
        
        # 创建目录结构
        os.makedirs(self.npu_path, exist_ok=True)
        os.makedirs(self.bench_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 创建测试文件
        self.create_test_files()
        
        # 初始化分析器
        self.analyzer = DiffAnalyzer(self.npu_path, self.bench_path, self.output_path)
    
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
        # 重置FileCache单例
        FileCache._instance = None
    
    def create_test_files(self):
        # 创建比较结果文件
        compare_result_rank0 = os.path.join(self.output_path, "compare_result_rank0_123456.json")
        compare_result_rank1 = os.path.join(self.output_path, "compare_result_rank1_123456.json")
        
        # 创建测试数据
        rank0_data = {
            "Torch.add.1": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 1.0, "NPU_Min": 0.0, "NPU_Mean": 0.5, "NPU_Norm": 0.7, "Stack": [["Torch.add.1", {"file": "test.py", "line": 10}]]}
                ]
            },
            "Distributed.all_reduce.2": {
                "is_same": False,
                "op_items": [
                    {"NPU_Name": "input.0.dst", "NPU_Max": 1, "Stack": [["Distributed.all_reduce.2", {"file": "test.py", "line": 20}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 2.0, "Stack": "N/A"}
                ]
            },
            "Torch.mul.3": {
                "is_same": False,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 2.0, "Stack": [["Torch.mul.3", {"file": "test.py", "line": 30}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 4.0, "Stack": "N/A"}
                ]
            }
        }
        
        rank1_data = {
            "Torch.add.1": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 1.0, "Stack": [["Torch.add.1", {"file": "test.py", "line": 10}]]}
                ]
            },
            "Distributed.all_reduce.2": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0.src", "NPU_Max": 0, "Stack": [["Distributed.all_reduce.2", {"file": "test.py", "line": 20}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 2.0, "Stack": "N/A"}
                ]
            }
        }
        
        # 写入测试数据
        with open(compare_result_rank0, "w") as f:
            json.dump(rank0_data, f)
        
        with open(compare_result_rank1, "w") as f:
            json.dump(rank1_data, f)

    def test_pre_process_when_compare_and_resolve_then_pass(self):
        self.analyzer._pre_process()

        # 验证路径解析
        self.assertEqual(len(self.analyzer._paths), 2)  # 应该有两个rank路径
        self.assertIn(0, self.analyzer._paths)
        self.assertIn(1, self.analyzer._paths)
    
    def test_resolve_input_path_when_valid_files_then_pass(self):
        # 测试解析输入路径
        self.analyzer._resolve_input_path(self.output_path)
        
        # 验证路径解析
        self.assertEqual(len(self.analyzer._paths), 2)  # 应该有两个rank路径
        self.assertIn(0, self.analyzer._paths)
        self.assertIn(1, self.analyzer._paths)
        self.assertEqual(self.analyzer._paths[0].rank, 0)
        self.assertEqual(self.analyzer._paths[1].rank, 1)
    
    @patch.object(FileCache, 'load_json')
    def test_pre_analyze_when_diff_before_communication_then_pass(self, mock_load_json):
        # 模拟加载JSON数据
        mock_load_json.side_effect = lambda path: {
            "Torch.add.1.forward": {"is_same": False, "op_items": []},
            "Distributed.all_reduce.2.forward": {"is_same": True, "op_items": []}
        } if "rank0" in path else {
            "Torch.add.1.forward": {"is_same": True, "op_items": []},
            "Distributed.all_reduce.2.forward": {"is_same": True, "op_items": []}
        }
        
        # 设置路径
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json")),
            1: RankPath(1, os.path.join(self.output_path, "compare_result_rank1_123456.json"))
        }
        
        # 执行预分析
        self.analyzer._pre_analyze()
        
        # 验证结果
        self.assertEqual(len(self.analyzer._diff_nodes), 1)  # 应该找到一个异常节点
        self.assertEqual(self.analyzer._diff_nodes[0].op_name, "Torch.add.1.forward")
        self.assertEqual(self.analyzer._first_comm_nodes[1], "Distributed.all_reduce.2.forward")
    
    @patch.object(DiffAnalyzer, '_analyze_comm_nodes')
    @patch.object(DiffAnalyzer, '_connect_comm_nodes')
    @patch.object(DiffAnalyzer, '_pruning')
    @patch.object(DiffAnalyzer, '_search_first_diff')
    def test__analyze_when_paths_set_then_pass(self, mock_search, mock_pruning, mock_connect, mock_analyze_comm):
        # 模拟分析过程
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json")),
            1: RankPath(1, os.path.join(self.output_path, "compare_result_rank1_123456.json"))
        }
        
        # 执行分析
        self.analyzer._analyze()
        
        # 验证调用
        mock_analyze_comm.assert_called()
        mock_connect.assert_called_once()
        mock_pruning.assert_called_once()
        mock_search.assert_called_once()
    
    @patch.object(FileCache, 'load_json')
    def test__analyze_comm_nodes_when_communication_ops_exist_then_pass(self, mock_load_json):
        # 模拟加载JSON数据
        mock_load_json.return_value = {
            "Distributed.all_reduce.1.forward": {"is_same": False, "op_items": []},
            "Torch.add.2": {"is_same": True, "op_items": []},
            "Distributed.all_reduce.3.forward": {"is_same": True, "op_items": []}
        }
        
        # 设置首个通信节点
        self.analyzer._first_comm_nodes = {0: "Distributed.all_reduce.1.forward"}
        
        # 设置路径
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json"))
        }
        
        # 执行通信节点分析
        result = self.analyzer._analyze_comm_nodes(0)
        
        # 验证结果
        self.assertEqual(len(result), 2)  # 应该有两个通信节点
        self.assertIn("0.Distributed.all_reduce.1.forward", result)
        self.assertIn("0.Distributed.all_reduce.3.forward", result)
    
    def test__get_node_by_id_when_valid_and_invalid_then_pass(self):
        # 设置通信节点字典
        node = MagicMock(spec=CommunicationNode)
        self.analyzer._rank_comm_nodes_dict = {0: {"0.Distributed.all_reduce.1.forward": node}}
        
        # 测试获取节点
        result = self.analyzer._get_node_by_id("0.Distributed.all_reduce.1.forward")
        
        # 验证结果
        self.assertEqual(result, node)
        
        # 测试无效节点ID
        with self.assertRaises(RuntimeError):
            self.analyzer._get_node_by_id("invalid_id")
    
    @patch('msprobe.core.compare.find_first.analyzer.save_json')
    @patch('msprobe.core.compare.find_first.analyzer.create_directory')
    @patch('msprobe.core.compare.find_first.analyzer.time')
    def test__gen_analyze_info_when_diff_nodes_exist_then_pass(self, mock_time, mock_create_directory, mock_save_json):
        # 模拟时间戳
        mock_time.time_ns.return_value = 123456789
        
        # 设置异常节点
        node = MagicMock(spec=DataNode)
        node.rank = 0
        node.gen_node_info.return_value = {"op_name": "test_op"}
        self.analyzer._diff_nodes = [node]
        
        # 设置路径
        self.analyzer._paths = {0: MagicMock(spec=RankPath)}
        
        # 生成分析信息
        self.analyzer._gen_analyze_info()
        
        # 验证调用
        mock_save_json.assert_called_once()

    @patch.object(DiffAnalyzer, '_gen_analyze_info')
    @patch.object(DiffAnalyzer, '_post_analyze')
    @patch.object(DiffAnalyzer, '_analyze')
    @patch.object(DiffAnalyzer, '_pre_analyze')
    @patch.object(DiffAnalyzer, '_pre_process')
    def test_analyze_when_pre_analyze_has_diff_then_pass(self, mock_pre_process, mock_pre_analyze,
                                                         mock_analyze, mock_post_analyze, mock_gen_info):
        # 默认无 diff
        self.analyzer._diff_nodes = []

        # 伪造首个阶段找到 diff
        def side_effect():
            self.analyzer._diff_nodes.append(MagicMock())
        mock_pre_analyze.side_effect = side_effect

        self.analyzer.analyze()

        # 只执行预分析，后续阶段不再调用
        mock_pre_process.assert_called_once()
        mock_pre_analyze.assert_called_once()
        mock_analyze.assert_not_called()
        mock_post_analyze.assert_not_called()
        mock_gen_info.assert_called_once()

    @patch.object(DiffAnalyzer, '_gen_analyze_info')
    @patch.object(DiffAnalyzer, '_post_analyze')
    @patch.object(DiffAnalyzer, '_analyze')
    @patch.object(DiffAnalyzer, '_pre_analyze')
    @patch.object(DiffAnalyzer, '_pre_process')
    def test_analyze_when_no_diff_in_any_stage_then_pass(self, mock_pre_process, mock_pre_analyze,
                                                          mock_analyze, mock_post_analyze, mock_gen_info):
        # 各阶段都不写入 diff
        self.analyzer._diff_nodes = []

        self.analyzer.analyze()

        mock_pre_process.assert_called_once()
        mock_pre_analyze.assert_called_once()
        mock_analyze.assert_called_once()
        mock_post_analyze.assert_called_once()
        mock_gen_info.assert_not_called()

    def test__post_analyze_when_after_comm_diffs_exist_then_pass(self):
        # 模拟通信后异常节点
        node1_rank0 = MagicMock()
        node2_rank0 = MagicMock()
        node1_rank1 = MagicMock()
        self.analyzer._after_comm_diffs = {
            0: [node1_rank0, node2_rank0],
            1: [node1_rank1],
            2: []  # 空列表不应增加 diff
        }
        self.analyzer._diff_nodes = []

        self.analyzer._post_analyze()

        # 只添加各 rank 的第一个节点
        self.assertEqual([node1_rank0, node1_rank1], self.analyzer._diff_nodes)

    @patch.object(DiffAnalyzer, '_find_connection')
    def test__connect_comm_nodes_when_layers_overflow_then_pass(self, mock_find_connection):
        # 使 _find_connection 总返回 True，避免 debug 分支
        mock_find_connection.return_value = True

        # 模拟两个 rank，仅处理第一个 rank
        node1 = MagicMock()
        node1.node_id = f'0{Const.SEP}Distributed.all_reduce.1'
        node1.layer = 1
        node1.find_connected_nodes.return_value = {'api': 'Distributed.all_reduce'}

        node2 = MagicMock()
        node2.node_id = f'0{Const.SEP}Distributed.all_reduce.2'
        node2.layer = 0  # 会触发 overflow，被调整
        node2.find_connected_nodes.return_value = {'api': 'Distributed.all_reduce'}

        self.analyzer._rank_comm_nodes_dict = {
            0: {
                node1.node_id: node1,
                node2.node_id: node2
            },
            1: {}
        }

        self.analyzer._connect_comm_nodes()

        # 检查 ranks 被补全为所有 rank
        for call in mock_find_connection.call_args_list:
            conn_info = call[0][0]
            self.assertIn('ranks', conn_info)
            self.assertEqual(set(self.analyzer._rank_comm_nodes_dict.keys()), set(conn_info['ranks']))

        # 后一个节点层级被调整
        self.assertGreater(node2.layer, node1.layer)

    def test__find_connection_when_dst_src_link_then_pass(self):
        # 当前节点，初始为未连接
        cur_node = MagicMock()
        cur_node.node_id = f'0{Const.SEP}Distributed.all_reduce.1'
        cur_node.rank = 0
        cur_node.layer = 1
        cur_node.connected = False

        # DST 类型节点
        dst_node = MagicMock()
        dst_node.node_id = f'1{Const.SEP}Distributed.all_reduce.1'
        dst_node.type = DiffAnalyseConst.DST
        dst_node.api = 'Distributed.all_reduce'
        dst_node.find_connected_nodes.return_value = {'ranks': {0}}

        # SRC 类型节点
        src_node = MagicMock()
        src_node.node_id = f'1{Const.SEP}Distributed.all_reduce.2'
        src_node.type = DiffAnalyseConst.SRC
        src_node.api = 'Distributed.all_reduce'
        src_node.find_connected_nodes.return_value = {'ranks': {0}}

        # LINK 类型节点
        link_node = MagicMock()
        link_node.node_id = f'1{Const.SEP}Distributed.all_reduce.3'
        link_node.type = DiffAnalyseConst.LINK
        link_node.api = 'Distributed.all_reduce'
        link_node.find_connected_nodes.return_value = {'ranks': {0}}

        self.analyzer._rank_comm_nodes_dict = {
            0: {cur_node.node_id: cur_node},
            1: {
                dst_node.node_id: dst_node,
                src_node.node_id: src_node,
                link_node.node_id: link_node
            }
        }

        # DST 连接
        conn_info = {'ranks': {1}, 'api': 'Distributed.all_reduce', 'type': DiffAnalyseConst.DST}
        searched_ranks = set()
        seen_nodes = set()

        found = self.analyzer._find_connection(conn_info, cur_node, searched_ranks, seen_nodes)
        self.assertTrue(found)
        cur_node.add_dst.assert_called_once_with(dst_node)

        # LINK 连接
        conn_info_link = {'ranks': {1}, 'api': 'Distributed.all_reduce', 'type': DiffAnalyseConst.LINK}
        seen_nodes.clear()
        found_link = self.analyzer._find_connection(conn_info_link, cur_node, searched_ranks, seen_nodes)
        self.assertTrue(found_link)
        cur_node.add_link.assert_called_once_with(link_node)

        # SRC 连接，同时检查层级对齐
        conn_info_src = {'ranks': {1}, 'api': 'Distributed.all_reduce', 'type': DiffAnalyseConst.SRC}
        seen_nodes.clear()
        src_node.layer = 0
        found_src = self.analyzer._find_connection(conn_info_src, cur_node, searched_ranks, seen_nodes)
        self.assertTrue(found_src)
        self.assertEqual(src_node.layer, cur_node.layer)
        src_node.add_dst.assert_called_once_with(cur_node)

    def test__pruning_when_nodes_without_diff_then_pass(self):
        # is_diff 为 True 的不删除
        node_diff = MagicMock()
        node_diff.is_diff = True
        node_diff.compute_ops = []

        # compute_ops 不空的不删除
        node_with_compute = MagicMock()
        node_with_compute.is_diff = False
        node_with_compute.compute_ops = [MagicMock()]

        # 需要删除的节点
        node_to_delete = MagicMock()
        node_to_delete.is_diff = False
        node_to_delete.compute_ops = []

        self.analyzer._rank_comm_nodes_dict = {
            0: {
                '0.node_diff': node_diff,
                '0.node_with_compute': node_with_compute,
                '0.node_delete': node_to_delete
            }
        }

        self.analyzer._pruning()

        # 保留的节点还在
        nodes = self.analyzer._rank_comm_nodes_dict[0]
        self.assertIn('0.node_diff', nodes)
        self.assertIn('0.node_with_compute', nodes)
        # 删除的节点不存在
        self.assertNotIn('0.node_delete', nodes)
        node_to_delete.delete.assert_called_once()

    @patch('msprobe.core.compare.find_first.analyzer.analyze_diff_in_group')
    def test__search_first_diff_when_multi_diff_nodes_then_pass(self, mock_analyze_group):
        # 两个 rank 各有一个通信节点
        node_rank0 = MagicMock()
        node_rank0.node_id = f'0{Const.SEP}node0'
        node_rank0.layer = 0
        node_rank0.sub_layer = 1
        node_rank0.link_nodes = {}
        node_rank0.src_nodes = {}
        node_rank0.dst_nodes = {}

        node_rank1 = MagicMock()
        node_rank1.node_id = f'1{Const.SEP}node1'
        node_rank1.layer = 1
        node_rank1.sub_layer = 0
        node_rank1.link_nodes = {}
        node_rank1.src_nodes = {}
        node_rank1.dst_nodes = {}

        self.analyzer._rank_comm_nodes_dict = {
            0: {node_rank0.node_id: node_rank0},
            1: {node_rank1.node_id: node_rank1}
        }

        # analyze_diff_in_group 返回两个 diff 节点，层级不同
        diff_node1 = MagicMock()
        diff_node1.layer = 2
        diff_node1.sub_layer = 1

        diff_node2 = MagicMock()
        diff_node2.layer = 1
        diff_node2.sub_layer = 0

        mock_analyze_group.return_value = [diff_node1, diff_node2]

        self.analyzer._diff_nodes = []
        self.analyzer._search_first_diff()

        # 所有保留下来的节点都具有最小的 layer/sub_layer
        self.assertGreaterEqual(len(self.analyzer._diff_nodes), 1)
        for node in self.analyzer._diff_nodes:
            self.assertEqual((node.layer, node.sub_layer), (1, 0))


if __name__ == '__main__':
    unittest.main()
