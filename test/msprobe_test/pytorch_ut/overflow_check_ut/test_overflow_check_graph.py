#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

import unittest
from unittest.mock import Mock, patch
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msprobe.overflow_check.graph import DataNode, CommunicationNode


class TestDataNode(unittest.TestCase):
    """DataNode 单元测试类"""

    def setUp(self):
        """测试前准备"""
        self.op_data = {
            'inputs': [{'type': 'tensor', 'value': 1.0}],
            'input_args': [{'type': 'float', 'value': 0.5}],
            'input_kwargs': {'lr': {'type': 'float', 'value': 0.01}},
            'output': {'result': {'type': 'tensor', 'value': 2.0}}
        }

    def test_init_with_custom_sub_layer(self):
        """测试DataNode初始化自定义sub_layer"""
        data_node = DataNode('test_op', 0, self.op_data, sub_layer=5)
        self.assertEqual(data_node.sub_layer, 5)

    def test_find_complete_construct_simple(self):
        """测试查找完整构造链(简单情况)"""
        construct_info = {
            'op1': 'op2',
            'op2': 'op3',
            'op3': None
        }
        result = DataNode.find_complete_construct(construct_info, 'op1')
        self.assertEqual(result, ['op3', 'op2', 'op1'])

    def test_find_complete_construct_with_list(self):
        """测试查找完整构造链(包含列表)"""
        construct_info = {
            'op1': ['op2', 'extra'],
            'op2': 'op3',
            'op3': None
        }
        result = DataNode.find_complete_construct(construct_info, 'op1')
        self.assertEqual(result, ['op3', 'op2', 'op1'])

    def test_find_complete_construct_circular(self):
        """测试查找完整构造链(循环引用)"""
        construct_info = {
            'op1': 'op2',
            'op2': 'op1'
        }
        result = DataNode.find_complete_construct(construct_info, 'op1')
        self.assertEqual(result, ['op1', 'op2', 'op1'])

    def test_find_stack_valid(self):
        """测试查找堆栈信息(有效情况)"""
        data_node = DataNode('test_op', 0, self.op_data)
        stack_info = {
            'key1': ['op1_test_op_op3', {'file': 'test.py', 'line': 10}],
            'key2': ['other_op', {'file': 'other.py', 'line': 20}]
        }
        result = data_node.find_stack(stack_info)
        self.assertEqual(result, {'file': 'test.py', 'line': 10})

    def test_find_stack_not_found(self):
        """测试查找堆栈信息(未找到)"""
        data_node = DataNode('test_op', 0, self.op_data)
        stack_info = {
            'key1': ['other_op', {'file': 'test.py', 'line': 10}]
        }
        result = data_node.find_stack(stack_info)
        self.assertEqual(result, {})

    def test_find_stack_invalid_value_type(self):
        """测试查找堆栈信息(值类型无效)"""
        data_node = DataNode('test_op', 0, self.op_data)
        stack_info = {
            'key1': 'invalid_string'
        }
        with self.assertRaises(Exception) as context:
            data_node.find_stack(stack_info)
        self.assertIn("value's type in stack.json should be a list", str(context.exception))

    @patch('msprobe.overflow_check.graph.check_item_anomaly')
    def test_is_anomaly_true(self, mock_check):
        """测试异常检测(返回True)"""
        # 模拟输入正常，输出异常
        mock_check.side_effect = [False, False, False, True]

        data_node = DataNode('test_op', 0, self.op_data)
        result = data_node.is_anomaly()
        self.assertTrue(result)

    @patch('msprobe.overflow_check.graph.check_item_anomaly')
    def test_is_anomaly_false_due_to_input(self, mock_check):
        """测试异常检测(因输入异常返回False)"""
        # 模拟输入异常
        mock_check.side_effect = [True, False, False, False]

        data_node = DataNode('test_op', 0, self.op_data)
        result = data_node.is_anomaly()
        self.assertFalse(result)

    @patch('msprobe.overflow_check.graph.check_item_anomaly')
    def test_is_anomaly_false_due_to_output(self, mock_check):
        """测试异常检测(因输出正常返回False)"""
        # 模拟输入正常，输出正常
        mock_check.side_effect = [False, False, False, False]

        data_node = DataNode('test_op', 0, self.op_data)
        result = data_node.is_anomaly()
        self.assertFalse(result)

    @patch('msprobe.overflow_check.graph.is_ignore_op')
    def test_is_anomaly_ignore_op(self, mock_ignore):
        """测试异常检测(忽略的操作)"""
        mock_ignore.return_value = True

        data_node = DataNode('test_op', 0, self.op_data)
        result = data_node.is_anomaly()
        self.assertFalse(result)

    def test_gen_node_info_forward(self):
        """测试生成节点信息(前向传播)"""
        data_node = DataNode('forward_test', 0, self.op_data)

        with patch('msprobe.overflow_check.graph.FileCache') as MockCache:
            mock_cache = Mock()
            mock_cache.load_json.side_effect = [
                {'op1': 'op2', 'op2': None},  # construct
                {'key': ['forward_test_extra', {'stack': 'info'}]}  # stack
            ]
            MockCache.return_value = mock_cache

            mock_path = Mock()
            mock_path.construct_path = '/path/to/construct'
            mock_path.stack_path = '/path/to/stack'

            result = data_node.gen_node_info(mock_path)

            expected_data_info = {
                'input_args': [{'type': 'float', 'value': 0.5}],
                'input_kwargs': {'lr': {'type': 'float', 'value': 0.01}},
                'output': {'result': {'type': 'tensor', 'value': 2.0}}
            }

            self.assertEqual(result['op_name'], 'forward_test')
            self.assertEqual(result['data_info'], expected_data_info)
            self.assertEqual(result['construct_info'], ['forward_test'])
            self.assertEqual(result['stack_info'], {'stack': 'info'})


class TestCommunicationNode(unittest.TestCase):
    """CommunicationNode 单元测试类"""

    def setUp(self):
        """测试前准备"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {},
            'output': {}
        }
        self.data_node = DataNode('Distributed.send.1.2', 0, op_data)

    def test_init_valid_op_name(self):
        """测试CommunicationNode初始化(有效操作名)"""
        node = CommunicationNode('node1', 0, self.data_node)

        self.assertEqual(node.node_id, 'node1')
        self.assertEqual(node.rank, 0)
        self.assertEqual(node.data, self.data_node)
        self.assertEqual(node.layer, 0)
        self.assertEqual(node.api, 'send')
        self.assertEqual(node.call_cnt, '1')
        self.assertEqual(node.pre_node, None)
        self.assertEqual(node.link_nodes, {})
        self.assertEqual(node.dst_nodes, {})
        self.assertEqual(node.src_nodes, {})
        self.assertEqual(node.next_nodes, {})
        self.assertEqual(node.compute_ops, [])
        self.assertEqual(node.connected, False)

    def test_init_invalid_op_name(self):
        """测试CommunicationNode初始化(无效操作名)"""
        invalid_op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {},
            'output': {}
        }
        invalid_data_node = DataNode('InvalidName', 0, invalid_op_data)

        with self.assertRaises(RuntimeError) as context:
            CommunicationNode('node1', 0, invalid_data_node)
        self.assertIn('invalid op_name', str(context.exception))

    def test_init_with_custom_layer(self):
        """测试CommunicationNode初始化(自定义layer)"""
        node = CommunicationNode('node1', 0, self.data_node, layer=3)
        self.assertEqual(node.layer, 3)
        self.assertEqual(node.data.layer, 0)

    def test_init_with_kwargs(self):
        """测试CommunicationNode初始化(带kwargs参数)"""
        mock_node = Mock(spec=CommunicationNode)
        mock_node.node_id = 'node2'

        node = CommunicationNode(
            'node1', 0, self.data_node,
            pre_node=mock_node,
            link_nodes={'node3': mock_node},
            dst_nodes={'node4': mock_node},
            src_nodes={'node5': mock_node},
            next_nodes={'node6': mock_node},
            compute_ops=['op1', 'op2']
        )

        self.assertEqual(node.pre_node, mock_node)
        self.assertEqual(len(node.link_nodes), 1)
        self.assertEqual(len(node.dst_nodes), 1)
        self.assertEqual(len(node.src_nodes), 1)
        self.assertEqual(len(node.next_nodes), 1)
        self.assertEqual(node.compute_ops, ['op1', 'op2'])

    def test_add_next(self):
        """测试添加下一个节点"""
        node1 = CommunicationNode('node1', 0, self.data_node, layer=1)
        node2_data = DataNode('Distributed.recv.2.3', 0, {})
        node2 = CommunicationNode('node2', 0, node2_data)

        node1.add_next(node2)

        self.assertIn('node2', node1.next_nodes)
        self.assertEqual(node1.next_nodes['node2'], node2)
        self.assertEqual(node2.pre_node, node1)
        self.assertEqual(node2.layer, 2)
        self.assertEqual(node2.data.layer, 2)

    def test_add_link(self):
        """测试添加链接节点"""
        node1 = CommunicationNode('node1', 0, self.data_node, layer=1)
        node2_data = DataNode('Distributed.allreduce.2.3', 0, {})
        node2 = CommunicationNode('node2', 1, node2_data)

        node1.add_link(node2)

        self.assertIn('node2', node1.link_nodes)
        self.assertIn('node1', node2.link_nodes)
        self.assertEqual(node1.link_nodes['node2'], node2)
        self.assertEqual(node2.link_nodes['node1'], node1)
        self.assertEqual(node2.layer, 1)
        self.assertEqual(node2.data.layer, 1)
        self.assertTrue(node1.connected)
        self.assertTrue(node2.connected)

    def test_add_dst(self):
        """测试添加目标节点"""
        node1 = CommunicationNode('node1', 0, self.data_node, layer=1)
        node2_data = DataNode('Distributed.recv.2.3', 0, {})
        node2 = CommunicationNode('node2', 1, node2_data)

        node1.add_dst(node2)

        self.assertIn('node2', node1.dst_nodes)
        self.assertIn('node1', node2.src_nodes)
        self.assertEqual(node1.dst_nodes['node2'], node2)
        self.assertEqual(node2.src_nodes['node1'], node1)
        self.assertEqual(node2.layer, 1)
        self.assertEqual(node2.data.layer, 1)
        self.assertTrue(node1.connected)
        self.assertTrue(node2.connected)

    @patch('msprobe.overflow_check.graph.check_item_anomaly')
    def test_has_nan_inf_true(self, mock_check):
        """测试NaN/INF检测(有异常)"""
        # 模拟输入有异常
        mock_check.side_effect = [True, True]

        node = CommunicationNode('node1', 0, self.data_node)
        result = node.has_nan_inf()
        self.assertTrue(result)

    def test_input_has_nan_inf_true(self):
        """测试输入NaN/INF检测(有异常)"""
        with patch('msprobe.overflow_check.graph.check_item_anomaly') as mock_check:
            mock_check.return_value = True

            node = CommunicationNode('node1', 0, self.data_node)
            result = node.input_has_nan_inf()
            self.assertTrue(result)

    def test_input_has_nan_inf_false(self):
        """测试输入NaN/INF检测(无异常)"""
        with patch('msprobe.overflow_check.graph.check_item_anomaly') as mock_check:
            mock_check.return_value = False

            node = CommunicationNode('node1', 0, self.data_node)
            result = node.input_has_nan_inf()
            self.assertFalse(result)

    def test_resolve_type_src(self):
        """测试解析节点类型(SRC)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {'src': {'value': 0}},
            'output': {}
        }
        data_node = DataNode('Distributed.recv.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        self.assertEqual(node.type, 'src')

    def test_resolve_type_dst(self):
        """测试解析节点类型(DST)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {'src': {'value': 1}},
            'output': {}
        }
        data_node = DataNode('Distributed.send.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        self.assertEqual(node.type, 'dst')

    def test_resolve_type_link(self):
        """测试解析节点类型(LINK)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {},
            'output': {}
        }
        data_node = DataNode('Distributed.allreduce.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        self.assertEqual(node.type, 'link')

    def test_find_connected_nodes_with_dst(self):
        """测试查找连接节点(有dst)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {'dst': {'value': 1}},
            'output': {}
        }
        data_node = DataNode('Distributed.send.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        result = node.find_connected_nodes()
        self.assertEqual(result['ranks'], {1})
        self.assertEqual(result['api'], 'Distributed.recv')

    def test_find_connected_nodes_with_src(self):
        """测试查找连接节点(有src)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {'src': {'value': 2}},
            'output': {}
        }
        data_node = DataNode('Distributed.recv.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        result = node.find_connected_nodes()
        self.assertEqual(result['ranks'], {2})
        self.assertEqual(result['api'], 'Distributed.send')
        self.assertEqual(result['type'], 'src')

    def test_find_connected_nodes_with_group(self):
        """测试查找连接节点(有group)"""
        op_data = {
            'inputs': [],
            'input_args': [],
            'input_kwargs': {'group': {'group_ranks': {0, 1, 2}}},
            'output': {}
        }
        data_node = DataNode('Distributed.allreduce.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        result = node.find_connected_nodes()
        self.assertEqual(result['ranks'], {0, 1, 2})
        self.assertEqual(result['api'], 'Distributed.allreduce')
        self.assertEqual(result['type'], 'link')

    def test_find_connected_nodes_with_args(self):
        """测试查找连接节点(有args参数)"""
        op_data = {
            'inputs': [],
            'input_args': [{'type': 'int', 'value': 3}],
            'input_kwargs': {},
            'output': {}
        }
        data_node = DataNode('Distributed.send.1.2', 0, op_data)
        node = CommunicationNode('node1', 0, data_node)

        result = node.find_connected_nodes()
        self.assertEqual(result['ranks'], {3})
        self.assertEqual(result['api'], 'Distributed.recv')

    def test_p2p_api_mapping(self):
        """测试P2P API映射"""
        with patch.dict('msprobe.overflow_check.graph.OverFlowCheckConst.P2P_API_MAPPING',
                        {'test_api': 'mapped_api'}):
            op_data = {
                'inputs': [],
                'input_args': [],
                'input_kwargs': {},
                'output': {}
            }
            data_node = DataNode('Distributed.test_api.1.2', 0, op_data)
            node = CommunicationNode('node1', 0, data_node)

            result = node.find_connected_nodes()
            self.assertEqual(result['api'], 'Distributed.mapped_api')


def run_tests():
    """运行测试"""
    # 创建测试套件
    loader = unittest.TestLoader()

    # 添加测试类
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDataNode))
    suite.addTests(loader.loadTestsFromTestCase(TestCommunicationNode))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)



if __name__ == '__main__':
    # 运行测试
    run_tests()
