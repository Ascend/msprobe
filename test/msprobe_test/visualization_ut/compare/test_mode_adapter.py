import json
import unittest
from msprobe.visualization.compare.mode_adapter import ModeAdapter
from msprobe.visualization.graph.base_node import BaseNode
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import GraphConst, ToolTip
from msprobe.core.common.const import CompareConst


class TestModeAdapter(unittest.TestCase):

    def setUp(self):
        self.node_op = NodeOp.module
        self.node_id = "node_1"
        self.node = BaseNode(self.node_op, self.node_id)
        self.compare_mode = GraphConst.REAL_DATA_COMPARE
        self.adapter = ModeAdapter(self.compare_mode)
        self.compare_data_dict = [{}, {}]

    def test_match_data(self):
        compare_data = ['Module.module.Float16Module.forward.0.input.0',
                        'Module.module.Float16Module.forward.0.input.0', 'torch.int64', 'torch.int64', [4, 4096],
                        [4, 4096], 0.0, 0.0, 0.0, 0.0, '0.0%', '0.0%', '0.0%', '0.0%', 30119.0, 1.0, 8466.25,
                        1786889.625, 30119.0, 1.0, 8466.25, 1786889.625, '', '']
        data_dict = {'type': 'torch.Tensor', 'dtype': 'torch.int64', 'shape': [4, 4096], 'Max': 30119.0, 'Min': 1.0,
                     'Mean': 8466.25, 'Norm': 1786889.625, 'requires_grad': False,
                     'full_op_name': 'Module.module.Float16Module.forward.0.input.0', 'data_name': '-1',
                     'md5': '00000000'}
        id_list = [6, 7, 8, 9, 10, 11, 12, 13]
        id_list1 = [6, 7, 8, 9, 10, 11, 12, 13, 14]
        key_list = ['Max diff', 'Min diff', 'Mean diff', 'L2norm diff', 'MaxRelativeErr', 'MinRelativeErr',
                    'MeanRelativeErr', 'NormRelativeErr']
        ModeAdapter._match_data(data_dict, compare_data, key_list, id_list1)
        self.assertNotIn('Max diff', data_dict)
        ModeAdapter._match_data(data_dict, compare_data, key_list, id_list)
        self.assertIn('Max diff', data_dict)

    def test_check_list_len(self):
        data_list = [1, 2]
        with self.assertRaises(ValueError):
            ModeAdapter._check_list_len(data_list, 3)

    def test_parse_result(self):
        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        precision_index = self.adapter.parse_result(self.node, self.compare_data_dict, CompareConst.PASS)
        self.assertEqual(precision_index, 0)

        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        precision_index = self.adapter.parse_result(self.node, self.compare_data_dict, CompareConst.WARNING)
        self.assertEqual(precision_index, 0.5)

        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        precision_index = self.adapter.parse_result(self.node, self.compare_data_dict, CompareConst.ERROR)
        self.assertEqual(precision_index, 1)

    def test_prepare_real_data(self):
        result = self.adapter.prepare_real_data(self.node)
        self.assertTrue(result)

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        result = self.adapter.prepare_real_data(self.node)
        self.assertFalse(result)

    def test_add_csv_data(self):
        compare_result_list = ['result1', 'result2']
        self.adapter.add_csv_data(compare_result_list)
        self.assertEqual(self.adapter.csv_data, compare_result_list)

    def test_get_tool_tip(self):
        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({'md5': ToolTip.MD5}))

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({
            CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
            CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
            CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
            CompareConst.NORM_DIFF: ToolTip.NORM_DIFF}))

        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        tips = self.adapter.get_tool_tip()
        algorithm_columns = CompareConst.ALL_COMPARE_INDEX
        expect_tips = {col: "" for col in algorithm_columns}
        self.assertEqual(tips, json.dumps(expect_tips))
