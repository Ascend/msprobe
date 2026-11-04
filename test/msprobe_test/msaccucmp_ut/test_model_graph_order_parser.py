#!/usr/bin/env python
# coding=utf-8
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at:
#
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from unittest import mock

from cmp_utils.constant.compare_error import CompareError
from vector_cmp.fusion_manager.model_graph_order_parser import ModelGraphOrderParser


class TestModelGraphOrderParser(unittest.TestCase):

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_uses_op_array_positions_instead_of_ids(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [
                {
                    'op': [
                        {'name': 'relu', 'id': 7},
                        {'name': 'cast', 'id': 4},
                        {'name': 'data'},
                        {'name': 'block/op.1', 'id': 5},
                        {'name': 'output'},
                    ]
                }
            ]
        }

        result = ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(result, {'relu': 0, 'cast': 1, 'data': 2, 'block_op_1': 3, 'output': 4})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_keeps_graphs_together_with_missing_and_reversed_ids(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [
                {'op': [{'name': 'first', 'id': 10}, {'name': 'between'}]},
                {'op': [{'name': 'second', 'id': 1}, {'name': 'tail'}]},
            ]
        }

        result = ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(result, {'first': 0, 'between': 1, 'second': 2, 'tail': 3})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_uses_json_order_when_all_nodes_have_no_id(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [
                {'op': [{'name': 'first'}, {'name': 'second'}]},
                {'op': [{'name': 'third'}]},
            ]
        }

        result = ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(result, {'first': 0, 'second': 1, 'third': 2})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_does_not_use_id_metadata(self, mock_load_json_file):
        for ignored_id in (True, False, -1, 1.5, '1', None):
            with self.subTest(ignored_id=ignored_id):
                mock_load_json_file.return_value = {
                    'graph': [{'op': [{'name': 'first', 'id': ignored_id}, {'name': 'second', 'id': 0}]}]
                }
                self.assertEqual(
                    ModelGraphOrderParser('/home/demo/model.json').parse(), {'first': 0, 'second': 1}
                )

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_rejects_normalized_name_collision(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [{'op': [{'name': 'block/op', 'id': 1}, {'name': 'block.op', 'id': 2}]}]
        }

        with self.assertRaises(CompareError) as error:
            ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_preserves_traversal_order_for_equal_ids(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [
                {'op': [{'name': 'first', 'id': 1}, {'name': 'second', 'id': 1}, {'name': 'zero', 'id': 0}]},
                {'op': [{'name': 'input'}, {'name': 'third', 'id': 1}]},
            ]
        }

        result = ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(result, {'first': 0, 'second': 1, 'zero': 2, 'input': 3, 'third': 4})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_preserves_main_graph_and_subgraph_array_order(self, mock_load_json_file):
        graphs = [
            {'op': [{'name': 'input', 'type': 'Data'}, {'name': 'shape', 'id': 1}, {'name': 'output', 'id': 31}]},
            {'op': [{'name': 'batch_1', 'id': 20}, {'name': 'batch_1_output', 'id': 29}]},
            {'op': [{'name': 'batch_0', 'id': 7}, {'name': 'batch_0_output', 'id': 18}]},
        ]
        mock_load_json_file.return_value = {'graph': graphs}
        result = ModelGraphOrderParser('/home/demo/model.json').parse()

        self.assertEqual(list(result), ['input', 'shape', 'output', 'batch_1', 'batch_1_output',
                                       'batch_0', 'batch_0_output'])

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_rejects_ambiguous_names_across_graphs(self, mock_load_json_file):
        mock_load_json_file.return_value = {
            'graph': [{'op': [{'name': 'block/op'}]}, {'op': [{'name': 'block.op'}]}]
        }
        with self.assertRaises(CompareError):
            ModelGraphOrderParser('/home/demo/model.json').parse()

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_uses_standard_dump_special_fields_in_graph_order(self, mock_load_json_file):
        mock_load_json_file.return_value = {'graph': [
            {'op': [{'name': 'partition0_sgt_graph_0_block/op.1_lxslice0', 'id': 90}]},
            {'op': [{'name': 'partition1_sgt_graph_1_tail_lxslice1', 'id': 1}]},
        ]}
        self.assertEqual(ModelGraphOrderParser('/home/demo/model.json').parse(), {'block_op_1': 0, 'tail': 1})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_ffts_keeps_special_fields_and_graph_identity(self, mock_load_json_file):
        names = ['partition0_sgt_graph_0_relu_lxslice0', 'partition1_sgt_graph_1_relu_lxslice1']
        mock_load_json_file.return_value = {'graph': [{'op': [{'name': name}]} for name in names]}
        self.assertEqual(ModelGraphOrderParser('/home/demo/model.json', ffts=True).parse(),
                         {names[0]: 0, names[1]: 1})

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_rejects_special_field_collisions_within_and_across_graphs(self, mock_load_json_file):
        first = {'name': 'partition0_sgt_graph_0_relu_lxslice0'}
        second = {'name': 'partition1_sgt_graph_1_relu_lxslice1'}
        for graphs in ([{'op': [first, second]}], [{'op': [first]}, {'op': [second]}]):
            with self.subTest(graphs=graphs):
                mock_load_json_file.return_value = {'graph': graphs}
                with self.assertRaises(CompareError) as error:
                    ModelGraphOrderParser('/home/demo/model.json').parse()
                self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)
                self.assertIn('collide', error.exception.message)

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_rejects_invalid_or_empty_special_names(self, mock_load_json_file):
        for name in ('relu_lxslice', 'relu_sgt_graph', '_lxslice0'):
            with self.subTest(name=name):
                mock_load_json_file.return_value = {'graph': [{'op': [{'name': name}]}]}
                with self.assertRaises(CompareError) as error:
                    ModelGraphOrderParser('/home/demo/model.json').parse()
                self.assertEqual(error.exception.code, CompareError.MSACCUCMP_INVALID_JSON_FILE_ERROR)

    @mock.patch('vector_cmp.fusion_manager.model_graph_order_parser.FileUtils.load_json_file')
    def test_parse_rejects_invalid_structure_and_names(self, mock_load_json_file):
        for invalid_graph in (
            [], {}, {'graph': {}}, {'graph': []}, {'graph': [None]}, {'graph': [{'op': {}}]},
            {'graph': [{'op': [None]}]}, {'graph': [{'op': [{'name': ''}]}]},
            {'graph': [{'op': [{'name': 1}]}]},
        ):
            with self.subTest(invalid_graph=invalid_graph):
                mock_load_json_file.return_value = invalid_graph
                with self.assertRaises(CompareError):
                    ModelGraphOrderParser('/home/demo/model.json').parse()


if __name__ == '__main__':
    unittest.main()
