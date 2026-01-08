# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

import unittest
from unittest.mock import MagicMock, patch

import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.runtime import Runtime
from msprobe.core.dump.data_dump.scope import ModuleRangeScope
from msprobe.core.dump.hook_manager import HookSet
from msprobe.mindspore.common import utils
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import has_kwargs_in_forward_hook, is_backward_hook_output_a_view
from msprobe.mindspore.dump.cell_processor import CellProcessor, get_cell_construct


class TestCellProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CellProcessor.reset_cell_stats()
        cls.scope = MagicMock(spec=ModuleRangeScope)
        cls.processor = CellProcessor(cls.scope)

    @classmethod
    def tearDownClass(cls):
        CellProcessor.reset_cell_stats()

    def test_class_attribute(self):
        self.assertTrue(hasattr(CellProcessor, 'cell_count'))
        self.assertTrue(hasattr(CellProcessor, 'cell_stack'))
        self.assertTrue(hasattr(CellProcessor, 'api_parent_node'))
        self.assertTrue(hasattr(CellProcessor, 'module_node'))
        self.assertTrue(hasattr(CellProcessor, 'cell_bw_hook_kernels'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_pre_hook'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_hook'))

    def test__init(self):
        self.assertIsInstance(self.processor.scope, ModuleRangeScope)
        processor = CellProcessor(None)
        self.assertIsNone(processor.scope)

    def test_get_cell_construct(self):
        def construct(self, *args, **kwargs):
            return len(args)

        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertFalse(hasattr(self, 'msprobe_input_kwargs'))
        self.assertEqual(ret, 1)

        setattr(self, 'msprobe_hook', True)
        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertEqual(self.msprobe_input_kwargs, {})
        self.assertEqual(ret, 1)

        del self.msprobe_hook
        del self.msprobe_input_kwargs

    def test_set_and_get_calls_number(self):
        CellProcessor.cell_count = {}
        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 0)
        self.assertEqual(CellProcessor.cell_count["cell"], 0)

        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 1)
        self.assertEqual(CellProcessor.cell_count["cell"], 1)

        CellProcessor.cell_count = {}

    def test_reset_cell_stats(self):
        CellProcessor.cell_count['cell'] = 0
        CellProcessor.cell_stack['tid'] = 'cell'
        CellProcessor.api_parent_node['tid'] = 'cell'
        CellProcessor.module_node['cell'] = 'null'
        CellProcessor.cell_bw_hook_kernels['cell'] = 'bw'
        CellProcessor.cell_backward_pre_hook.append('backward_pre_hook')
        CellProcessor.cell_backward_hook.append('backward_hook')

        CellProcessor.reset_cell_stats()
        self.assertEqual(CellProcessor.cell_count, {})
        self.assertEqual(CellProcessor.cell_stack, {})
        self.assertEqual(CellProcessor.api_parent_node, {})
        self.assertEqual(CellProcessor.module_node, {})
        self.assertEqual(CellProcessor.cell_bw_hook_kernels, {})
        self.assertEqual(CellProcessor.cell_backward_pre_hook, [])
        self.assertEqual(CellProcessor.cell_backward_hook, [])

    def test_register_cell_hook(self):
        with self.assertRaises(MsprobeException) as context:
            self.processor.register_cell_hook([], None, 'config')
        self.assertEqual(str(context.exception), '[msprobe] 无效参数：The model cannot be None, when level is "L0" or "mix"')

        with patch('msprobe.mindspore.dump.cell_processor.is_mindtorch') as mock_is_mindtorch, \
             patch(
                 'msprobe.mindspore.dump.cell_processor.get_cells_and_names_with_index') as mock_get_cells_and_names, \
             patch('msprobe.mindspore.dump.cell_processor.CellProcessor.build_cell_hook') as mock_build_cell_hook, \
             patch('msprobe.mindspore.dump.cell_processor.get_cell_construct') as mock_get_cell_construct, \
             patch('msprobe.mindspore.dump.cell_processor.is_graph_mode_cell_dump_allowed') \
             as mock_is_graph_mode_cell_dump_allowed, \
             patch.object(logger, 'info') as mock_logger_info:
            mock_cell = MagicMock()
            mock_sub_cell = MagicMock()
            mock_get_cells_and_names.return_value = ({'-1': [('cell', mock_cell), ('sub_cell', mock_sub_cell)]}, {})
            mock_build_cell_hook.return_value = 'forward_pre_hook'
            mock_get_cell_construct.return_value = '_construct'
            mock_is_graph_mode_cell_dump_allowed.return_value = False

            actual_kwargs_check_result = has_kwargs_in_forward_hook()

            mock_is_mindtorch.return_value = False
            utils.kwargs_exist_in_forward_hook = False
            setattr(MagicMock, '_run_construct', '_run_construct')
            self.processor.register_cell_hook(mock_cell, None, 'config')
            self.assertTrue(mock_sub_cell.__class__.msprobe_construct)
            mock_get_cell_construct.assert_called_with('_run_construct')
            self.assertEqual(mock_sub_cell.__class__._run_construct, '_construct')
            self.assertTrue(mock_sub_cell.msprobe_hook)
            mock_build_cell_hook.assert_called_with('Cell.sub_cell.MagicMock.', None)
            mock_cell.assert_not_called()
            mock_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_sub_cell.register_forward_hook.assert_not_called()
            mock_logger_info.assert_called_with('The cell hook function is successfully mounted to the model.')

            del MagicMock._run_construct
            del mock_sub_cell.__class__._run_construct
            del mock_sub_cell.__class__.msprobe_construct

            utils.kwargs_exist_in_forward_hook = True
            mock_get_cell_construct.reset_mock()
            mock_another_sub_cell = MagicMock()
            mock_get_cells_and_names.return_value = (
                {'-1': [('cell', mock_cell), ('another_sub_cell', mock_another_sub_cell)]},
                {}
            )
            self.processor.register_cell_hook(mock_cell, None, 'config')
            mock_get_cell_construct.assert_not_called()
            self.assertFalse(hasattr(mock_another_sub_cell.__class__, 'msprobe_construct'))
            mock_another_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_another_sub_cell.register_forward_hook.assert_not_called()

            utils.kwargs_exist_in_forward_hook = False
            mock_get_cell_construct.reset_mock()
            mock_another_sub_cell = MagicMock()
            setattr(mock_another_sub_cell.__class__, 'msprobe_construct', True)
            mock_get_cells_and_names.return_value = (
                {'-1': [('cell', mock_cell), ('another_sub_cell', mock_another_sub_cell)]},
                {}
            )
            self.processor.register_cell_hook(mock_cell, None, 'config')
            mock_get_cell_construct.assert_not_called()
            mock_another_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_another_sub_cell.register_forward_hook.assert_not_called()

            del mock_another_sub_cell.__class__.msprobe_construct

            mock_build_cell_hook.reset_mock()
            mock_get_cell_construct.reset_mock()
            mock_another_sub_cell.reset_mock()
            setattr(MagicMock, '_call_impl', '_call_impl')
            mock_is_mindtorch.return_value = True
            self.processor.register_cell_hook(mock_cell, None, 'config')
            self.assertTrue(mock_another_sub_cell.__class__.msprobe_construct)
            mock_get_cell_construct.assert_called_with('_call_impl')
            mock_build_cell_hook.assert_called_with('Module.another_sub_cell.MagicMock.', None)
            mock_cell.assert_not_called()
            mock_another_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_another_sub_cell.register_forward_hook.assert_not_called()

            utils.kwargs_exist_in_forward_hook = actual_kwargs_check_result
            del MagicMock._call_impl
            del mock_another_sub_cell.__class__._call_impl
            del mock_another_sub_cell.__class__.msprobe_construct

    def test_build_cell_hook(self):
        CellProcessor.reset_cell_stats()
        Runtime.is_running = True

        cell_name = 'Cell.cell.Cell.'
        mock_build_data_hook = MagicMock()
        mock_backward_data_hook = MagicMock()
        target_grad_output = (Tensor([0.5]),)
        mock_backward_data_hook.return_value = target_grad_output
        mock_hook_set = HookSet(backward_hook=mock_backward_data_hook)
        mock_build_data_hook.return_value = mock_hook_set
        mock_cell = MagicMock()

        with patch.object(_inner_ops, 'CellBackwardHook') as mock_CellBackwardHook:
            forward_pre_hook = self.processor.build_cell_hook(cell_name, mock_build_data_hook)
            forward_hook = forward_pre_hook.__closure__[1].cell_contents.__closure__[2].cell_contents

            mock_bw = mock_CellBackwardHook.return_value
            mock_bw.return_value = (Tensor([0.0]),)
            args = (Tensor([1.0]),)
            target_args = (Tensor([0.0]),)
            full_forward_name = f'{cell_name}{Const.FORWARD}.0'
            full_backward_name = f'{cell_name}{Const.BACKWARD}.0'
            # call testing function - forward_pre_hook
            ret = forward_pre_hook(mock_cell, args)
            self.assertIsNone(CellProcessor.module_node[full_forward_name])
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [full_forward_name])
            self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], full_forward_name)
            self.scope.begin_module.assert_called_with(full_forward_name)
            mock_build_data_hook.assert_called_with('Module', full_forward_name)
            self.assertEqual(len(CellProcessor.cell_backward_hook), 1)
            mock_CellBackwardHook.assert_called_with(full_backward_name, mock_cell,
                                                     CellProcessor.cell_backward_hook[-1])
            mock_bw.register_backward_hook.assert_called_once()
            if is_backward_hook_output_a_view():
                mock_bw.assert_called_with(args)
            else:
                mock_bw.assert_called_with(*args)
            self.assertTrue((ret[0] == target_args[0]).all())

            backward_hook = CellProcessor.cell_backward_hook[-1][full_backward_name]
            grad_input = (Tensor([1.0]),)
            grad_output = (Tensor([2.0]),)
            # call testing function - backward_hook
            ret = backward_hook(mock_cell, grad_input, grad_output)
            mock_backward_data_hook.assert_called_with(mock_cell, grad_input, grad_output)
            self.assertFalse(mock_cell.has_pre_hook_called)
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [])
            self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
            self.scope.end_module.assert_called_with(full_backward_name)
            self.assertTrue((ret[0] == target_grad_output[0]).all())

            mock_build_data_hook.reset_mock()
            args = (Tensor([1], dtype=ms.int32),)
            full_forward_name = f'{cell_name}{Const.FORWARD}.1'
            # call testing function - forward_pre_hook
            ret = forward_pre_hook(mock_cell, args)
            self.assertIsNone(CellProcessor.module_node[full_forward_name])
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [full_forward_name])
            self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], full_forward_name)
            self.scope.begin_module.assert_called_with(full_forward_name)
            self.assertEqual(len(CellProcessor.cell_backward_hook), 1)
            mock_build_data_hook.assert_not_called()

            full_forward_name = f'{cell_name}{Const.FORWARD}.0'
            CellProcessor.cell_count = {cell_name: 0}
            CellProcessor.cell_stack[threading.get_ident()] = [full_forward_name]
            CellProcessor.api_parent_node[threading.get_ident()] = full_forward_name
            CellProcessor.module_node = {full_forward_name: None}
            self.scope.reset_mock()
            mock_CellBackwardHook.reset_mock()
            mock_bw.reset_mock()
            target_output = Tensor([0.5])
            args = (Tensor([1.0]),)
            output = Tensor([2.0])
            mock_bw.return_value = target_output
            mock_backward_data_hook.reset_mock()
            mock_forward_data_hook = MagicMock()
            mock_forward_data_hook.return_value = output
            mock_build_data_hook.return_value = HookSet(
                forward_hook=mock_forward_data_hook, backward_hook=mock_backward_data_hook
            )
            # call testing function - forward_hook
            ret = forward_hook(mock_cell, args, output)
            self.assertEqual(CellProcessor.cell_count.get(cell_name), 0)
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [])
            self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
            self.scope.end_module.assert_called_with(full_forward_name)
            self.assertEqual(mock_bw.call_count, 2)
            self.assertEqual(mock_bw.call_args_list[0][0][0], output)
            self.assertEqual(mock_bw.call_args_list[1][0][0], target_output)
            self.assertEqual(mock_CellBackwardHook.call_count, 1)
            self.assertEqual(len(CellProcessor.cell_backward_pre_hook), 1)
            self.assertTrue((ret == target_output).all())

            backward_pre_hook = CellProcessor.cell_backward_pre_hook[-1][full_backward_name]
            mock_backward_data_hook.reset_mock()
            grad_output = (Tensor([2.0]),)
            # call testing function - backward_pre_hook
            ret = backward_pre_hook(mock_cell, grad_output)
            self.assertTrue(mock_cell.has_pre_hook_called)
            self.scope.begin_module.assert_called_with(full_backward_name)
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [full_backward_name])
            self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], full_backward_name)
            self.assertEqual(CellProcessor.module_node, {full_forward_name: None, full_backward_name: None})
            self.scope.begin_module.assert_called_with(full_backward_name)
            mock_backward_data_hook.assert_not_called()
            self.assertIsNone(ret)

            CellProcessor.cell_count = {cell_name: 0}
            CellProcessor.cell_stack[threading.get_ident()] = [full_forward_name]
            CellProcessor.api_parent_node[threading.get_ident()] = full_forward_name
            CellProcessor.module_node = {full_forward_name: None}
            mock_bw.reset_mock()
            args = (Tensor([1.0]),)
            output = (Tensor([2.0]),)
            mock_forward_data_hook.return_value = output
            target_output = (Tensor([0.5]),)
            # call testing function - forward_hook
            ret = forward_hook(mock_cell, args, output)
            self.assertEqual(mock_bw.call_count, 2)
            if is_backward_hook_output_a_view():
                self.assertEqual(mock_bw.call_args_list[0][0][0], output)
            else:
                self.assertEqual(mock_bw.call_args_list[0][0][0], *output)
            self.assertEqual(mock_bw.call_args_list[1][0][0], mock_bw.return_value)
            self.assertTrue((ret[0] == target_output[0]).all())

            CellProcessor.cell_count = {cell_name: 0}
            CellProcessor.cell_stack[threading.get_ident()] = [full_forward_name]
            CellProcessor.api_parent_node[threading.get_ident()] = full_forward_name
            CellProcessor.module_node = {full_forward_name: None}
            CellProcessor.cell_bw_hook_kernels.clear()
            CellProcessor.cell_backward_pre_hook.clear()
            mock_bw.reset_mock()
            mock_bw.return_value = (Tensor([0.5]),)
            output = (Tensor([1.0]), Tensor([2.0]))
            mock_forward_data_hook.return_value = output
            if is_backward_hook_output_a_view():
                forward_hook(mock_cell, args, output)
                mock_bw.assert_called_with(output)
            else:
                with self.assertRaises(TypeError) as context:
                    # call testing function - forward_hook
                    forward_hook(mock_cell, args, output)
                self.assertEqual(str(context.exception),
                                 'The backward pre hook return value size is 1 not equal to output size 2')
                mock_bw.assert_called_with(*output)

            self.scope.reset_mock()
            backward_pre_hook = CellProcessor.cell_backward_pre_hook[-1][full_backward_name]
            # call testing function - backward_pre_hook
            ret = backward_pre_hook(mock_cell, grad_output)
            self.assertFalse(mock_cell.has_pre_hook_called)
            self.scope.begin_module.assert_called_with(full_backward_name)
            mock_backward_data_hook.assert_called_with(mock_cell, (), grad_output)
            self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], [])
            self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
            self.assertEqual(CellProcessor.module_node, {full_forward_name: None, full_backward_name: None})
            self.scope.end_module.assert_called_with(full_backward_name)
            self.assertIsNone(ret)

            CellProcessor.reset_cell_stats()
            Runtime.is_running = False

    def test_set_construct_info_in_pre_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_pre_hook('full_name')
        self.assertEqual(CellProcessor.module_node['full_name'], None)
        self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], ['full_name'])
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'full_name')
        self.scope.begin_module.assert_called_with('full_name')

        self.scope.begin_module.reset_mock()
        self.processor.set_construct_info_in_pre_hook('sub_cell_name')
        self.assertEqual(CellProcessor.module_node, {'full_name': None, 'sub_cell_name': 'full_name'})
        self.assertEqual(CellProcessor.cell_stack[threading.get_ident()], ['full_name', 'sub_cell_name'])
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'sub_cell_name')
        self.scope.begin_module.assert_called_with('sub_cell_name')

        CellProcessor.reset_cell_stats()

    def test_set_construct_info_in_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_hook('full_name')
        self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack[threading.get_ident()] = ['full_name']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, {threading.get_ident(): []})
        self.assertIsNone(CellProcessor.api_parent_node[threading.get_ident()])
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack[threading.get_ident()] = ['Cell.0', 'Cell.1']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, {threading.get_ident(): ['Cell.0']})
        self.assertEqual(CellProcessor.api_parent_node[threading.get_ident()], 'Cell.0')
        self.scope.end_module.assert_called_with('full_name')

        CellProcessor.reset_cell_stats()
