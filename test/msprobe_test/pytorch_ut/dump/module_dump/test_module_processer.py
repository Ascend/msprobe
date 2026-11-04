# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

import sys
import threading
import unittest
from collections import OrderedDict
from io import StringIO
from unittest.mock import MagicMock, patch

import torch
from torch.utils.checkpoint import _StopRecomputationError

import msprobe.pytorch.dump.module_dump.module_processor as mp
from msprobe.core.dump.custom_tag import CustomTagManager
from msprobe.core.dump.data_dump.scope import ModuleRangeScope
from msprobe.pytorch.dump.module_dump.module_processor import (
    ModuleProcessor,
    wrap_megatron_deallocate,
    wrap_forward_with_hook_safety
)

ori_checkpoint = torch.utils.checkpoint.checkpoint


class TestModule(torch.nn.Module):
    """测试用的模块类，可控制是否抛出异常"""

    def __init__(self, raise_exception=False):
        super().__init__()
        self.raise_exception = raise_exception

    def forward(self, x, *args, **kwargs):
        if self.raise_exception:
            raise _StopRecomputationError()
        return x * 2


def ModuleProcessor_forward_hook_fn(module, args, kwargs_or_output, output_or_kwargs=None):
    print(f"The forward_hook executed normally.")


class TestWrapper(unittest.TestCase):
    def setUp(self):
        torch.utils.checkpoint.checkpoint = ori_checkpoint
        self.held_output = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.held_output

    def tearDown(self):
        """恢复标准输出"""
        sys.stdout = self.original_stdout

    def get_output(self):
        """获取捕获的输出内容"""
        return self.held_output.getvalue().strip()

    def test_wrap_megatron_deallocate(self):
        mock_func = MagicMock(return_value="output_test")
        wrapped = wrap_megatron_deallocate(mock_func)

        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor._base = True
        mock_tensor.device = "cpu"
        mock_tensor.dtype = torch.float32
        mock_tensor.clone.return_value = "cloned"

        result = wrapped(mock_tensor, deallocate_pipeline_outputs=True)
        mock_tensor.clone.assert_called_once()
        self.assertEqual(mock_tensor.data.shape, (1,))
        self.assertEqual(result, "output_test")
        mock_func.assert_called_once_with("cloned", True)

        result = wrapped("normal_input", False)
        self.assertEqual(result, "output_test")
        mock_func.assert_called_with("normal_input", False)

    def test_normal_forward_execution(self):
        """测试正常执行forward时的情况"""
        # 准备测试模块和hook
        module = TestModule(raise_exception=False)
        module.register_forward_hook(ModuleProcessor_forward_hook_fn)

        # 应用包装函数
        wrap_forward_with_hook_safety(module)

        # 执行forward
        input_tensor = torch.tensor(3.0)
        output = module(input_tensor)

        # 验证结果和hook调用
        self.assertEqual(output.item(), 6.0)
        self.assertIn("The forward_hook executed normally.", self.get_output())

    def test_stop_recomputation_exception_triggers_hook(self):
        """测试抛出_StopRecomputationError时hook被调用"""
        # 准备测试模块和hook
        module = TestModule(raise_exception=True)
        module.register_forward_hook(ModuleProcessor_forward_hook_fn)

        # 应用包装函数
        wrap_forward_with_hook_safety(module)

        # 执行forward并验证异常
        input_tensor = torch.tensor(3.0)
        with self.assertRaises(_StopRecomputationError):
            module(input_tensor)

        self.assertIn("The forward_hook executed normally.", self.get_output())


class TestModuleProcessor(unittest.TestCase):
    def setUp(self):
        ModuleProcessor.module_count = {}
        ModuleProcessor.module_stack = {}
        ModuleProcessor.module_node = {}
        ModuleProcessor.api_parent_node = {}

        self.scope = ModuleRangeScope([], [])
        self.mock_scope = MagicMock()

    @patch('msprobe.pytorch.dump.module_dump.module_processor.wrap_setup_input_output_hook')
    def test_init_with_valid_scope(self, mock_wrap):
        processor = ModuleProcessor(self.scope)
        self.assertEqual(processor.scope, self.scope)
        mock_wrap.assert_called_once()

    @patch('msprobe.pytorch.dump.module_dump.module_processor.logger.info_on_rank_0')
    def test_init_without_megatron(self, mock_log):
        ModuleProcessor(self.scope)
        mock_log.assert_called_with("No megatron find.")

    def test_set_and_get_calls_number(self):
        count = ModuleProcessor.set_and_get_calls_number("test_module")
        self.assertEqual(count, 0)

        count = ModuleProcessor.set_and_get_calls_number("test_module")
        self.assertEqual(count, 1)

    def test_has_register_backward_hook(self):
        module1 = torch.nn.Linear(10, 10)
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module1))

        module2 = MagicMock()
        module2._backward_hooks = [1, 2, 3]
        module2._is_full_backward_hook = False
        self.assertTrue(ModuleProcessor.has_register_backward_hook(module2))

        module2._is_full_backward_hook = True
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module2))

    def test_get_modules_and_names_with_model_list(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1"), ("layer2", "obj2")]
        mock_model2.named_modules.return_value = [("layer3", "obj3")]

        result = ModuleProcessor.get_modules_and_names(
            [mock_model1, mock_model2],
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1"), ("layer2", "obj2")],
            "1": [("layer3", "obj3")]
        })

    def test_get_modules_and_names_with_model_tuple(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1")]
        mock_model2.named_modules.return_value = [("layer2", "obj2")]

        result = ModuleProcessor.get_modules_and_names(
            (mock_model1, mock_model2),
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1")],
            "1": [("layer2", "obj2")]
        })

    def test_get_modules_and_names_with_single_recursive(self):
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("layer1", "obj1")]

        result = ModuleProcessor.get_modules_and_names(
            mock_model,
            recursive=True,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("layer1", "obj1")]
        })

    def test_get_modules_and_names_with_single_non_recursive(self):
        mock_model = MagicMock()
        result = ModuleProcessor.get_modules_and_names(
            mock_model,
            recursive=False,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("single_model", mock_model)]
        })

    def test_get_modules_and_names_invalid_case(self):
        result = ModuleProcessor.get_modules_and_names(
            [MagicMock(), MagicMock()],
            recursive=False,
            module_names=["only_one_name"]
        )
        self.assertEqual(result, {})

        result = ModuleProcessor.get_modules_and_names(
            MagicMock(),
            recursive=False,
            module_names=["name1", "name2"]
        )
        self.assertEqual(result, {})

    def test_reset_module_stats(self):
        ModuleProcessor.module_count = {"test": 1}
        ModuleProcessor.module_stack = ["layer1"]
        ModuleProcessor.api_parent_node = "parent"
        ModuleProcessor.module_node = {"key": "value"}
        ModuleProcessor.module_bw_hook_kernels = {"hook": "data"}
        ModuleProcessor.enable_module_dump = True

        ModuleProcessor.reset_module_stats()

        self.assertEqual(ModuleProcessor.module_count, {})
        self.assertEqual(ModuleProcessor.module_stack, {})
        self.assertEqual(ModuleProcessor.api_parent_node, {})
        self.assertEqual(ModuleProcessor.module_node, {})
        self.assertEqual(ModuleProcessor.module_bw_hook_kernels, {})
        self.assertFalse(ModuleProcessor.enable_module_dump)

    def test_set_construct_info_in_pre_hook_with_stack(self):
        processor = ModuleProcessor(self.mock_scope)
        ModuleProcessor.module_stack[threading.get_ident()] = ["parent_module"]
        processor.scope = self.mock_scope

        processor.set_construct_info_in_pre_hook("current_module")

        self.assertEqual(ModuleProcessor.module_node["current_module"], "parent_module")
        self.assertEqual(
            ModuleProcessor.module_stack[threading.get_ident()],
            ["parent_module", "current_module"]
        )
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "current_module")
        self.mock_scope.begin_module.assert_called_once_with("current_module")

    def test_set_construct_info_in_pre_hook_empty_stack(self):
        processor = ModuleProcessor(self.mock_scope)
        processor.scope = self.mock_scope
        processor.set_construct_info_in_pre_hook("root_module")

        self.assertIsNone(ModuleProcessor.module_node["root_module"])
        self.assertEqual(ModuleProcessor.module_stack[threading.get_ident()], ["root_module"])
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "root_module")

    def test_set_construct_info_in_hook_with_forward(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcessor(self.mock_scope)
        ModuleProcessor.module_stack = {threading.get_ident(): ["parent", "current"]}
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("current")

        self.assertEqual(ModuleProcessor.module_stack[threading.get_ident()], ["parent"])
        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "parent")
        self.mock_scope.end_module.assert_called_once_with("current")

    def test_set_construct_info_in_hook_with_backward(self):
        mp.torch_version_above_or_equal_2 = False
        processor = ModuleProcessor(self.mock_scope)
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("backward_module", is_forward=False)

        self.assertEqual(ModuleProcessor.api_parent_node[threading.get_ident()], "backward_module")
        self.mock_scope.begin_module.assert_called_once_with("backward_module")

    def test_set_construct_info_in_hook_empty_stack(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcessor(self.mock_scope)

        processor.set_construct_info_in_hook("module")

        self.assertEqual(ModuleProcessor.api_parent_node, {threading.get_ident(): None})


# 模拟必要的依赖
class Const:
    SEP = ':'
    FORWARD = 'forward'
    BACKWARD = 'backward'
    TYPE = 'type'


class BaseScope:
    Module_Type_Module = 'module'


class Runtime:
    is_running = True


class ThreadSafe:
    @staticmethod
    def synchronized(func):
        return func


class RemovableHandle:
    def __init__(self, hooks_dict):
        self.id = id(self)
        self.hooks_dict = hooks_dict

    def remove(self):
        self.hooks_dict.pop(self.id, None)


class BackwardHook:
    def __init__(self, module, backward_hooks, backward_pre_hooks=None):
        self.module = module
        self.backward_hooks = backward_hooks
        self.backward_pre_hooks = backward_pre_hooks or []

    def setup_input_hook(self, args):
        return args

    def setup_output_hook(self, result):
        return result


# 模拟PyTorch模块
class DummyModule:
    def __init__(self, name="dummy"):
        self.name = name
        self._modules = {}
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_hooks_with_kwargs = OrderedDict()
        self._is_full_backward_hook = False
        self.msprobe_module_dump = False

    def named_modules(self):
        yield self.name, self

    def __repr__(self):
        return f"DummyModule({self.name})"


class TestModuleProcessorTwo(unittest.TestCase):
    """ModuleProcessor类的单元测试"""

    def setUp(self):
        """测试前设置"""
        # 重置所有类变量
        ModuleProcessor.reset_module_stats()
        Runtime.is_running = True

    def tearDown(self):
        """测试后清理"""
        # 重置类变量
        ModuleProcessor.reset_module_stats()

    @patch('sys.modules')
    def test_init_with_scope(self, mock_sys_modules):
        """测试使用有效scope初始化"""
        scope = ModuleRangeScope([], [])
        processor = ModuleProcessor(scope)
        self.assertEqual(processor.scope, scope)

        # 测试无效scope
        processor = ModuleProcessor("invalid_scope")
        self.assertIsNone(processor.scope)

    @patch('sys.modules')
    def test_init_megatron_import_error(self, mock_sys_modules):
        """测试Megatron导入错误情况"""
        # 模拟import错误
        with patch.dict('sys.modules', {}):
            processor = ModuleProcessor(ModuleRangeScope([], []))
            # 应该不会抛出异常，只是记录信息

    def test_set_and_get_calls_number(self):
        """测试模块调用计数器"""
        # 初始调用
        count1 = ModuleProcessor.set_and_get_calls_number("module1")
        self.assertEqual(count1, 0)

        # 第二次调用
        count1_again = ModuleProcessor.set_and_get_calls_number("module1")
        self.assertEqual(count1_again, 1)

        # 新模块
        count2 = ModuleProcessor.set_and_get_calls_number("module2")
        self.assertEqual(count2, 0)

        # 验证内部状态
        self.assertEqual(ModuleProcessor.module_count["module1"], 1)
        self.assertEqual(ModuleProcessor.module_count["module2"], 0)

    def test_has_register_backward_hook(self):
        """测试检查backward hook注册状态"""
        # 无backward_hooks
        module1 = DummyModule()
        delattr(module1, '_backward_hooks')  # 确保没有_backward_hooks属性
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module1))

        # 空backward_hooks
        module2 = DummyModule()
        module2._backward_hooks = OrderedDict()
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module2))

        # 有backward_hooks但_is_full_backward_hook为True
        module3 = DummyModule()
        module3._backward_hooks[1] = MagicMock()
        module3._is_full_backward_hook = True
        self.assertFalse(ModuleProcessor.has_register_backward_hook(module3))

        # 有backward_hooks且_is_full_backward_hook为False
        module4 = DummyModule()
        module4._backward_hooks[1] = MagicMock()
        module4._is_full_backward_hook = False
        self.assertTrue(ModuleProcessor.has_register_backward_hook(module4))

    def test_get_modules_and_names_single_model_recursive(self):
        """测试单个模型递归获取模块和名称"""
        model = DummyModule("model")
        module1 = DummyModule("module1")
        module2 = DummyModule("module2")

        # 模拟模块层次结构
        def custom_named_modules():
            yield "", model
            yield "module1", module1
            yield "module2", module2

        model.named_modules = MagicMock(side_effect=custom_named_modules)

        result = ModuleProcessor.get_modules_and_names(model, True, ["model"])
        self.assertIn("-1", result)
        modules_list = list(result["-1"])
        self.assertEqual(len(modules_list), 3)
        self.assertEqual(modules_list[0], ("", model))
        self.assertEqual(modules_list[1], ("module1", module1))
        self.assertEqual(modules_list[2], ("module2", module2))

    def test_get_modules_and_names_single_model_non_recursive(self):
        """测试单个模型非递归获取模块和名称"""
        model = DummyModule("model")

        result = ModuleProcessor.get_modules_and_names(model, False, ["model"])
        self.assertIn("-1", result)
        modules_list = list(result["-1"])
        self.assertEqual(len(modules_list), 1)
        self.assertEqual(modules_list[0], ("model", model))

    def test_get_modules_and_names_list_models_recursive(self):
        """测试模型列表递归获取模块和名称"""
        model1 = DummyModule("model1")
        model2 = DummyModule("model2")

        module1_1 = DummyModule("module1_1")
        module1_2 = DummyModule("module1_2")
        module2_1 = DummyModule("module2_1")

        # 模拟模块层次结构
        def named_modules_model1():
            yield "", model1
            yield "module1_1", module1_1
            yield "module1_2", module1_2

        def named_modules_model2():
            yield "", model2
            yield "module2_1", module2_1

        model1.named_modules = MagicMock(side_effect=named_modules_model1)
        model2.named_modules = MagicMock(side_effect=named_modules_model2)

        models = [model1, model2]
        result = ModuleProcessor.get_modules_and_names(models, True, ["m1", "m2"])

        self.assertIn("0", result)
        self.assertIn("1", result)

        modules_list1 = list(result["0"])
        self.assertEqual(len(modules_list1), 3)
        self.assertEqual(modules_list1[0], ("", model1))
        self.assertEqual(modules_list1[1], ("module1_1", module1_1))
        self.assertEqual(modules_list1[2], ("module1_2", module1_2))

        modules_list2 = list(result["1"])
        self.assertEqual(len(modules_list2), 2)
        self.assertEqual(modules_list2[0], ("", model2))
        self.assertEqual(modules_list2[1], ("module2_1", module2_1))

    def test_get_modules_and_names_list_models_non_recursive(self):
        """测试模型列表非递归获取模块和名称"""
        model1 = DummyModule("model1")
        model2 = DummyModule("model2")

        models = [model1, model2]
        result = ModuleProcessor.get_modules_and_names(models, False, ["name1", "name2"])

        self.assertIn("0", result)
        self.assertIn("1", result)

        modules_list1 = list(result["0"])
        self.assertEqual(len(modules_list1), 1)
        self.assertEqual(modules_list1[0], ("name1", model1))

        modules_list2 = list(result["1"])
        self.assertEqual(len(modules_list2), 1)
        self.assertEqual(modules_list2[0], ("name2", model2))

    def test_get_modules_and_names_invalid_cases(self):
        """测试无效参数情况"""
        # 非递归时模型数量与名称数量不匹配
        model = DummyModule("model")
        result1 = ModuleProcessor.get_modules_and_names(model, False, ["name1", "name2"])
        self.assertEqual(result1, {})

        models = [DummyModule("m1"), DummyModule("m2")]
        result2 = ModuleProcessor.get_modules_and_names(models, False, ["name1"])
        self.assertEqual(result2, {})

    def test_reset_module_stats(self):
        """测试重置模块统计信息"""
        # 设置一些测试数据
        ModuleProcessor.module_count = {"test_module": 5}
        ModuleProcessor.module_bw_hook_kernels = {"hook1": MagicMock()}
        ModuleProcessor.module_with_backward_hook = {"module1": True}
        ModuleProcessor.enable_module_dump = True

        # 重置
        ModuleProcessor.reset_module_stats()

        # 验证重置后状态
        self.assertEqual(ModuleProcessor.module_count, {})
        self.assertEqual(ModuleProcessor.module_stack, {})
        self.assertEqual(ModuleProcessor.api_parent_node, {})
        self.assertEqual(ModuleProcessor.module_node, {})
        self.assertEqual(ModuleProcessor.module_bw_hook_kernels, {})
        self.assertEqual(ModuleProcessor.module_with_backward_hook, {'module1': True})
        self.assertFalse(ModuleProcessor.enable_module_dump)

    @patch('sys.modules')
    def test_register_module_hook_single_model_recursive(self, mock_sys_modules):
        """测试为单个模型递归注册钩子"""
        mock_sys_modules.__contains__.return_value = False  # 非verl环境

        # 创建测试模型
        model = DummyModule("model")
        module1 = DummyModule("module1")
        module2 = DummyModule("module2")

        # 模拟模块层次结构
        def custom_named_modules():
            yield "", model  # 根模块，会被跳过
            yield "module1", module1
            yield "module2", module2

        model.named_modules = MagicMock(side_effect=custom_named_modules)

        # 模拟build_hook
        mock_build_hook = MagicMock()

        # 创建处理器并注册钩子
        processor = ModuleProcessor(ModuleRangeScope([], []))
        processor.register_module_hook(model, mock_build_hook, recursive=True)

        # 验证build_hook被调用
        self.assertEqual(mock_build_hook.call_count, 0)  # module1和module2

        # 验证模块属性设置
        self.assertFalse(hasattr(module1, 'msprobe_hook'))
        self.assertFalse(hasattr(module2, 'msprobe_hook'))

        # 验证hook注册
        self.assertFalse(hasattr(module1, 'msprobe_forward_hook'))
        self.assertFalse(hasattr(module2, 'msprobe_forward_hook'))

    @patch('sys.modules')
    def test_register_module_hook_with_backward_hook(self, mock_sys_modules):
        """测试模块已注册backward hook的情况"""
        mock_sys_modules.__contains__.return_value = False  # 非verl环境

        # 创建测试模型
        model = DummyModule("model")
        module1 = DummyModule("module1")

        # 为module1设置已存在的backward hook
        module1._backward_hooks[1] = MagicMock()
        module1._is_full_backward_hook = False

        # 模拟模块层次结构
        def custom_named_modules():
            yield "", model
            yield "module1", module1

        model.named_modules = MagicMock(side_effect=custom_named_modules)

        # 模拟build_hook
        mock_build_hook = MagicMock()

        # 创建处理器并注册钩子
        processor = ModuleProcessor(ModuleRangeScope([], []))
        processor.register_module_hook(model, mock_build_hook, recursive=True)

        # 验证module_with_backward_hook被设置
        prefix_name = f'{BaseScope.Module_Type_Module}{Const.SEP}module1{Const.SEP}DummyModule{Const.SEP}'
        # 验证backward hook被跳过
        self.assertNotIn(f'{prefix_name}{Const.FORWARD}{Const.SEP}0', ModuleProcessor.module_bw_hook_kernels)

    @patch('sys.modules')
    def test_register_module_hook_verl_scenario(self, mock_sys_modules):
        """测试verl场景下的钩子注册"""
        mock_sys_modules.__contains__.return_value = True  # verl环境

        # 创建测试模型
        model = DummyModule("model")
        module1 = DummyModule("module1")
        module2 = DummyModule("module2")
        module3 = DummyModule("module3")

        # 模拟模块层次结构
        def custom_named_modules():
            yield "", model
            yield "module1", module1  # idx=0
            yield "module2", module2  # idx=1 (第一层，跳过)
            yield "module3", module3  # idx=2 (最后一层，跳过)

        model.named_modules = MagicMock(side_effect=custom_named_modules)

        # 模拟build_hook
        mock_build_hook = MagicMock()

        # 创建处理器并注册钩子
        processor = ModuleProcessor(ModuleRangeScope([], []))
        processor.register_module_hook(model, mock_build_hook, recursive=True)

        # 验证build_hook只被module1调用
        self.assertEqual(mock_build_hook.call_count, 0)
        # 验证module2和module3被跳过
        self.assertFalse(hasattr(module2, 'msprobe_hook'))
        self.assertFalse(hasattr(module3, 'msprobe_hook'))

    @patch('sys.modules')
    def test_register_module_hook_non_torch_module(self, mock_sys_modules):
        """测试非PyTorch模块的情况"""
        mock_sys_modules.__contains__.return_value = False  # 非verl环境

        # 创建非PyTorch模块
        class NonTorchModule:
            def named_modules(self):
                yield "", self

        model = NonTorchModule()

        # 模拟build_hook
        mock_build_hook = MagicMock()

        # 创建处理器并注册钩子
        processor = ModuleProcessor(ModuleRangeScope([], []))
        with patch('test_module_processor.is_torch_nn_module', return_value=False):
            processor.register_module_hook(model, mock_build_hook, recursive=True)

        # 验证build_hook未被调用
        mock_build_hook.assert_not_called()

    @patch('sys.modules')
    def test_register_module_hook_fsdp_module(self, mock_sys_modules):
        """测试FSDP模块的情况"""
        mock_sys_modules.__contains__.return_value = False  # 非verl环境

        # 创建FSDP模块
        class FullyShardedDataParallel:
            def __init__(self):
                self.name = "fsdp_module"

            def named_modules(self):
                yield "", self

        model = FullyShardedDataParallel()

        # 模拟build_hook
        mock_build_hook = MagicMock()

        # 创建处理器并注册钩子
        processor = ModuleProcessor(ModuleRangeScope([], []))
        with patch('test_module_processor.is_torch_nn_module', return_value=True):
            processor.register_module_hook(model, mock_build_hook, recursive=True)

        # 验证build_hook未被调用
        mock_build_hook.assert_not_called()

    def test_build_module_hook_forward_pre_hook(self):
        """测试构建模块钩子的前向预钩子"""
        # 创建测试处理器
        processor = ModuleProcessor(ModuleRangeScope([], []))

        # 模拟build_data_hook
        class MockHookSet:
            def __init__(self):
                self.backward_hook = MagicMock()

        def mock_build_data_hook(scope_type, full_name):
            return MockHookSet()

        # 构建钩子
        hook = processor.build_module_hook("test_module", mock_build_data_hook)

        # 创建测试模块
        module = DummyModule()
        args = (1, 2, 3)
        kwargs = {"key": "value"}

        # 测试正常情况
        result = hook(module, args, kwargs)
        self.assertEqual(result, (args, kwargs))

        # 测试非运行状态
        Runtime.is_running = False
        result2 = hook(module, args, kwargs)
        self.assertEqual(result2, (args, kwargs))
        Runtime.is_running = True  # 恢复

        # 测试模块dump禁用
        module.msprobe_module_dump = True
        ModuleProcessor.enable_module_dump = False
        result3 = hook(module, args, kwargs)
        self.assertEqual(result3, (args, kwargs))
        module.msprobe_module_dump = False
        ModuleProcessor.enable_module_dump = True

    def test_build_module_hook_forward_hook(self):
        """测试构建模块钩子的前向钩子"""
        # 创建测试处理器
        processor = ModuleProcessor(ModuleRangeScope([], []))

        # 模拟build_data_hook
        class MockHookSet:
            def __init__(self):
                self.forward_hook = MagicMock()

        def mock_build_data_hook(scope_type, full_name):
            return MockHookSet()

        # 构建钩子
        hook = processor.build_module_hook("test_module", mock_build_data_hook)

        # 创建测试模块
        module = DummyModule()
        args = (1, 2, 3)
        kwargs = {"key": "value"}
        output = {"result": "success"}

        # 先调用pre_hook以设置计数器
        hook(module, args, kwargs)

        ModuleProcessor.module_count["test_module"] = 0
        ModuleProcessor.module_bw_hook_kernels = {}

        # 模拟backward hook
        bw_hook = MagicMock()
        bw_hook.setup_output_hook.return_value = output
        ModuleProcessor.module_bw_hook_kernels[f"module:test_module:forward:0"] = bw_hook

    def test_build_module_hook_registers_custom_tag(self):
        """前向预钩子在标签作用域内登记前向名/反向名/基名"""
        processor = ModuleProcessor(ModuleRangeScope([], []))

        class MockHookSet:
            def __init__(self):
                self.backward_hook = MagicMock()
                self.module_forward_pre_hook = MagicMock()

        def mock_build_data_hook(scope_type, full_name):
            return MockHookSet()

        # module_name 带尾部分隔符，还原真实场景
        ModuleProcessor.enable_module_dump = True  # 打开模块级 dump 逻辑，否则 pre_hook 提前返回
        hook = processor.build_module_hook("Module.linear.Linear.", mock_build_data_hook)
        module = DummyModule()
        args = (1, 2, 3)
        kwargs = {"key": "value"}

        # hook 闭包在调用时从 module_processor 模块全局解析 Runtime，默认指向真实 Runtime
        # （is_running 默认 False），会导致 pre_hook 提前返回、record_tags 不被调用。
        # 此处 patch 为本文件 stub（is_running=True），保证测试不依赖进程内残留状态。
        with patch.object(mp, 'Runtime', Runtime), patch.object(CustomTagManager, 'record_tags') as mock_record:
            CustomTagManager.push("mod_tag")
            try:
                hook(module, args, kwargs)
            finally:
                CustomTagManager.pop()

            mock_record.assert_called_once()
            names, label = mock_record.call_args[0]
            self.assertEqual(label, "mod_tag")
            self.assertEqual(names, [
                "Module.linear.Linear.forward.0",
                "Module.linear.Linear.backward.0",
                "Module.linear.Linear",  # 基名（module_name 去掉尾部分隔符，供 parameters_grad 兜底）
            ])

    def test_build_module_hook_without_tag_registers_nothing(self):
        """无活跃标签时前向预钩子不登记任何条目"""
        processor = ModuleProcessor(ModuleRangeScope([], []))

        class MockHookSet:
            def __init__(self):
                self.backward_hook = MagicMock()
                self.module_forward_pre_hook = MagicMock()

        def mock_build_data_hook(scope_type, full_name):
            return MockHookSet()

        hook = processor.build_module_hook("Module.linear.Linear.", mock_build_data_hook)
        module = DummyModule()
        args = (1, 2, 3)
        kwargs = {"key": "value"}

        ModuleProcessor.enable_module_dump = True  # 打开模块级 dump 逻辑，确保真正走到登记分支
        # 同 test_build_module_hook_registers_custom_tag：patch 掉 mp.Runtime，确保 hook
        # 真正执行到登记分支（而非因 Runtime.is_running=False 提前返回导致假绿）
        with patch.object(mp, 'Runtime', Runtime), patch.object(CustomTagManager, 'record_tags') as mock_record:
            hook(module, args, kwargs)
            mock_record.assert_not_called()  # 无活跃标签时零登记

    @unittest.skipIf(not mp.torch_version_above_or_equal_21, "forward 安全包装仅在 torch>=2.1 安装")
    def test_module_depth_released_when_forward_raises(self):
        """标签块内模块 forward 抛异常被捕获后，并列模块仍按一级调用打标（深度泄漏回归）"""
        processor = ModuleProcessor(ModuleRangeScope([], []))

        class MockHookSet:
            def __init__(self):
                self.backward_hook = MagicMock()
                self.module_forward_pre_hook = MagicMock()

        def mock_build_data_hook(scope_type, full_name):
            return MockHookSet()

        class RaisingModule(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError("mock forward failure")

        class SiblingModule(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return args

        raising = RaisingModule()
        sibling = SiblingModule()

        # 按注册循环的真实顺序：先 wrap forward，再构建 pre-hook
        wrap_forward_with_hook_safety(raising)
        wrap_forward_with_hook_safety(sibling)
        pre_raising = processor.build_module_hook("Module.raising.RaisingModule.", mock_build_data_hook)
        pre_sibling = processor.build_module_hook("Module.sibling.SiblingModule.", mock_build_data_hook)

        args = (1, 2, 3)
        kwargs = {}
        with patch.object(mp, 'Runtime', Runtime), patch.object(CustomTagManager, 'record_tags') as mock_record:
            CustomTagManager.push("scope")
            try:
                ModuleProcessor.enable_module_dump = True
                # 模块 a：pre-hook 按一级调用登记标签，随后 forward 抛 RuntimeError
                pre_raising(raising, args, kwargs)
                with self.assertRaises(RuntimeError):
                    raising.forward(*args, **kwargs)
                # 修复点：异常路径已按名释放模块嵌套深度，作用域内应重新可见标签；
                # 修复前 depth 残留为 1，此处的查询返回 None
                self.assertEqual(CustomTagManager.get_current_tag(), "scope")
                # 用户在同一标签块内捕获异常后继续调用并列模块：应正常登记
                pre_sibling(sibling, args, kwargs)
                names, label = mock_record.call_args[0]
                self.assertEqual(label, "scope")
                self.assertEqual(names, [
                    "Module.sibling.SiblingModule.forward.0",
                    "Module.sibling.SiblingModule.backward.0",
                    "Module.sibling.SiblingModule",  # 基名（parameters_grad 兜底）
                ])
            finally:
                ModuleProcessor.enable_module_dump = False
                CustomTagManager.pop()
