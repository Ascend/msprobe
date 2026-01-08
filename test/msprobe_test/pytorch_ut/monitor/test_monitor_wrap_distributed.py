import unittest
from collections import defaultdict
from unittest.mock import patch, MagicMock
import torch
from torch import distributed as dist
from msprobe.pytorch.monitor.module_hook import CommunicationContext

from msprobe.pytorch.monitor.distributed.wrap_distributed import get_distributed_ops, DistributedOPTemplate, \
    ApiRegistry, \
    get_process_group, stack_filter, get_callstack, op_aggregate, update_data, is_target_line, \
    create_async_callback_func, \
    ORIGIN_WAIT, PENDING_ASYNC_CC_BY_HANDLE, catch_data, create_hooks

WrapDistributedOps = ['all_reduce', 'broadcast', 'all_gather']


class TestGetDistributedOps(unittest.TestCase):

    def test_get_distributed_ops(self):
        expected = {'all_reduce', 'broadcast', 'all_gather'}
        with patch('msprobe.pytorch.monitor.distributed.wrap_distributed.WrapDistributedOps', \
                   new=['all_reduce', 'broadcast', 'all_gather']):
            result = get_distributed_ops()

        self.assertEqual(result, expected)

    def test_get_distributed_ops_with_non_existent_op(self):
        expected = {'all_reduce', 'broadcast'}
        with patch('msprobe.pytorch.monitor.distributed.wrap_distributed.WrapDistributedOps', \
                   new=['all_reduce', 'broadcast', 'non_existent_op']):
            result = get_distributed_ops()

        self.assertEqual(result, expected)

    def test_get_distributed_ops_only_exclusions(self):
        expected = set()
        with patch('msprobe.pytorch.monitor.distributed.wrap_distributed.WrapDistributedOps', new=['exclusion']):
            result = get_distributed_ops()

        self.assertEqual(result, expected)


class TestDistributedOPTemplate(unittest.TestCase):

    def hook(name):
        def forward_pre_hook(nope, input, kwargs):
            pass

        def forward_hook(nope, input, kwargs, result):
            pass

        return forward_pre_hook, forward_hook

    def test_distributed_op(self):
        self.setUp()
        op_name = 'all_reduce'
        if op_name in get_distributed_ops():
            op = DistributedOPTemplate(op_name, [self.hook()[0]], [self.hook()[1]])
            self.assertEqual(op.op_name_, op_name)


class TestApiRegistry(unittest.TestCase):

    def setUp(self) -> None:
        self.ApiRegistry = ApiRegistry()
        global ORIGIN_WAIT
        global PENDING_ASYNC_CC_BY_HANDLE
        self.attr_dict = {"b2": 2, "b3": 3}

    def hook(name):
        def forward_pre_hook(nope, input, kwargs):
            pass

        def forward_hook(nope, input, kwargs, result):
            pass

        return forward_pre_hook, forward_hook

    def tearDown(self) -> None:
        # 清空 PENDING_ASYNC_CC_BY_HANDLE
        PENDING_ASYNC_CC_BY_HANDLE.clear()

    def test_store_ori_attr(self):
        class A():
            a1 = 1

        class B():
            a = A()
            b1 = 1
            b2 = 2

        api_list = ["a.a1", "b1", "b2"]
        expect_output = {"a.a1": 1, "b1": 1, "b2": 2}
        actual_output = dict()
        ApiRegistry.store_ori_attr(B, api_list, actual_output)
        self.assertEqual(actual_output, expect_output)

    def test_set_api_attr(self):
        class A():
            a1 = 1

        class B():
            a = A().__class__
            b1 = 1

        attr_dict = {"a.a2": 2, "b2": 2, "b3": 3}
        ApiRegistry.set_api_attr(B, attr_dict)

        for k, v in attr_dict.items():
            if '.' in k:
                sub_module_name, sub_op = k.rsplit('.', 1)
                sub_module = getattr(B, sub_module_name, None)

                self.assertEqual(getattr(sub_module, sub_op), v)
            else:
                self.assertEqual(getattr(B, k), v)

    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.torch.distributed.Work')
    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.ORIGIN_WAIT')
    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.PENDING_ASYNC_CC_BY_HANDLE')
    def test_redirect_wait_with_pending(self, mock_handle, mock_wait, mock_work):
        # 注册一个待处理的函数
        mock_wait.return_value = MagicMock()
        mock_handle["handle"] = MagicMock()

        # 执行 redirect_wait
        ApiRegistry.redirect_wait()

        # 调用 wrapped_wait
        wrapped_wait = dist.Work.wait
        wrapped_wait("handle")

        # 验证 ORIGIN_WAIT 被调用
        mock_wait.assert_called_once_with("handle")

    def test_redirect_api(self):
        self.ApiRegistry.distributed_attr_hooked = self.attr_dict
        self.ApiRegistry.redirect_api()

        self.assertEqual(dist.b2, 2)
        self.assertEqual(dist.distributed_c10d.b2, 2)

    def test_restore_api(self):
        self.ApiRegistry.distributed_attr_origin = self.attr_dict
        self.ApiRegistry.restore_api()

        self.assertEqual(dist.b2, 2)
        self.assertEqual(dist.distributed_c10d.b2, 2)

    def test_initialize_hook(self):
        self.ApiRegistry.initialize_hook([self.hook()[0]], [self.hook()[1]])

        self.assertEqual(len(get_distributed_ops()), len(self.ApiRegistry.distributed_attr_origin))
        self.assertEqual(len(get_distributed_ops()), len(self.ApiRegistry.distributed_attr_hooked))


class TestFunctions(unittest.TestCase):

    def test_get_process_group(self):
        process_group_element = dist.GroupMember.WORLD
        result = get_process_group(process_group_element)

        self.assertEqual(result, process_group_element)

    def test_get_process_group_with_none(self):
        result = get_process_group(None)

        self.assertEqual(result, dist.GroupMember.WORLD)

    def test_stack_filter_false(self):
        stack = 'msprobe/pytorch/monitor/distributed'
        result = stack_filter(stack)

        self.assertFalse(result)

    def test_stack_filter_true(self):
        stack = 'wrong/stack'
        result = stack_filter(stack)

        self.assertTrue(result)

    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.inspect')
    def test_get_callstack(self, mock_inspect):
        mock_inspect.stack.return_value = [(None, 'wrong/stack', 1, 'function', None, None)]
        expected = ['wrong/stack[1]   function']
        result = get_callstack()

        self.assertEqual(result, expected)

    def test_op_aggregate_with_tensor(self):
        tensor = torch.tensor([1, 2, 3])

        self.assertTrue(torch.equal(op_aggregate('', tensor), tensor))

    def test_op_aggregate_with_non_tensor(self):
        self.assertTrue(torch.isnan(op_aggregate('', None)))

    def test_op_aggregate_with_op_min(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('min', tensorlist), 1)

    def test_op_aggregate_with_op_max(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('max', tensorlist), 3)

    def test_op_aggregate_with_op_norm(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('norm', tensorlist), 6)

    def test_op_aggregate_with_op_zeros(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('zeros', tensorlist), 2)

    def test_op_aggregate_with_op_nans(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('nans', tensorlist), 6)

    def test_op_aggregate_with_op_mean(self):
        tensorlist = [1, 2, 3]

        self.assertEqual(op_aggregate('mean', tensorlist), 2)

    def test_op_aggregate_with_default_op(self):
        tensorlist = [1, 2, 3]
        res = op_aggregate('test_op', tensorlist)
        self.assertTrue(res.isnan().item())

    def test_op_aggregate_other(self):
        self.assertTrue(torch.isnan(op_aggregate('', None)))

    def test_update_data_new(self):
        old = {}
        new = {'tag1': {'op1': torch.tensor([1, 2, 3])}}
        expected = torch.tensor([1, 2, 3])
        old = update_data(old, new)

        self.assertIsInstance(old['tag1']['op1'], list)
        self.assertTrue(torch.equal(old['tag1']['op1'][0], expected))

    def test_update_data_append(self):
        old = {'tag1': {'op1': [torch.tensor([1, 2, 3])]}}
        new = {'tag1': {'op1': torch.tensor([2, 3, 4]), 'op2': torch.tensor([3, 4, 5])}}
        old = update_data(old, new)

        self.assertIsInstance(old['tag1']['op1'], list)
        self.assertEqual(len(old['tag1']['op1']), 2)
        self.assertTrue(torch.equal(old['tag1']['op1'][1], torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(old['tag1']['op2'][0], torch.tensor([3, 4, 5])))

    def test_is_target_line_with_empty(self):
        self.assertTrue(is_target_line([]))

    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.get_callstack')
    def test_is_target_line_with_pattern_found(self, mock_stack):
        mock_stack.return_value = ['stack1', 'stack2']

        self.assertTrue(is_target_line(['stack1']))

    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.get_callstack')
    def test_is_target_line_other(self, mock_stack):
        mock_stack.return_value = ['stack1', 'stack2']

        self.assertFalse(is_target_line(['stack3']))

    @patch('msprobe.pytorch.monitor.distributed.wrap_distributed.catch_data')
    def test_create_async_callback_func(self, mock_catch_data):
        context = 'test_context'
        cc_name = 'test_cc_name'
        ops = 'test_ops'
        args = 'test_args'
        prefix = 'test_prefix'

        # 创建回调函数
        callback_func = create_async_callback_func(context, cc_name, ops, args, prefix)

        # 调用回调函数
        callback_func()

        # 验证 catch_data 是否被调用，并且传递的参数正确
        mock_catch_data.assert_called_once_with(context, cc_name, ops, args, prefix)


class TestCatchData(unittest.TestCase):

    def setUp(self) -> None:
        self.cc_context = CommunicationContext()
        self.cc_name = 'cc_name'
        self.ops = ["min", "max"]
        self.prefix = 'prefix'
        self.target_key = "cc_name/prefix_0"

    def test_catch_data_with_tensor(self):
        args = [torch.tensor([1, 2, 3])]
        catch_data(self.cc_context, self.cc_name, self.ops, args, self.prefix)
        self.assertEqual(len(self.cc_context.data), 1)

    def test_catch_data_with_list_of_tensors(self):
        args = [[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]]
        catch_data(self.cc_context, self.cc_name, self.ops, args, self.prefix)
        self.assertEqual(len(self.cc_context.data), 1)


class TestCreateHooks(unittest.TestCase):

    def setUp(self) -> None:
        class MockMonitor:
            cc_logged_stack = defaultdict(set)
            cc_codeline = []
            ops = ["min", "max"]
            module_rank_list = []
            cc_log_only = False
            cc_pre_hook = False
            cc_context = defaultdict(CommunicationContext)

        self.monitor = MockMonitor()
        self.context = self.monitor.cc_context
        self.ori_dist_is_initialized = dist.is_initialized
        self.ori_dist_get_rank = dist.get_rank
        self.dist_mock = MagicMock()
        self.dist_mock.get_rank.return_value = 0
        self.dist_mock.is_initialized.return_value = True
        dist.is_initialized = self.dist_mock.is_initialized
        dist.get_rank = self.dist_mock.get_rank

    def tearDown(self):
        dist.is_initialized = self.ori_dist_is_initialized
        dist.get_rank = self.ori_dist_get_rank

    def test_create_hooks_without_hook(self):
        self.monitor.module_rank_list = [0]
        pre_hooks, hooks = create_hooks(self.context, self.monitor)
        self.assertEqual(len(pre_hooks), 0)
        self.assertEqual(len(hooks), 1)

    def test_create_hooks_with_cc_log_only(self):
        self.monitor.cc_log_only = True
        pre_hooks, hooks = create_hooks(self.context, self.monitor)
        self.assertEqual(hooks, [])
        cc_log_hook = pre_hooks[0]

        mock_get_callstack = MagicMock()
        mock_get_callstack.return_value = "test string"
        mock_module = MagicMock()
        mock_module.op_name_ = "op"
        cc_log_hook(mock_module, None, None)
        self.assertIn("op", self.monitor.cc_logged_stack)
        self.assertEqual(1, len(self.monitor.cc_logged_stack["op"]))

    def test_create_hooks_with_cc_pre_hook_and_cc_hook(self):
        self.monitor.cc_pre_hook = True
        pre_hooks, hooks = create_hooks(self.context, self.monitor)
        self.assertEqual(1, len(pre_hooks))
        self.assertEqual(1, len(hooks))

        cc_pre_hook = pre_hooks[0]
        mock_module = MagicMock()
        mock_module.op_name_ = "test_module"
        args = tuple([torch.tensor([1, 2, 3])])
        kwargs = {}
        cc_pre_hook(mock_module, args, kwargs)
        self.assertIn("test_module", self.context)
        self.assertIsInstance(self.context["test_module"], CommunicationContext)

        cc_hook = hooks[0]
        cc_hook(mock_module, args, kwargs)

        res = cc_hook(mock_module, args, kwargs, [])
        self.assertEqual(res, [])


if __name__ == '__main__':
    unittest.main()
