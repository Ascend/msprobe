#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import math
import os
import csv
import tempfile
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import (
    TensorDiff,
    ModuleDiff,
    CompareResult,
    _GradSlot,
    _to_f32_cpu,
    _build_grad_slots,
    _iter_tensor_slots,
    _materialize_grad_slots,
    _cmp_tensor,
    _cmp_list,
    _iter_tensor_pairs,
    _normalize_name,
    _is_orig_mod_node,
    _TensorStore,
    _CastWrapper,
    PrecisionChecker,
    _WRAP_ATTR,
    _IGNORE_ATTR,
)


class TestTensorDiff(unittest.TestCase):
    def test_str_ok(self):
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2, 3))
        s = str(td)
        self.assertIn('OK', s)
        self.assertIn('shape=(2, 3)', s)

    def test_str_fail(self):
        td = TensorDiff(max_abs=1.0, mean_abs=0.5, max_rel=0.8, allclose=False, shape=(4,))
        s = str(td)
        self.assertIn('!!', s)
        self.assertIn('max_abs=1.000e+00', s)


class TestCompareResult(unittest.TestCase):
    def test_loss_diff_normal(self):
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.5)
        self.assertAlmostEqual(cr.loss_diff, 0.5)

    def test_loss_diff_nan_eager(self):
        cr = CompareResult(loss_eager=float('nan'), loss_compiled=1.5)
        self.assertTrue(math.isnan(cr.loss_diff))

    def test_all_pass_true(self):
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='m1', fwd_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        self.assertTrue(cr.all_pass)

    def test_all_pass_false(self):
        td = TensorDiff(max_abs=1.0, mean_abs=0.5, max_rel=0.8, allclose=False, shape=(2,))
        md = ModuleDiff(name='m1', fwd_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        self.assertFalse(cr.all_pass)

    def test_all_pass_skip_note(self):
        md = ModuleDiff(name='m1', note='SKIP_inside_compiled')
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        self.assertTrue(cr.all_pass)

    def test_all_pass_ignored_note(self):
        md = ModuleDiff(name='m1', note='IGNORED')
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        self.assertTrue(cr.all_pass)

    def test_all_pass_none_diffs(self):
        md = ModuleDiff(name='m1', fwd_input=None, fwd_output=None, grad_input=None, grad_output=None)
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        self.assertTrue(cr.all_pass)

    def test_all_pass_mixed(self):
        td_ok = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        td_fail = TensorDiff(max_abs=1.0, mean_abs=0.5, max_rel=0.8, allclose=False, shape=(2,))
        md1 = ModuleDiff(name='m1', fwd_output=[td_ok])
        md2 = ModuleDiff(name='m2', fwd_output=[td_fail])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md1, md2])
        self.assertFalse(cr.all_pass)


class TestGradSlot(unittest.TestCase):
    def test_default_value(self):
        slot = _GradSlot()
        self.assertIsNone(slot.value)

    def test_set_value(self):
        slot = _GradSlot()
        t = torch.tensor([1.0, 2.0])
        slot.value = t
        self.assertTrue(torch.equal(slot.value, t))


class TestToF32Cpu(unittest.TestCase):
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        result = _to_f32_cpu(t)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device('cpu'))

    def test_int_tensor(self):
        t = torch.tensor([1, 2], dtype=torch.int64)
        result = _to_f32_cpu(t)
        self.assertEqual(result.dtype, torch.float32)

    def test_tuple(self):
        t = torch.tensor([1.0], dtype=torch.float16)
        result = _to_f32_cpu((t, t))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].dtype, torch.float32)

    def test_list(self):
        t = torch.tensor([1.0], dtype=torch.float16)
        result = _to_f32_cpu([t, t])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_non_tensor(self):
        self.assertEqual(_to_f32_cpu(42), 42)
        self.assertEqual(_to_f32_cpu("hello"), "hello")

    def test_nested(self):
        t = torch.tensor([1.0], dtype=torch.float16)
        result = _to_f32_cpu((t, [t, t]))
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[1], list)


class TestBuildGradSlots(unittest.TestCase):
    def test_tensor(self):
        t = torch.tensor([1.0])
        result = _build_grad_slots(t)
        self.assertIsInstance(result, _GradSlot)

    def test_tuple(self):
        t = torch.tensor([1.0])
        result = _build_grad_slots((t, t))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], _GradSlot)

    def test_list(self):
        t = torch.tensor([1.0])
        result = _build_grad_slots([t, t])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_non_tensor(self):
        result = _build_grad_slots(42)
        self.assertEqual(result, 42)


class TestIterTensorSlots(unittest.TestCase):
    def test_single_tensor(self):
        t = torch.tensor([1.0])
        slot = _GradSlot()
        result = list(_iter_tensor_slots(t, slot))
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0][0], t))
        self.assertIs(result[0][1], slot)

    def test_tuple(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        slots = (_GradSlot(), _GradSlot())
        result = list(_iter_tensor_slots((t1, t2), slots))
        self.assertEqual(len(result), 2)

    def test_list(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        slots = [_GradSlot(), _GradSlot()]
        result = list(_iter_tensor_slots([t1, t2], slots))
        self.assertEqual(len(result), 2)

    def test_nested(self):
        t = torch.tensor([1.0])
        slot = _GradSlot()
        result = list(_iter_tensor_slots((t,), (slot,)))
        self.assertEqual(len(result), 1)


class TestMaterializeGradSlots(unittest.TestCase):
    def test_slot_with_value(self):
        slot = _GradSlot()
        slot.value = torch.tensor([1.0])
        result = _materialize_grad_slots(slot)
        self.assertTrue(torch.equal(result, torch.tensor([1.0])))

    def test_slot_none(self):
        slot = _GradSlot()
        result = _materialize_grad_slots(slot)
        self.assertIsNone(result)

    def test_tuple(self):
        s1 = _GradSlot()
        s1.value = torch.tensor([1.0])
        s2 = _GradSlot()
        result = _materialize_grad_slots((s1, s2))
        self.assertIsInstance(result, tuple)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0])))
        self.assertIsNone(result[1])

    def test_list(self):
        s1 = _GradSlot()
        s1.value = torch.tensor([1.0])
        result = _materialize_grad_slots([s1])
        self.assertIsInstance(result, list)

    def test_non_slot(self):
        self.assertEqual(_materialize_grad_slots(42), 42)


class TestCmpTensor(unittest.TestCase):
    def test_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        diff = _cmp_tensor(a, b)
        self.assertTrue(diff.allclose)
        self.assertAlmostEqual(diff.max_abs, 0.0)
        self.assertEqual(diff.shape, (3,))

    def test_different(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 4.0])
        diff = _cmp_tensor(a, b)
        self.assertFalse(diff.allclose)
        self.assertAlmostEqual(diff.max_abs, 1.0)

    def test_shape_mismatch(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0])
        diff = _cmp_tensor(a, b)
        self.assertFalse(diff.allclose)
        self.assertEqual(diff.max_abs, float('inf'))

    def test_small_diff_pass(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0 + 1e-5, 2.0 + 1e-5])
        diff = _cmp_tensor(a, b)
        self.assertTrue(diff.allclose)


class TestCmpList(unittest.TestCase):
    def test_both_none(self):
        result = _cmp_list(None, None)
        self.assertIsNone(result)

    def test_one_none(self):
        t = torch.tensor([1.0])
        result = _cmp_list(t, None)
        self.assertIsNone(result)

    def test_single_tensors(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0])
        result = _cmp_list(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].allclose)

    def test_list_of_tensors(self):
        a = [torch.tensor([1.0]), torch.tensor([2.0])]
        b = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = _cmp_list(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)


class TestIterTensorPairs(unittest.TestCase):
    def test_single_pair(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        pairs = list(_iter_tensor_pairs(a, b))
        self.assertEqual(len(pairs), 1)

    def test_tuple_pairs(self):
        a = (torch.tensor([1.0]), torch.tensor([2.0]))
        b = (torch.tensor([3.0]), torch.tensor([4.0]))
        pairs = list(_iter_tensor_pairs(a, b))
        self.assertEqual(len(pairs), 2)

    def test_list_pairs(self):
        a = [torch.tensor([1.0])]
        b = [torch.tensor([2.0])]
        pairs = list(_iter_tensor_pairs(a, b))
        self.assertEqual(len(pairs), 1)

    def test_nested_pairs(self):
        a = (torch.tensor([1.0]), (torch.tensor([2.0]),))
        b = (torch.tensor([3.0]), (torch.tensor([4.0]),))
        pairs = list(_iter_tensor_pairs(a, b))
        self.assertEqual(len(pairs), 2)


class TestNormalizeName(unittest.TestCase):
    def test_orig_mod_prefix(self):
        self.assertEqual(_normalize_name('_orig_mod.linear'), 'linear')

    def test_orig_mod_middle(self):
        self.assertEqual(_normalize_name('layer._orig_mod.linear'), 'layer.linear')

    def test_module_dot(self):
        self.assertEqual(_normalize_name('layer.module.linear'), 'layer.linear')

    def test_module_suffix(self):
        self.assertEqual(_normalize_name('layer.module'), 'layer')

    def test_clean_name(self):
        self.assertEqual(_normalize_name('linear'), 'linear')

    def test_combined(self):
        self.assertEqual(_normalize_name('_orig_mod.layer.module.linear'), 'layer.linear')


class TestIsOrigModNode(unittest.TestCase):
    def test_orig_mod(self):
        self.assertTrue(_is_orig_mod_node('_orig_mod'))

    def test_orig_mod_suffix(self):
        self.assertTrue(_is_orig_mod_node('layer._orig_mod'))

    def test_module(self):
        self.assertTrue(_is_orig_mod_node('module'))

    def test_module_suffix(self):
        self.assertTrue(_is_orig_mod_node('layer.module'))

    def test_normal(self):
        self.assertFalse(_is_orig_mod_node('linear'))

    def test_contains_but_not_suffix(self):
        self.assertFalse(_is_orig_mod_node('module_linear'))


class TestTensorStore(unittest.TestCase):
    def test_init(self):
        store = _TensorStore()
        self.assertEqual(store.fwd_in, {})
        self.assertEqual(store.fwd_out, {})
        self.assertEqual(store.bwd, {})

    def test_clear(self):
        store = _TensorStore()
        store.fwd_in['a'] = [1]
        store.fwd_out['b'] = [2]
        store.bwd['c'] = {'grad_input': None}
        store.clear()
        self.assertEqual(store.fwd_in, {})
        self.assertEqual(store.fwd_out, {})
        self.assertEqual(store.bwd, {})


class TestCastWrapper(unittest.TestCase):
    def test_init(self):
        linear = nn.Linear(4, 4)
        wrapper = _CastWrapper(linear, torch.float16)
        self.assertIs(wrapper.module, linear)
        self.assertEqual(wrapper.cast_dtype, torch.float16)

    def test_forward_no_autocast(self):
        linear = nn.Linear(4, 4)
        wrapper = _CastWrapper(linear, None)
        x = torch.randn(2, 4)
        with patch.object(torch, 'autocast') as mock_autocast:
            mock_autocast.return_value.__enter__ = MagicMock(return_value=None)
            mock_autocast.return_value.__exit__ = MagicMock(return_value=None)
            result = wrapper(x)
            self.assertEqual(result.shape, (2, 4))


class TestPrecisionCheckerInit(unittest.TestCase):
    def test_defaults(self):
        pc = PrecisionChecker()
        self.assertEqual(pc.backend, 'aot_eager')
        self.assertAlmostEqual(pc.threshold, 1e-4)
        self.assertFalse(pc.dump_graphs)
        self.assertEqual(pc.graph_dir, './graph_dump')
        self.assertIsNone(pc.cast_dtype)
        self.assertTrue(pc.capture_input)
        self.assertTrue(pc.single_pass)

    def test_custom(self):
        pc = PrecisionChecker(
            backend='inductor',
            threshold=1e-3,
            dump_graphs=True,
            graph_dir='/tmp/graphs',
            cast_dtype=torch.float16,
            capture_input=False,
            single_pass=False,
        )
        self.assertEqual(pc.backend, 'inductor')
        self.assertAlmostEqual(pc.threshold, 1e-3)
        self.assertTrue(pc.dump_graphs)
        self.assertEqual(pc.graph_dir, '/tmp/graphs')
        self.assertEqual(pc.cast_dtype, torch.float16)
        self.assertFalse(pc.capture_input)
        self.assertFalse(pc.single_pass)


class TestPrecisionCheckerWrap(unittest.TestCase):
    def test_wrap_with_name(self):
        pc = PrecisionChecker()
        linear = nn.Linear(4, 4)
        result = pc.wrap(linear, name='my_linear')
        self.assertIs(result, linear)
        self.assertTrue(getattr(linear, _WRAP_ATTR))
        self.assertEqual(pc._wrapped_ids[id(linear)], 'my_linear')

    def test_wrap_without_name(self):
        pc = PrecisionChecker()
        linear = nn.Linear(4, 4)
        result = pc.wrap(linear)
        self.assertIs(result, linear)
        self.assertEqual(pc._wrapped_ids[id(linear)], 'Linear')

    def test_wrap_by_policy(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        pc.wrap_by_policy(model, (nn.ReLU,))
        relu = model[1]
        self.assertTrue(getattr(relu, _WRAP_ATTR))
        self.assertIn(id(relu), pc._wrapped_ids)

    def test_wrap_all_children(self):
        pc = PrecisionChecker()

        class ParentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.child1 = nn.Linear(2, 2)
                self.child2 = nn.Linear(2, 2)

        model = ParentModel()
        pc.wrap_all_children(model, depth=1)
        self.assertTrue(getattr(model.child1, _WRAP_ATTR))
        self.assertTrue(getattr(model.child2, _WRAP_ATTR))


class TestPrecisionCheckerIgnore(unittest.TestCase):
    def test_ignore(self):
        pc = PrecisionChecker()
        linear = nn.Linear(4, 4)
        result = pc.ignore(linear)
        self.assertIs(result, linear)
        self.assertTrue(getattr(linear, _IGNORE_ATTR))
        self.assertIn(id(linear), pc._ignored_ids)

    def test_ignore_by_policy(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 4))
        pc.ignore_by_policy(model, (nn.Dropout,))
        dropout = model[1]
        self.assertTrue(getattr(dropout, _IGNORE_ATTR))
        self.assertIn(id(dropout), pc._ignored_ids)


class TestPrecisionCheckerCollectPrefixes(unittest.TestCase):
    def test_collect_wrap_prefixes(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        pc.wrap(model[1], name='1')
        prefixes = pc._collect_prefixes(model, _WRAP_ATTR)
        self.assertEqual(prefixes, {'1'})

    def test_collect_ignore_prefixes(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        pc.ignore(model[1])
        prefixes = pc._collect_prefixes(model, _IGNORE_ATTR)
        self.assertEqual(prefixes, {'1'})

    def test_collect_no_wrap_returns_none(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4))
        prefixes = pc._collect_prefixes(model, _WRAP_ATTR)
        self.assertIsNone(prefixes)

    def test_collect_no_ignore_returns_empty(self):
        pc = PrecisionChecker()
        model = nn.Sequential(nn.Linear(4, 4))
        prefixes = pc._collect_prefixes(model, _IGNORE_ATTR)
        self.assertEqual(prefixes, set())


class TestPrecisionCheckerRngState(unittest.TestCase):
    def test_save_and_restore(self):
        pc = PrecisionChecker()
        torch.manual_seed(42)
        state = pc._save_rng_state()
        _ = torch.randn(10)
        pc._restore_rng_state(*state)
        a = torch.randn(10)
        torch.manual_seed(42)
        b = torch.randn(10)
        self.assertTrue(torch.allclose(a, b))


class TestPrecisionCheckerBuildEagerCast(unittest.TestCase):
    def test_no_cast_dtype(self):
        pc = PrecisionChecker()
        model = nn.Linear(4, 4)
        result = pc._build_eager_cast(model)
        self.assertIs(result, model)

    def test_with_cast_dtype(self):
        pc = PrecisionChecker(cast_dtype=torch.float16)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        pc.wrap(model[0], name='0')
        result = pc._build_eager_cast(model)
        self.assertIsInstance(result, nn.Sequential)
        self.assertIsInstance(result[0], _CastWrapper)


class TestPrecisionCheckerBuildDiffs(unittest.TestCase):
    def test_ignored_module(self):
        pc = PrecisionChecker()
        e_store = _TensorStore()
        c_store = _TensorStore()
        e_store.fwd_out['ignored_mod'] = [torch.tensor([1.0])]
        c_store.fwd_out['ignored_mod'] = [torch.tensor([1.0])]
        diffs = pc._build_diffs(e_store, c_store, set(), {'ignored_mod'})
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].note, 'IGNORED')

    def test_matching_fwd_output(self):
        pc = PrecisionChecker()
        e_store = _TensorStore()
        c_store = _TensorStore()
        e_store.fwd_out['linear'] = [torch.tensor([1.0, 2.0])]
        c_store.fwd_out['linear'] = [torch.tensor([1.0, 2.0])]
        diffs = pc._build_diffs(e_store, c_store, set(), set())
        self.assertEqual(len(diffs), 1)
        self.assertIsNotNone(diffs[0].fwd_output)
        self.assertTrue(diffs[0].fwd_output[0].allclose)

    def test_missing_fwd_in_compiled(self):
        pc = PrecisionChecker()
        e_store = _TensorStore()
        c_store = _TensorStore()
        e_store.fwd_out['linear'] = [torch.tensor([1.0])]
        diffs = pc._build_diffs(e_store, c_store, set(), set())
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].note, 'MISSING_fwd_in_compiled')

    def test_missing_fwd_in_eager(self):
        pc = PrecisionChecker()
        e_store = _TensorStore()
        c_store = _TensorStore()
        c_store.fwd_out['linear'] = [torch.tensor([1.0])]
        diffs = pc._build_diffs(e_store, c_store, set(), set())
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].note, 'MISSING_fwd_in_eager')

    def test_inside_compiled_skip(self):
        pc = PrecisionChecker()
        e_store = _TensorStore()
        c_store = _TensorStore()
        e_store.fwd_out['outer.inner'] = [torch.tensor([1.0])]
        wrapper_names = {'outer'}
        diffs = pc._build_diffs(e_store, c_store, wrapper_names, set())
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].note, 'SKIP_inside_compiled')


class TestPrecisionCheckerBuildDiffsSinglePass(unittest.TestCase):
    def test_ignored(self):
        pc = PrecisionChecker()
        store = _TensorStore()
        store.fwd_out['mod'] = [TensorDiff(0.0, 0.0, 0.0, True, (2,))]
        diffs = pc._build_diffs_single_pass(store, {'mod'})
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].note, 'IGNORED')

    def test_fwd_output(self):
        pc = PrecisionChecker()
        store = _TensorStore()
        td = TensorDiff(0.0, 0.0, 0.0, True, (2,))
        store.fwd_out['linear'] = [td]
        diffs = pc._build_diffs_single_pass(store, set())
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].fwd_output, [td])

    def test_fwd_input_tensor(self):
        pc = PrecisionChecker(capture_input=True)
        store = _TensorStore()
        store.fwd_out['linear'] = [TensorDiff(0.0, 0.0, 0.0, True, (2,))]
        store.fwd_in['linear'] = torch.tensor([1.0, 2.0])
        diffs = pc._build_diffs_single_pass(store, set())
        self.assertEqual(len(diffs), 1)
        self.assertIsNotNone(diffs[0].fwd_input)
        self.assertTrue(diffs[0].fwd_input[0].allclose)

    def test_fwd_input_list(self):
        pc = PrecisionChecker(capture_input=True)
        store = _TensorStore()
        store.fwd_out['linear'] = [TensorDiff(0.0, 0.0, 0.0, True, (2,))]
        store.fwd_in['linear'] = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        diffs = pc._build_diffs_single_pass(store, set())
        self.assertEqual(len(diffs), 1)
        self.assertIsNotNone(diffs[0].fwd_input)
        self.assertEqual(len(diffs[0].fwd_input), 2)

    def test_no_capture_input(self):
        pc = PrecisionChecker(capture_input=False)
        store = _TensorStore()
        store.fwd_out['linear'] = [TensorDiff(0.0, 0.0, 0.0, True, (2,))]
        store.fwd_in['linear'] = torch.tensor([1.0, 2.0])
        diffs = pc._build_diffs_single_pass(store, set())
        self.assertIsNone(diffs[0].fwd_input)

    def test_bwd_data(self):
        pc = PrecisionChecker()
        store = _TensorStore()
        store.fwd_out['linear'] = [TensorDiff(0.0, 0.0, 0.0, True, (2,))]
        td = TensorDiff(0.0, 0.0, 0.0, True, (2,))
        store.bwd['linear'] = {'grad_input': [td], 'grad_output': None}
        diffs = pc._build_diffs_single_pass(store, set())
        self.assertEqual(diffs[0].grad_input, [td])
        self.assertIsNone(diffs[0].grad_output)


class TestPrecisionCheckerRecordLoss(unittest.TestCase):
    def test_record_tensor_loss(self):
        pc = PrecisionChecker()
        pc._install_loss_c = None
        loss = torch.tensor(0.5)
        pc.record_loss(loss)
        self.assertAlmostEqual(pc._install_loss_c, 0.5)

    def test_record_float_loss(self):
        pc = PrecisionChecker()
        pc._install_loss_c = None
        pc.record_loss(1.5)
        self.assertAlmostEqual(pc._install_loss_c, 1.5)


class TestPrecisionCheckerInstall(unittest.TestCase):
    def test_install_requires_single_pass(self):
        pc = PrecisionChecker(single_pass=False)
        model = nn.Linear(4, 4)
        with self.assertRaises(RuntimeError) as ctx:
            pc.install(model)
        self.assertIn('single_pass', str(ctx.exception))


class TestPrecisionCheckerReport(unittest.TestCase):
    def test_report_two_pass(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertIn('ALL PASS', printed)

    def test_report_single_pass(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td])
        cr = CompareResult(loss_eager=float('nan'), loss_compiled=1.0, diffs=[md])
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertIn('single_pass', printed)

    def test_report_failed(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=1.0, mean_abs=0.5, max_rel=0.8, allclose=False, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=2.0, diffs=[md])
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertIn('FAILED', printed)

    def test_report_with_skip_note(self):
        pc = PrecisionChecker()
        md = ModuleDiff(name='inner', note='SKIP_inside_compiled')
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertIn('skip', printed)

    def test_report_with_cast_dtype(self):
        pc = PrecisionChecker(cast_dtype=torch.float16)
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, cast_dtype=torch.float16)
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertIn('cast_dtype', printed)

    def test_report_no_capture_input(self):
        pc = PrecisionChecker(capture_input=False)
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td], fwd_input=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with patch('builtins.print') as mock_print:
            pc.report(cr)
            printed = ''.join(str(c) for c in mock_print.call_args_list)
            self.assertNotIn('FORWARD INPUT', printed)


class TestPrecisionCheckerCsvReport(unittest.TestCase):
    def test_write_csv_two_pass(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=0.001, mean_abs=0.0005, max_rel=0.002, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        try:
            with patch('builtins.print'):
                pc.report(cr, csv_path=csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            self.assertGreater(len(rows), 1)
            self.assertEqual(rows[0][0], 'module_name')
            found_linear = any('linear' in row[0] for row in rows)
            self.assertTrue(found_linear)
        finally:
            os.unlink(csv_path)

    def test_write_csv_single_pass(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', fwd_output=[td])
        cr = CompareResult(loss_eager=float('nan'), loss_compiled=1.0, diffs=[md])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        try:
            with patch('builtins.print'):
                pc.report(cr, csv_path=csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            loss_row = [r for r in rows if r[0] == 'LOSS']
            self.assertEqual(len(loss_row), 1)
            self.assertIn('single_pass', loss_row[0][-1])
        finally:
            os.unlink(csv_path)

    def test_write_csv_with_ignored(self):
        pc = PrecisionChecker()
        md = ModuleDiff(name='drop', note='IGNORED')
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        try:
            with patch('builtins.print'):
                pc.report(cr, csv_path=csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            drop_rows = [r for r in rows if r[0] == 'drop']
            self.assertEqual(len(drop_rows), 1)
            self.assertEqual(drop_rows[0][3], 'SKIP')
        finally:
            os.unlink(csv_path)

    def test_write_csv_with_grad(self):
        pc = PrecisionChecker()
        td = TensorDiff(max_abs=0.0, mean_abs=0.0, max_rel=0.0, allclose=True, shape=(2,))
        md = ModuleDiff(name='linear', grad_input=[td], grad_output=[td])
        cr = CompareResult(loss_eager=1.0, loss_compiled=1.0, diffs=[md])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        try:
            with patch('builtins.print'):
                pc.report(cr, csv_path=csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            grad_rows = [r for r in rows if r[1] in ('grad_input', 'grad_output')]
            self.assertEqual(len(grad_rows), 2)
        finally:
            os.unlink(csv_path)


class TestPrecisionCheckerCompare(unittest.TestCase):
    @patch('msprobe.pytorch.compile_accuracy_checker.precision_checker.PrecisionChecker._compare_single_pass')
    @patch('msprobe.pytorch.compile_accuracy_checker.precision_checker.PrecisionChecker._compare_two_pass')
    def test_compare_single_pass(self, mock_two_pass, mock_single_pass):
        mock_result = CompareResult(loss_eager=1.0, loss_compiled=1.0)
        mock_single_pass.return_value = mock_result
        pc = PrecisionChecker(single_pass=True)
        model = nn.Linear(4, 4)
        fn = lambda m: m(torch.randn(2, 4)).sum()
        result = pc.compare(fn, model)
        mock_single_pass.assert_called_once()
        mock_two_pass.assert_not_called()

    @patch('msprobe.pytorch.compile_accuracy_checker.precision_checker.PrecisionChecker._compare_single_pass')
    @patch('msprobe.pytorch.compile_accuracy_checker.precision_checker.PrecisionChecker._compare_two_pass')
    def test_compare_two_pass(self, mock_two_pass, mock_single_pass):
        mock_result = CompareResult(loss_eager=1.0, loss_compiled=1.0)
        mock_two_pass.return_value = mock_result
        pc = PrecisionChecker(single_pass=False)
        model = nn.Linear(4, 4)
        fn = lambda m: m(torch.randn(2, 4)).sum()
        result = pc.compare(fn, model)
        mock_two_pass.assert_called_once()
        mock_single_pass.assert_not_called()


if __name__ == '__main__':
    unittest.main()
