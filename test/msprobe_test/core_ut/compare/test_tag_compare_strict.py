# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# coding=utf-8
import unittest
from unittest import mock

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.compare.acc_compare import (
    Comparator, ModeConfig, MappingConfig, ParseData, CreateTable, CalcStatsDiff,
)
from msprobe.core.compare.indicator_analysis.calculator import calculate_tag_result_df
from msprobe.core.compare.tag_compare_strict import TagCompareStrict


def t_tensor(shape, norm, req_grad=True):
    """Synthetic dump tensor dict; the norm makes each tensor unique."""
    return {
        'type': 'torch.Tensor',
        'dtype': 'torch.float32',
        'shape': shape,
        'Max': norm,
        'Min': -norm / 2.0,
        'Mean': norm / 10.0,
        'Norm': norm,
        'requires_grad': req_grad,
    }


def build_legacy_scope(tag, y_norm=None):
    """One region of a dump, tagged with a bare tag: the tagged module call
    (forward, backward, parameter gradients) carries the alias, the
    surrounding entries stay untagged. y_norm perturbs the tagged call's
    output on the value level (pairing never depends on values)."""
    x = t_tensor([2, 2], 61.0)
    y = t_tensor([2, 2], 62.0 if y_norm is None else y_norm)
    entries = {
        'Torch.sigmoid.1.forward': {'input_args': [t_tensor([2, 2], 60.0)], 'output': [x]},
        f'Module.blk_{tag}.Linear.forward.0': {
            'alias': tag, 'input_args': [x], 'output': [y],
            'parameters': {'weight': t_tensor([2, 2], 63.0), 'bias': t_tensor([2], 64.0)},
        },
        'Module.head2.Linear.forward.0': {'input_args': [y], 'output': [t_tensor([2, 2], 65.0)]},
        f'Module.blk_{tag}.Linear.backward.0': {
            'alias': tag, 'input': [t_tensor([2, 2], 66.0)], 'output': [t_tensor([2, 2], 67.0)],
        },
        'Torch.sigmoid.1.backward': {'input': [t_tensor([2, 2], 68.0)], 'output': [t_tensor([2, 2], 69.0)]},
        'Module.head2.Linear.backward.0': {'input': [t_tensor([2, 2], 72.0)], 'output': [t_tensor([2, 2], 73.0)]},
        f'Module.blk_{tag}.Linear.parameters_grad.0': {
            'alias': tag, 'weight': [t_tensor([2, 2], 70.0)], 'bias': [t_tensor([2], 71.0)]},
    }
    return entries


def build_scope_dump(tag, y_norm=None):
    """One whole dump: the tagged scope plus its untagged surroundings."""
    return {'data': build_legacy_scope(tag, y_norm=y_norm)}


def build_grad_plain_scope(tag, cls):
    """The plain-table crash scenario: the forward and backward calls carry
    the tag (the strict unit consumes both) while the parameter gradients
    stay untagged, so the plain sub-table reaching match_plain_data holds
    parameter-gradient rows only - their call_direction ('parameters_grad')
    carries no '.'."""
    x = t_tensor([2, 2], 61.0)
    y = t_tensor([2, 2], 62.0)
    entries = {
        f'Module.blk_{tag}.{cls}.forward.0': {'alias': tag, 'input_args': [x], 'output': [y]},
        f'Module.blk_{tag}.{cls}.backward.0': {
            'alias': tag, 'input': [t_tensor([2, 2], 66.0)], 'output': [t_tensor([2, 2], 67.0)]},
        f'Module.blk_{tag}.{cls}.parameters_grad.0': {
            'weight': [t_tensor([2, 2], 70.0)], 'bias': [t_tensor([2], 71.0)]},
    }
    return {'data': entries}


# full-row expectations of a bare-tag scope: every row of the single tagged
# call appears, nothing of it is compressed away
SOLO_ROWS = (
    'Module.blk_solo.Linear.forward.0.input.0',
    'Module.blk_solo.Linear.forward.0.output.0',
    'Module.blk_solo.Linear.forward.0.parameters.weight',
    'Module.blk_solo.Linear.forward.0.parameters.bias',
    'Module.blk_solo.Linear.backward.0.input.0',
    'Module.blk_solo.Linear.backward.0.output.0',
    'Module.blk_solo.Linear.parameters_grad.0.weight',
    'Module.blk_solo.Linear.parameters_grad.0.bias',
)


def parse_df(dump_json, device, dump_mode=Const.SUMMARY):
    mode_config = ModeConfig(dump_mode=dump_mode)
    parse_data = ParseData(mode_config, '', {})
    df = parse_data.gen_data_df(dump_json, None, device)
    df[[Const.DTYPE, Const.SHAPE]] = df[[Const.DTYPE, Const.SHAPE]].astype(str)
    return df


def tag_rows(df, alias):
    """Every row whose entries were tagged with this alias."""
    return df[df['Tag'] == alias]


def paired_count(sub):
    if len(sub) == 0:
        return 0
    return int(((sub['op_name_x'] != CompareConst.N_A) & (sub['op_name_y'] != CompareConst.N_A)).sum())


def paired_count_final(sub):
    """Paired count on a finalized result table (NPU Name / Bench Name columns)."""
    if len(sub) == 0:
        return 0
    return int(((sub['NPU Name'] != CompareConst.N_A) & (sub['Bench Name'] != CompareConst.N_A)).sum())


class TestTagCompareStrict(unittest.TestCase):
    """Tag units: full-row pairing of every tagged call, any call count."""

    def run_strict(self, npu_json, bench_json):
        mode_config = ModeConfig(dump_mode=Const.SUMMARY)
        npu_df = parse_df(npu_json, CompareConst.DEVICE_NPU)
        bench_df = parse_df(bench_json, CompareConst.DEVICE_BENCH)
        return TagCompareStrict(mode_config).match_tag_dfs(npu_df, bench_df)

    def run_comparator(self, npu_json, bench_json, dump_mode=Const.SUMMARY):
        mode_config = ModeConfig(dump_mode=dump_mode)
        comparator = Comparator(None, mode_config, MappingConfig())
        npu_df = parse_df(npu_json, CompareConst.DEVICE_NPU, dump_mode)
        bench_df = parse_df(bench_json, CompareConst.DEVICE_BENCH, dump_mode)
        return comparator.compare_statistics(npu_df, bench_df)

    def assert_full_rows(self, result, alias, names, msg=None):
        """Every listed row of the unit is present exactly once and paired
        with its same-name counterpart."""
        sub = tag_rows(result, alias)
        self.assertEqual(sorted(sub['op_name_x']), sorted(names), msg or alias)
        self.assertEqual(paired_count(sub), len(names), msg or alias)
        self.assertTrue((sub['op_name_x'] == sub['op_name_y']).all(), msg or alias)

    def test_self_unit_full_rows(self):
        """Every row of the single tagged call appears exactly once and pairs
        with its same-name counterpart - forward inputs/outputs, parameters,
        backward rows and parameter gradients alike."""
        result = self.run_strict(build_scope_dump('solo'), build_scope_dump('solo'))
        self.assertFalse(result.empty)
        self.assertEqual(set(result['Mode']), {CompareConst.TAG_MODE_STRICT})
        self.assert_full_rows(result, 'solo', SOLO_ROWS)
        # no row was invented outside the unit
        self.assertEqual(set(result['Tag']), {'solo'})
        # the parameter-gradient rows pair by name
        p_grad = result[result['op_name_x'] == 'Module.blk_solo.Linear.parameters_grad.0.weight']
        self.assertEqual(len(p_grad), 1)
        self.assertEqual(p_grad.iloc[0]['Extraction'], 'name')

    def test_bare_single_call_is_self_only_unit(self):
        """A tag marking one call per side is a strict unit: every row of the
        tagged call pairs with its same-name counterpart and the tag is
        folded into the name columns as a '(tag)name' prefix; untagged
        entries around it are compared normally without the prefix."""
        result = self.run_comparator(build_scope_dump('solo'), build_scope_dump('solo', y_norm=62.5))
        self.assertFalse(result.empty)
        self.assertNotIn(CompareConst.TAG, result.columns)
        # the strict rows carry the '(solo)' name prefix
        for name in SOLO_ROWS:
            rows = result[result['NPU Name'] == f'(solo){name}']
            self.assertEqual(len(rows), 1, name)
            self.assertEqual(rows.iloc[0]['Bench Name'], f'(solo){name}', name)
        # the untagged surroundings are normal comparison rows (no prefix)
        sig_rows = result[result['NPU Name'].str.contains('Torch.sigmoid.1', regex=False)]
        self.assertEqual(paired_count_final(sig_rows), 4)
        self.assertTrue((~sig_rows['NPU Name'].str.startswith('(')).all())

    def test_suffix_named_tag_is_an_ordinary_bare_tag(self):
        """A tag whose name merely ends in '_in'/'_out' is an ordinary tag:
        the former suffix routing is gone, and a single-call suffixed name
        forms a strict unit like any other name."""
        result = self.run_comparator(build_scope_dump('edge_in'), build_scope_dump('edge_in', y_norm=62.5))
        strict = result[result['NPU Name'].str.startswith('(edge_in)')]
        self.assertEqual(len(strict), len(SOLO_ROWS))
        self.assertTrue((strict['NPU Name'].str.contains('blk_edge_in', regex=False)).all())
        self.assertTrue((strict['Bench Name'] == strict['NPU Name']).all())

    def test_full_pipeline_folds_tag_into_names(self):
        """The final table keeps the native columns exactly: the tag
        metadata (Tag/Role/Extraction/Mode) stays on the intermediate frame
        only and the tag is folded into the name columns as a '(tag)name'
        prefix."""
        mode_config = ModeConfig(dump_mode=Const.SUMMARY)
        npu_df = parse_df(build_scope_dump('solo'), CompareConst.DEVICE_NPU)
        bench_df = parse_df(build_scope_dump('solo', y_norm=62.5), CompareConst.DEVICE_BENCH)
        match_result = TagCompareStrict(mode_config).match_tag_dfs(npu_df, bench_df)
        result_df, header = CreateTable(mode_config).make_result_df(match_result)
        result_df = CalcStatsDiff(mode_config).calc_accuracy(result_df, header)
        for col in (CompareConst.ROLE, CompareConst.TAG_EXTRACTION, CompareConst.TAG_MODE, CompareConst.TAG):
            self.assertNotIn(col, result_df.columns)
        self.assertNotIn(CompareConst.TAG, header)
        # strict rows carry the '(solo)' prefix on both name columns
        strict_rows = result_df.iloc[:len(match_result)]
        self.assertTrue(strict_rows['NPU Name'].str.startswith('(solo)').all())
        self.assertTrue(strict_rows['Bench Name'].str.startswith('(solo)').all())
        # the perturbed output of the tagged call shows on its output row
        y_row = result_df[result_df['NPU Name'] == '(solo)Module.blk_solo.Linear.forward.0.output.0']
        self.assertEqual(len(y_row), 1)
        self.assertNotEqual(float(y_row.iloc[0]['L2norm diff']), 0.0)
        # the tagged call's input row is identical on both sides
        x_row = result_df[result_df['NPU Name'] == '(solo)Module.blk_solo.Linear.forward.0.input.0']
        self.assertEqual(float(x_row.iloc[0]['L2norm diff']), 0.0)
        # the backward rows of the tagged call stay identical (the fixture keeps the grads equal)
        g_row = result_df[result_df['NPU Name'] == '(solo)Module.blk_solo.Linear.backward.0.output.0']
        self.assertEqual(float(g_row.iloc[0]['L2norm diff']), 0.0)

    def test_combined_result_keeps_full_picture(self):
        """A healthy tag with the comparison entered automatically: the
        tagged rows and the normal name-based comparison of every untagged
        entry land in one result table with the native columns; the rows
        follow the npu-side dump order, so the '(tag)name'-prefixed rows
        interleave with the unprefixed rows where they executed."""
        result = self.run_comparator(build_scope_dump('solo'), build_scope_dump('solo', y_norm=62.5))
        self.assertFalse(result.empty)
        self.assertNotIn(CompareConst.TAG, result.columns)
        strict_len = len(result[result['NPU Name'].str.startswith('(solo)')])
        self.assertEqual(strict_len, len(SOLO_ROWS))
        # every row of the tagged call is paired with its same-name counterpart
        for name in SOLO_ROWS:
            rows = result[result['NPU Name'] == f'(solo){name}']
            self.assertEqual(len(rows), 1, name)
            self.assertEqual(rows.iloc[0]['Bench Name'], f'(solo){name}', name)
        # the untagged entries appear as normal comparison rows without the prefix
        sig_rows = result[result['NPU Name'].str.contains('Torch.sigmoid.1', regex=False)]
        self.assertEqual(len(sig_rows), 4)  # forward/backward, input/output rows
        self.assertEqual(paired_count_final(sig_rows), 4)
        head_rows = result[result['NPU Name'].str.contains('Module.head2', regex=False)]
        self.assertEqual(len(head_rows), 4)
        self.assertTrue((~head_rows['NPU Name'].str.startswith('(')).all())

    def test_rows_follow_execution_order(self):
        """The result rows follow the npu-side dump order, which is the
        execution order of the run: a tagged row sits where its npu row
        executed, so tagged and untagged rows interleave instead of the
        tagged rows being hoisted to the top of the table as one block."""
        npu_json = build_scope_dump('solo')
        result = self.run_comparator(npu_json, build_scope_dump('solo', y_norm=62.5))
        expected = parse_df(npu_json, CompareConst.DEVICE_NPU)[CompareConst.OP_NAME].tolist()
        stripped = [name[len('(solo)'):] if name.startswith('(solo)') else name
                    for name in result['NPU Name'].tolist()]
        self.assertEqual(stripped, expected)

    def test_counter_skew_ordinal_fallback(self):
        """Global call counters skewed by earlier code must not break pairing:
        entries whose names diverge fall back to the group ordinal, and the
        parameter gradients still pair through their module base."""
        bench = build_scope_dump('solo')
        for old, new in (('Module.blk_solo.Linear.forward.0', 'Module.blk_solo.Linear.forward.2'),
                         ('Module.blk_solo.Linear.backward.0', 'Module.blk_solo.Linear.backward.2'),
                         ('Module.blk_solo.Linear.parameters_grad.0', 'Module.blk_solo.Linear.parameters_grad.2')):
            bench['data'][new] = bench['data'].pop(old)
        result = self.run_strict(build_scope_dump('solo'), bench)
        sub = tag_rows(result, 'solo')
        self.assertEqual(len(sub), len(SOLO_ROWS))
        self.assertEqual(paired_count(sub), len(SOLO_ROWS))
        # the pairs follow the group ordinal: npu .0 entries pair with bench .2 entries
        for npu_name, bench_name in (
                ('Module.blk_solo.Linear.forward.0.input.0', 'Module.blk_solo.Linear.forward.2.input.0'),
                ('Module.blk_solo.Linear.backward.0.output.0', 'Module.blk_solo.Linear.backward.2.output.0')):
            rows = sub[sub['op_name_x'] == npu_name]
            self.assertEqual(len(rows), 1, npu_name)
            self.assertEqual(rows.iloc[0]['op_name_y'], bench_name)

    def test_slot_shape_mismatch(self):
        """A slot with a different dtype/shape still pairs - the mismatch
        never breaks the pair (the native compare rules report it
        downstream), and the other slots of the same entry pair as usual."""
        bench = build_scope_dump('solo')
        bench['data']['Module.blk_solo.Linear.forward.0']['output'] = [t_tensor([2, 3], 62.0)]  # same stats, other shape
        result = self.run_strict(build_scope_dump('solo'), bench)
        sub = tag_rows(result, 'solo')
        # every tagged row pairs, including the shape-mismatching output
        self.assertEqual(paired_count(sub), len(sub))
        fwd_out = sub[sub['op_name_x'] == 'Module.blk_solo.Linear.forward.0.output.0']
        self.assertEqual(len(fwd_out), 1)
        self.assertEqual(fwd_out.iloc[0]['op_name_y'], 'Module.blk_solo.Linear.forward.0.output.0')
        # no row of the unit stays unpaired
        self.assertEqual(int((sub['Extraction'] == 'unpaired').sum()), 0)

    def test_dtype_shape_mismatch_pair_reports_native_style(self):
        """A dtype/shape mismatch inside a tag never breaks the pair: the
        pair stays and the standard indicator checkers report it with the
        native error messages, exactly like a name-matched plain pair."""
        npu = build_scope_dump('solo')
        bench = build_scope_dump('solo')
        # dtype mismatch on the forward output slot (float32 vs float16)
        out = t_tensor([2, 2], 62.0)
        out['dtype'] = 'torch.float16'
        bench['data']['Module.blk_solo.Linear.forward.0']['output'] = [out]
        # shape mismatch on the backward input slot (same dtype, other shape)
        bench['data']['Module.blk_solo.Linear.backward.0']['input'] = [t_tensor([2, 3], 66.0)]
        result = self.run_comparator(npu, bench)
        # the checker stage of the real flow (see calc_compare_result)
        calculate_tag_result_df(result, Const.SUMMARY, '')
        solo = result[result['NPU Name'].str.startswith('(solo)')]
        # every tagged row stays paired - the mismatches never break a pair
        self.assertEqual(paired_count_final(solo), len(SOLO_ROWS))
        out_row = solo[solo['NPU Name'].str.contains('forward.0.output', regex=False)]
        self.assertEqual(len(out_row), 1)
        self.assertIn('The dtype of NPU and Bench are inconsistent', str(out_row.iloc[0]['Err_message']))
        bwd_row = solo[solo['NPU Name'].str.contains('backward.0.input', regex=False)]
        self.assertEqual(len(bwd_row), 1)
        self.assertIn('The shape of NPU and Bench are inconsistent', str(bwd_row.iloc[0]['Err_message']))
        self.assertEqual(bwd_row.iloc[0]['Result'], 'error')

    def test_cross_name_pair_with_mismatch_reports_native_style(self):
        """Cross-name entries inside one tag (a fused implementation vs a
        hand-written one) pair by group ordinal, and a dtype/shape mismatch
        never breaks that pair either - the native error messages still land
        in the row, pairing and reporting stay independent."""
        npu = {'data': {
            'Fused.Attn.forward.0': {
                'alias': 'x',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        out = t_tensor([2, 2], 61.0)
        out['dtype'] = 'torch.float16'
        bench = {'data': {
            'Manual.Attn.forward.0': {
                'alias': 'x',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [out],
            },
        }}
        result = self.run_comparator(npu, bench)
        # the checker stage of the real flow (see calc_compare_result)
        calculate_tag_result_df(result, Const.SUMMARY, '')
        sub = result[result['NPU Name'].str.startswith('(x)')]
        # the differently-named entries pair by ordinal: both rows stay paired
        self.assertEqual(len(sub), 2)
        self.assertEqual(paired_count_final(sub), 2)
        self.assertTrue(sub['NPU Name'].str.contains('Fused.Attn.forward.0', regex=False).all())
        self.assertTrue(sub['Bench Name'].str.contains('Manual.Attn.forward.0', regex=False).all())
        # the dtype-mismatching output row keeps the pair and reports natively
        out_row = sub[sub['NPU Name'].str.contains('output', regex=False)]
        self.assertEqual(len(out_row), 1)
        self.assertIn('The dtype of NPU and Bench are inconsistent', str(out_row.iloc[0]['Err_message']))
        self.assertEqual(out_row.iloc[0]['Result'], 'error')

    def test_tensor_vs_scalar_slot_pairs_like_native(self):
        """A slot that is a tensor on one side and a scalar on the other
        pairs like a name-matched plain row: native never gates a name
        match on the content type - the dtype condition (a tensor dtype
        never matches a scalar type) N/As the bench columns, so the row
        reads exactly like the native display instead of two visible
        unpaired rows."""
        npu = {'data': {
            'Op.Clash.forward.0': {
                'alias': 'clash',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        bench = {'data': {
            'Op.Clash.forward.0': {
                'alias': 'clash',
                'input_args': [{'type': 'int', 'value': 7}],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        result = self.run_strict(npu, bench)
        sub = tag_rows(result, 'clash')
        # the type-clashing input slot pairs: one npu-anchored row per slot,
        # no extra bench-anchored row
        self.assertEqual(len(sub), 2)
        self.assertEqual(paired_count(sub), 2)
        clash = sub[sub['op_name_x'].str.contains('input', regex=False)]
        self.assertEqual(len(clash), 1)
        self.assertEqual(clash.iloc[0]['op_name_y'], 'Op.Clash.forward.0.input.0')
        # native display: the dtype condition N/As the bench columns of the pair
        final = self.run_comparator(npu, bench)
        calculate_tag_result_df(final, Const.SUMMARY, '')
        row = final[final['NPU Name'].str.contains('Clash.forward.0.input', regex=False)]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]['Bench Name'], CompareConst.N_A)
        self.assertEqual(row.iloc[0]['Bench Dtype'], CompareConst.N_A)
        self.assertEqual(row.iloc[0]['Result'], 'pass')
        self.assertEqual(row.iloc[0]['Err_message'], '[]')

    def test_cross_group_tensor_pair_nas_bench_columns_like_native(self):
        """A tensor-tensor pair whose dtypes cross the dtype match groups
        (float32 vs int64) keeps the pair; the native dtype condition N/As
        the bench columns of the pair - the same branch a tensor-vs-scalar
        slot takes - and no alarm is raised, so the row reads exactly like
        the native display."""
        npu = {'data': {
            'Op.DtypeGroup.forward.0': {
                'alias': 'dgroup',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        tin = t_tensor([2, 2], 60.0)
        tin['dtype'] = 'torch.int64'
        bench = {'data': {
            'Op.DtypeGroup.forward.0': {
                'alias': 'dgroup',
                'input_args': [tin],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        result = self.run_comparator(npu, bench)
        # the checker stage of the real flow (see calc_compare_result)
        calculate_tag_result_df(result, Const.SUMMARY, '')
        # the cross-group input keeps the pair with the bench columns N/A'd
        in_row = result[result['NPU Name'].str.contains('DtypeGroup.forward.0.input', regex=False)]
        self.assertEqual(len(in_row), 1)
        self.assertIn('(dgroup)', in_row.iloc[0]['NPU Name'])
        self.assertEqual(in_row.iloc[0]['Bench Name'], CompareConst.N_A)
        self.assertEqual(in_row.iloc[0]['Bench Dtype'], CompareConst.N_A)
        self.assertEqual(in_row.iloc[0]['Result'], 'pass')
        self.assertEqual(in_row.iloc[0]['Err_message'], '[]')
        # the same-dtype output slot of the same entry pairs and computes
        out_row = result[result['NPU Name'].str.contains('DtypeGroup.forward.0.output', regex=False)]
        self.assertEqual(len(out_row), 1)
        self.assertNotEqual(out_row.iloc[0]['Bench Name'], CompareConst.N_A)

    def test_missing_backward_group_dropped_like_native(self):
        """A backward direction present on the bench side only: the native
        display never grows a bench-only row, so the bench backward rows are
        dropped with a logged reason while the rest of the unit pairs as
        usual (a missing backward never refuses the unit); the npu-anchored
        rows keep the native unpaired display (bench columns N/A)."""
        npu = build_scope_dump('solo')
        del npu['data']['Module.blk_solo.Linear.backward.0']
        with mock.patch('msprobe.core.compare.tag_compare_strict.logger') as tag_logger:
            result = self.run_strict(npu, build_scope_dump('solo', y_norm=62.5))
        sub = tag_rows(result, 'solo')
        # no bench-anchored row exists: every record is npu-anchored
        self.assertTrue((sub['op_name_x'] != CompareConst.N_A).all())
        self.assertNotIn('backward', ''.join(sub['op_name_x']))
        warn_texts = [str(c) for c in tag_logger.warning.call_args_list]
        self.assertTrue(any("bench-side surplus entries are not shown" in t for t in warn_texts), warn_texts)
        # the forward and parameter-gradient rows are untouched
        self.assertEqual(paired_count(sub), 6)  # forward 4 + parameter gradients 2

    def test_bench_surplus_slot_and_entry_dropped_like_native(self):
        """Bench-side surplus rows (an extra input slot, an extra backward
        entry) never enter the table: the native display drops bench-only
        rows, so the tag path drops them too and logs the reason; the npu
        rows of the same entries keep the native unpaired display."""
        npu = {'data': {
            'Op.Clash.forward.0': {
                'alias': 'x',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [t_tensor([2, 2], 61.0)],
            },
        }}
        bench = {'data': {
            'Op.Clash.forward.0': {
                'alias': 'x',
                'input_args': [t_tensor([2, 2], 60.0), t_tensor([2, 2], 62.0)],
                'output': [t_tensor([2, 2], 61.0)],
            },
            'Op.Clash.backward.0': {
                'alias': 'x', 'input': [t_tensor([2, 2], 63.0)], 'output': [t_tensor([2, 2], 64.0)],
            },
        }}
        with mock.patch('msprobe.core.compare.tag_compare_strict.logger') as tag_logger:
            result = self.run_strict(npu, bench)
        sub = tag_rows(result, 'x')
        # no bench-anchored row exists; the npu rows pair as usual
        self.assertTrue((sub['op_name_x'] != CompareConst.N_A).all())
        self.assertEqual(paired_count(sub), 2)
        # both drop reasons are logged: the surplus slot and the surplus entry
        warn_texts = [str(c) for c in tag_logger.warning.call_args_list]
        self.assertTrue(any("slot 1 of the bench entry" in t for t in warn_texts), warn_texts)
        self.assertTrue(any("bench-side surplus entries are not shown" in t for t in warn_texts), warn_texts)

    def test_bench_surplus_slots_aggregate_into_one_warning(self):
        """Bench-only surplus slots log one aggregated warning per group
        direction (total count + first example), not one per slot: a
        tagged loop whose bench side carries a per-iteration surplus slot
        must not flood the log channel with iteration-many identical
        warnings that drown the entry-count warnings."""
        npu = {'data': {}}
        bench = {'data': {}}
        for i in range(3):
            name = f'Op.Loop.forward.{i}'
            npu['data'][name] = {
                'alias': 'loop',
                'input_args': [t_tensor([2, 2], 60.0)],
                'output': [t_tensor([2, 2], 61.0)],
            }
            bench['data'][name] = {
                'alias': 'loop',
                'input_args': [t_tensor([2, 2], 60.0), t_tensor([2, 2], 62.0)],
                'output': [t_tensor([2, 2], 61.0)],
            }
        with mock.patch('msprobe.core.compare.tag_compare_strict.logger') as tag_logger:
            result = self.run_strict(npu, bench)
        sub = tag_rows(result, 'loop')
        # three iterations x (input.0 + output.0) pair; the surplus input
        # slots never grow bench-anchored rows
        self.assertEqual(len(sub), 6)
        self.assertEqual(paired_count(sub), 6)
        self.assertTrue((sub['op_name_x'] != CompareConst.N_A).all())
        # the three surplus slots land in ONE aggregated warning
        slot_warns = [str(c) for c in tag_logger.warning.call_args_list if 'bench-side slot' in str(c)]
        self.assertEqual(len(slot_warns), 1, slot_warns)
        self.assertIn('3 bench-side slot(s)', slot_warns[0])
        self.assertIn("slot 1 of the bench entry 'Op.Loop.forward.0'", slot_warns[0])

    def test_missing_requires_grad_field_shows_native_display(self):
        """Leaves without a requires_grad field: the plain path fillna's the
        missing values to N/A before the ReqGrad consistency check, so the
        strict records do the same - the columns read N/A / True exactly
        like the native display instead of empty / False."""
        def no_grad_leaf(norm):
            return {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                    'Max': norm, 'Min': -norm / 2.0, 'Mean': norm / 10.0, 'Norm': norm}
        npu = {'data': {'Op.NoGrad.forward.0': {'alias': 'g',
                'input_args': [no_grad_leaf(60.0)], 'output': [no_grad_leaf(61.0)]}}}
        bench = {'data': {'Op.NoGrad.forward.0': {'alias': 'g',
                'input_args': [no_grad_leaf(60.0)], 'output': [no_grad_leaf(61.0)]}}}
        result = self.run_comparator(npu, bench)
        self.assertFalse(result.empty)
        self.assertTrue((result['NPU Requires_grad'] == CompareConst.N_A).all())
        self.assertTrue((result['Bench Requires_grad'] == CompareConst.N_A).all())
        self.assertTrue((result[CompareConst.REQ_GRAD_CONSIST] == True).all())  # noqa: E712

    def test_all_mode_keeps_native_na_and_fillna_conventions(self):
        """Real-data mode (Const.ALL): the strict pairs follow the native
        dtype N/A-ification and the requires_grad fillna exactly like in
        summary mode - both live in compare_tag_statistics, before the
        real-data stage runs, so the conventions must not depend on the
        dump mode."""
        def leaf(norm, dtype=None):
            row = {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                  'Max': norm, 'Min': -norm / 2.0, 'Mean': norm / 10.0, 'Norm': norm}
            if dtype is not None:
                row['dtype'] = dtype
            return row

        npu = {'data': {'Op.AllMode.forward.0': {
            'alias': 'amode', 'input_args': [leaf(60.0)], 'output': [leaf(61.0)],
        }}}
        bench = {'data': {'Op.AllMode.forward.0': {
            'alias': 'amode', 'input_args': [leaf(60.0, dtype='torch.int64')], 'output': [leaf(61.0)],
        }}}
        result = self.run_comparator(npu, bench, dump_mode=Const.ALL)
        self.assertFalse(result.empty)
        # cross-group dtype pair: the bench columns are N/A'd like the native display
        in_row = result[result['NPU Name'].str.contains('AllMode.forward.0.input', regex=False)]
        self.assertEqual(len(in_row), 1)
        self.assertEqual(in_row.iloc[0]['Bench Name'], CompareConst.N_A)
        self.assertEqual(in_row.iloc[0]['Bench Dtype'], CompareConst.N_A)
        self.assertEqual(in_row.iloc[0]['NPU Dtype'], 'torch.float32')
        # missing requires_grad fields: filled to N/A / N/A with a True
        # consistency, exactly like the native fillna convention
        self.assertTrue((result['NPU Requires_grad'] == CompareConst.N_A).all())
        self.assertTrue((result['Bench Requires_grad'] == CompareConst.N_A).all())
        self.assertTrue((result[CompareConst.REQ_GRAD_CONSIST] == True).all())  # noqa: E712

    def test_scalar_config_mismatch_visible(self):
        """A scalar argument that differs between the two sides pairs like
        any other row, but is explicitly marked as a config row (Extraction
        'config') with a warning - a differing operator configuration can
        masquerade as a precision problem."""
        npu = build_scope_dump('solo')
        bench = build_scope_dump('solo', y_norm=62.5)
        x = t_tensor([2, 2], 61.0)
        npu['data']['Module.blk_solo.Linear.forward.0']['input_args'] = [x, {'type': 'int', 'value': 1}]
        bench['data']['Module.blk_solo.Linear.forward.0']['input_args'] = [x, {'type': 'int', 'value': 2}]
        result = self.run_strict(npu, bench)
        sub = tag_rows(result, 'solo')
        # the tensor rows pair as usual and the config row carries both sides
        self.assertEqual(paired_count(sub), len(sub))
        # the disagreeing scalar joins as an explicitly marked config row
        config_rows = sub[sub['Extraction'] == 'config']
        self.assertEqual(len(config_rows), 1)
        self.assertEqual(config_rows.iloc[0]['op_name_x'], 'Module.blk_solo.Linear.forward.0.input.1')
        self.assertEqual(config_rows.iloc[0]['op_name_y'], 'Module.blk_solo.Linear.forward.0.input.1')
        self.assertEqual(config_rows.iloc[0]['summary_x'], [1, 1, 1, 1])
        self.assertEqual(config_rows.iloc[0]['summary_y'], [2, 2, 2, 2])

    def test_loop_calls_pair_iteration_by_iteration(self):
        """A tag around a loop body executed several times on both sides
        pairs iteration by iteration; global call counters skewed by earlier
        code (bench .3/.4/.5 vs npu .0/.1/.2) do not break the pairing."""
        def build_loop_dump(start):
            entries = {}
            for i in range(3):
                entries[f'Module.blk_loop.Linear.forward.{start + i}'] = {
                    'alias': 'loop',
                    'input_args': [t_tensor([2, 2], 60.0 + i)],
                    'output': [t_tensor([2, 2], 70.0 + i)],
                }
            return {'data': entries}

        result = self.run_strict(build_loop_dump(0), build_loop_dump(3))
        sub = tag_rows(result, 'loop')
        self.assertEqual(len(sub), 6)  # 3 iterations x (input + output)
        self.assertEqual(paired_count(sub), 6)
        # the i-th npu call pairs with the i-th bench call (group ordinal)
        for i in range(3):
            rows = sub[sub['op_name_x'] == f'Module.blk_loop.Linear.forward.{i}.input.0']
            self.assertEqual(len(rows), 1, i)
            self.assertEqual(rows.iloc[0]['op_name_y'], f'Module.blk_loop.Linear.forward.{i + 3}.input.0')

    def test_tag_with_mismatched_call_counts_falls_back(self):
        """A tag with differing forward call counts on the two sides (or
        present on one side only) cannot be compared through the tag: it is
        refused with a warning and its entries fall back to normal
        name-based matching instead of being compared through inferred
        boundaries."""
        npu = build_scope_dump('solo')
        bench = build_scope_dump('solo', y_norm=62.5)
        t1 = t_tensor([2, 2], 81.0)
        t2 = t_tensor([2, 2], 82.0)
        for dump in (npu, bench):
            dump['data']['Module.wide.L1.forward.0'] = {
                'alias': 'wide', 'input_args': [t_tensor([2, 2], 80.0)], 'output': [t1]}
        # npu tagged the loop twice, bench only once: 2 npu / 1 bench forward calls
        npu['data']['Module.wide.L2.forward.0'] = {'alias': 'wide', 'input_args': [t1], 'output': [t2]}
        # a tag present on the npu side only
        npu['data']['Module.only.L1.forward.0'] = {
            'alias': 'onlynpu', 'input_args': [t_tensor([2, 2], 87.0)], 'output': [t_tensor([2, 2], 88.0)]}
        result = self.run_comparator(npu, bench)
        # the refused tag fell back to normal matching: same-name entries pair,
        # the npu-surplus entry stays visible as unpaired, no '(wide)' prefix
        wide_rows = result[result['NPU Name'].str.contains('Module.wide', regex=False)]
        self.assertTrue(paired_count_final(wide_rows) > 0)
        l1_rows = wide_rows[wide_rows['NPU Name'].str.contains('L1.', regex=False)]
        self.assertEqual(paired_count_final(l1_rows), 2)
        self.assertTrue((l1_rows['Bench Name'] == l1_rows['NPU Name']).all())
        l2_rows = wide_rows[wide_rows['NPU Name'].str.contains('L2.', regex=False)]
        self.assertEqual(len(l2_rows), 2)  # input and output row, npu side only
        self.assertTrue((l2_rows['Bench Name'] == CompareConst.N_A).all())
        only_rows = result[result['NPU Name'].str.contains('Module.only', regex=False)]
        self.assertEqual(len(only_rows), 2)  # input and output row, npu side only
        self.assertTrue((only_rows['Bench Name'] == CompareConst.N_A).all())
        # the healthy single-call unit still compares through its tag
        solo_strict = result[result['NPU Name'].str.startswith('(solo)')]
        self.assertEqual(len(solo_strict), len(SOLO_ROWS))

    def test_untagged_grads_alone_still_make_report(self):
        """Regression: forward and backward both tagged, parameter gradients
        untagged. The strict unit consumes the forward/backward entries and
        the plain sub-table reaching match_plain_data holds gradient rows
        only; the comparison must still produce a report that keeps the
        gradient rows instead of dying in the backward-call renumbering
        (rsplit of the dot-less 'parameters_grad' direction)."""
        result = self.run_comparator(
            build_grad_plain_scope('grad', 'NpuLinear'),
            build_grad_plain_scope('grad', 'BenchLinear'),
        )
        self.assertFalse(result.empty)
        # the tagged forward/backward rows pair through the tag across the
        # differing implementation names
        strict = result[result['NPU Name'].str.startswith('(grad)')]
        self.assertEqual(len(strict), 4)  # forward in/out + backward in/out
        self.assertTrue(strict['NPU Name'].str.contains('NpuLinear', regex=False).all())
        self.assertTrue(strict['Bench Name'].str.contains('BenchLinear', regex=False).all())
        # the untagged parameter-gradient rows are kept in the report; the
        # cross-named bench counterparts never pair through the plain path
        grad_rows = result[result['NPU Name'].str.contains('parameters_grad', regex=False)]
        self.assertEqual(len(grad_rows), 2)
        self.assertFalse(grad_rows['Bench Name'].astype(str).str.contains('BenchLinear').any())


if __name__ == '__main__':
    unittest.main()
