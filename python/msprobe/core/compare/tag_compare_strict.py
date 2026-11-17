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

"""Strict tag-group comparison for the custom-tag (alias) entries.

The user marks a region of the model code with a tag on each dump::

    with tag("attn"):          # same statement on both sides, any call count
        out = my_attention(q, k)

Every dump entry inside the scope carries the tag as its ``alias`` field.
Entries wrapped by the same tag on the two dumps form one strict unit:
they pair by full entry name first (the same call of the same tagged
statement), then by their ordinal inside the tag group, so two
differently-named implementations of the same logical call - and a tagged
loop body executed several times - still pair iteration by iteration.
Paired entries contribute **all of their rows** - forward inputs, outputs,
backward inputs, outputs, parameter gradients and scalar configuration rows
alike - to the result table: nothing inside a tag scope is compressed away.
No tensor identity inference and no dataflow reconstruction is involved.

Cross-side pairing key: (tag, entry pair, slot of the row inside the entry).
Tensor ports pair by their slot regardless of dtype/shape - a mismatching
dtype or shape never breaks the pair, exactly like the name-based matching
of the plain path never gates on them. The native compare rules take over
downstream: a dtype mismatch N/As the bench columns of the pair and the
standard indicator checkers put the native error messages in the result
row (e.g. 'The dtype/shape of NPU and Bench are inconsistent.').

Failure modes therefore only have two values: a correct pair (Extraction
``slot`` / ``name``; disagreeing scalar/config rows join as Extraction
``config``) or a visible npu-anchored unpaired row with a logged reason.
Bench-side surplus rows are not shown: the native display is npu-anchored
and drops them, so the tag path logs their reason instead - nothing tagged
disappears silently, the table just never grows a bench-anchored row.

The strict rows carry the tag on the intermediate match frame (columns
Tag/Role/Extraction/Mode). The final table keeps the native column layout:
the tag is folded into the name columns as a ``(tag)name`` prefix, so
tagged rows stay recognizable without an extra leading column.

Alias routing (see also ``Comparator.compare_tag_statistics``, entered
automatically whenever the dumps carry alias fields):

- a tag marking the same non-zero number of forward calls on both sides ->
  strict unit (this module);
- every other tag (differing forward call counts, or present on one side
  only) and untagged rows -> normal name-based matching, with a warning
  naming the tags that could not be compared through the tag and why.
  Nothing is silently dropped and no boundary is ever guessed.
"""

import pandas as pd

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.log import logger

ROLE_PARAMETER_GRAD = 'parameter_grad'

EXTRACT_SLOT = 'slot'
EXTRACT_NAME = 'name'
EXTRACT_UNPAIRED = 'unpaired'
EXTRACT_CONFIG = 'config'

# dtypes dumped like tensors but never part of the tensor ports
_NON_TENSOR_DTYPES = frozenset(('torch.Size', 'slice', 'ellipsis', 'torch.ProcessGroup'))


def _param_grad_module_base(name):
    """``Module.blk.Linear.parameters_grad.0`` -> ``Module.blk.Linear``."""
    parts = name.split(Const.SEP)
    if len(parts) >= 3 and parts[-2] == Const.PARAMS_GRAD and parts[-1].isdigit():
        return Const.SEP.join(parts[:-2])
    return name


class _StrictEntry:
    """One dump entry with its tensor rows, grouped by row state (df order)."""

    __slots__ = ('name', 'rows')

    def __init__(self, name):
        self.name = name
        self.rows = {}  # state -> [df_index], df (execution) order preserved


class _StrictSide:
    """Alias/direction grouped view of one parsed dump dataframe."""

    def __init__(self, df):
        self.df = df
        self.groups = {}  # alias -> {direction: [_StrictEntry]}
        self.aliases = []  # distinct aliases in first-appearance order
        self.dtypes = []
        self.shapes = []
        self.summaries = []
        self.op_names = []
        self._build()

    def _build(self):
        df = self.df
        count = len(df)
        if count == 0:
            return
        self.dtypes = df[Const.DTYPE].tolist()
        self.shapes = df[Const.SHAPE].tolist()
        self.summaries = df[Const.SUMMARY].tolist() if Const.SUMMARY in df.columns else [None] * count
        self.op_names = df[CompareConst.OP_NAME].tolist() if CompareConst.OP_NAME in df.columns else [''] * count
        names = df[Const.API_ORIGIN_NAME].tolist()
        states = df[Const.STATE].tolist()
        directions = df[Const.DIRECTION].tolist()
        aliases = df[Const.ALIAS].tolist() if Const.ALIAS in df.columns else [None] * count

        entry_by_name = {}
        seen_alias = set()
        for i in range(count):
            name = names[i]
            entry = entry_by_name.get(name)
            if entry is None:
                entry = _StrictEntry(name)
                entry_by_name[name] = entry
                alias = aliases[i]
                if isinstance(alias, str) and alias:
                    if alias not in seen_alias:
                        seen_alias.add(alias)
                        self.aliases.append(alias)
                    by_direction = self.groups.setdefault(alias, {})
                    by_direction.setdefault(directions[i], []).append(entry)
            entry.rows.setdefault(states[i], []).append(i)

    def group(self, alias, direction):
        return self.groups.get(alias, {}).get(direction, [])

    def is_tensor_row(self, idx):
        """Same non-tensor semantics as the heuristic path: plain python
        scalars (``<class ...>`` dtype), special containers and 'None'
        summary placeholders never take part in the ports.
        """
        if idx is None:
            return False
        dtype = self.dtypes[idx]
        if not isinstance(dtype, str) or dtype.startswith('<class') or dtype in _NON_TENSOR_DTYPES:
            return False
        summary = self.summaries[idx]
        if not isinstance(summary, (list, tuple)) or len(summary) != len(Const.SUMMARY_METRICS_LIST):
            return False
        return not any(isinstance(value, str) for value in summary)


class TagCompareStrict:
    """Strict comparison of the tagged entries (tag units).

    Single responsibility: produce the strict full-row records (every row of
    every call the tags mark). The routing of the remaining rows (refused
    tags / plain name matching) lives in
    ``Comparator.compare_tag_statistics``, which calls
    :meth:`split_plain_rows` after :meth:`match_tag_dfs`.
    """

    def __init__(self, mode_config):
        self.mode_config = mode_config
        self.merge_cols = [
            CompareConst.OP_NAME,
            Const.DTYPE,
            Const.SHAPE,
            Const.SUMMARY,
            Const.STATE,
            Const.API_ORIGIN_NAME,
            Const.REQ_GRAD,
            Const.STACK_INFO,
        ]
        if mode_config.dump_mode == Const.ALL:
            self.merge_cols += [Const.DATA_NAME, Const.DIRTY_VALID_LEN]
        self._npu_side = None
        self._bench_side = None
        self._units = {}
        self._ordered_units = []

    # ------------------------------------------------------------------ entry
    def match_tag_dfs(self, npu_df, bench_df):
        self._npu_side = _StrictSide(npu_df)
        self._bench_side = _StrictSide(bench_df)
        self._units = self._discover_units()
        self._ordered_units = list(self._units)
        self._warn_plain_fallback_tags()

        records = []
        for unit in self._ordered_units:
            records.extend(self._compare_unit(unit))
        result = pd.DataFrame(records)
        if result.empty:
            self._warn_nothing_found()
        return result

    # ------------------------------------------------------------- routing api
    def has_tags(self):
        """Whether any custom tag (alias) exists in either dump."""
        return bool(self._npu_side.aliases or self._bench_side.aliases)

    def split_plain_rows(self, df):
        """Rows not consumed by strict units.

        These are left to the normal name-based matching: untagged entries
        and refused tags (differing call counts / one side only).
        """
        if df.empty or Const.ALIAS not in df.columns:
            return df.copy()
        mask = df[Const.ALIAS].isin(set(self._ordered_units))
        return df[~mask].copy()

    # -------------------------------------------------------------- discovery
    def _discover_units(self):
        """Tag -> unit, iff it marks the same non-zero number of forward
        calls on both sides.

        A tag with differing forward call counts or present on one side
        only cannot establish a one-to-one correspondence and is left to
        normal matching with a warning.
        """
        units = {}
        seen = set()
        for side in (self._npu_side, self._bench_side):
            for alias in side.aliases:
                if alias in seen:
                    continue
                seen.add(alias)
                if self._is_paired_group(alias):
                    units[alias] = {}
        return units

    def _is_paired_group(self, alias):
        """The alias marks the same non-zero number of forward entries on
        both sides.

        The forward calls pair by name first, then by group ordinal, so a
        tagged loop body executed several times pairs iteration by
        iteration. The backward direction needs no gate: surplus backward
        entries simply stay visible as unpaired rows.
        """
        npu_count = len(self._npu_side.group(alias, Const.FORWARD))
        bench_count = len(self._bench_side.group(alias, Const.FORWARD))
        return npu_count > 0 and npu_count == bench_count

    def _warn_plain_fallback_tags(self):
        """Warn about every tag that could not be compared through the tag:
        its entries fall back to normal name-based matching (nothing is
        silently dropped, but the tag adds no pairing for them).
        """
        warned = set()
        for side in (self._npu_side, self._bench_side):
            for alias in side.aliases:
                if alias in warned:
                    continue
                warned.add(alias)
                if alias in self._ordered_units:
                    continue  # compared through the tag
                n_npu = len(self._npu_side.group(alias, Const.FORWARD))
                n_bench = len(self._bench_side.group(alias, Const.FORWARD))
                logger.warning(
                    f"[tag compare] tag '{alias}' marks {n_npu} npu / {n_bench} bench forward calls "
                    f"and is not compared through the tag: the number of tagged calls must be "
                    f"the same (and non-zero) on both sides. Its entries fall back to normal "
                    f"name-based matching; check that both dumps tagged the same statements "
                    f"with matching loop counts."
                )

    # --------------------------------------------------------------- analysis
    def _compare_unit(self, unit):
        """Full-row comparison of one self-only strict unit.

        Every direction (forward / backward) of the tagged calls is compared
        independently: the group's entries are paired (name first, then
        group ordinal - a tagged loop body pairs iteration by iteration) and
        every row of each pair - forward/backward inputs, outputs and scalar
        configuration rows alike - goes to the result table. A missing side
        keeps its rows visible (unpaired).
        """
        records = []
        for direction in (Const.FORWARD, Const.BACKWARD):
            npu_entries = self._npu_side.group(unit, direction)
            bench_entries = self._bench_side.group(unit, direction)
            self._emit_group_full_rows(records, unit, unit, direction, npu_entries, bench_entries)
        self._emit_param_grads(records, unit)
        return records

    def _emit_group_full_rows(self, records, unit, alias, direction, npu_entries, bench_entries):
        if not npu_entries and not bench_entries:
            return
        if len(npu_entries) != len(bench_entries):
            # within a formed unit this can only happen in the backward
            # direction (e.g. one side's loss consumed no tagged output)
            logger.warning(
                f"[tag compare] strict unit '{unit}' tag '{alias}' ({direction}): tagged entry "
                f"counts differ ({len(npu_entries)} vs {len(bench_entries)}); the npu-side surplus "
                f"entries stay visible unpaired, the bench-side surplus entries are not shown (the "
                f"compare table is npu-anchored, exactly like the native display). "
                f"Check that both dumps tagged the same statements (loop counts "
                f"may diverge with accuracy)."
            )
        pairs, npu_left, _ = self._match_entries(npu_entries, bench_entries)
        surplus_slots = []  # (bench entry name, slot) of bench-only slots, aggregated below
        for npu_entry, bench_entry in pairs:
            self._pair_entry_all_rows(records, unit, alias, direction, npu_entry, bench_entry, surplus_slots)
        for entry in npu_left:
            self._emit_npu_side(records, unit, alias, direction, entry)
        if surplus_slots:
            # one aggregated warning per group direction, not one per slot: a
            # tagged loop with a per-iteration bench surplus must not flood
            # the log channel and drown the entry-count warnings that matter.
            first_entry, first_slot = surplus_slots[0]
            logger.warning(
                f"[tag compare] strict unit '{unit}' tag '{alias}' ({direction}): {len(surplus_slots)} "
                f"bench-side slot(s) have no npu counterpart and are not shown (the compare table is "
                f"npu-anchored, exactly like the native display; first example: slot {first_slot} of the "
                f"bench entry '{first_entry}'). Check that both dumps tagged structurally identical "
                f"statements."
            )

    @staticmethod
    def _match_entries(npu_entries, bench_entries):
        """Pair the entries of one tag group: by full name first (the same call
        of the same tagged statement), then by group ordinal (the global call
        counters may be skewed by divergent code before the tagged region).
        Entry names are unique within one step, so the name matching is exact.
        """
        bench_by_name = {entry.name: idx for idx, entry in enumerate(bench_entries)}
        used = set()
        pairs = []
        unmatched_npu = []
        for entry in npu_entries:
            idx = bench_by_name.get(entry.name)
            if idx is not None and idx not in used:
                used.add(idx)
                pairs.append((entry, bench_entries[idx]))
            else:
                unmatched_npu.append(entry)
        unmatched_bench = [entry for idx, entry in enumerate(bench_entries) if idx not in used]
        pairs.extend(zip(unmatched_npu, unmatched_bench))
        return pairs, unmatched_npu[len(unmatched_bench) :], unmatched_bench[len(unmatched_npu) :]

    def _pair_entry_all_rows(self, records, unit, alias, direction, npu_entry, bench_entry, surplus_slots):
        """Pair every row (all states, all slots) of two matched entries.

        Tensor rows pair regardless of dtype/shape - and even when the
        other side of the slot is a scalar/config row: a mismatch never
        breaks the pair, exactly like the name-based matching of the plain
        path; the native compare rules take over downstream and report
        the row (bench columns N/A'd on a dtype mismatch, native error
        messages in the result table). Scalar/config rows pair regardless
        of value; a disagreeing value additionally logs the configuration
        warning.
        """
        for state in self._ordered_states((npu_entry, bench_entry)):
            npu_rows = npu_entry.rows.get(state, [])
            bench_rows = bench_entry.rows.get(state, [])
            for slot in range(max(len(npu_rows), len(bench_rows))):
                npu_idx = npu_rows[slot] if slot < len(npu_rows) else None
                bench_idx = bench_rows[slot] if slot < len(bench_rows) else None
                npu_ok = self._npu_side.is_tensor_row(npu_idx)
                bench_ok = self._bench_side.is_tensor_row(bench_idx)
                if npu_ok and bench_ok:
                    records.append(self._make_record(unit, alias, direction, EXTRACT_SLOT, npu_idx, bench_idx))
                elif npu_idx is None or bench_idx is None:
                    # one side has no row in this slot: the npu row stays
                    # visible (native shows npu-anchored rows with the bench
                    # columns N/A'd); a bench-only surplus row is not shown -
                    # the native display drops it - and is reported once per
                    # group direction (see _emit_group_full_rows) instead,
                    # so nothing tagged disappears silently.
                    if npu_idx is not None:
                        records.append(self._make_record(unit, alias, direction, EXTRACT_UNPAIRED, npu_idx, None))
                    if bench_idx is not None:
                        surplus_slots.append((bench_entry.name, slot))
                elif not npu_ok and not bench_ok:
                    # non-tensor rows on both sides (scalars, None, torch.Size, ...):
                    # configuration rows - always shown; a disagreeing value
                    # additionally warns (it can masquerade as a precision problem)
                    extraction = EXTRACT_SLOT
                    if self._row_key(self._npu_side, npu_idx) != self._row_key(self._bench_side, bench_idx):
                        logger.warning(
                            f"[tag compare] strict unit '{unit}' tag '{alias}': slot {slot} of entries "
                            f"'{npu_entry.name}' / '{bench_entry.name}' carries different configuration "
                            f"values (npu {self._row_repr(self._npu_side, npu_idx)} vs bench "
                            f"{self._row_repr(self._bench_side, bench_idx)}); a differing operator "
                            f"configuration can masquerade as a precision problem. Check the scalar/"
                            f"config arguments of the tagged calls on both sides."
                        )
                        extraction = EXTRACT_CONFIG
                    records.append(self._make_record(unit, alias, direction, extraction, npu_idx, bench_idx))
                else:
                    # tensor on one side, scalar/container on the other: still a
                    # pair - the native compare never gates a name match on the
                    # content type, and its dtype condition (a tensor dtype never
                    # equals a scalar type) N/As the bench columns downstream, so
                    # the row reads exactly like the native display.
                    records.append(self._make_record(unit, alias, direction, EXTRACT_SLOT, npu_idx, bench_idx))

    @staticmethod
    def _ordered_states(entries):
        """States of the entries' rows in first-appearance (dump) order."""
        first = {}
        for entry in entries:
            for state, idxs in entry.rows.items():
                if state not in first or idxs[0] < first[state]:
                    first[state] = idxs[0]
        return sorted(first, key=first.get)

    @staticmethod
    def _row_key(side, idx):
        """Comparable identity of a non-tensor row: the dumped value is kept
        in the summary metrics (see ``gen_op_item``), so (dtype, shape,
        summary) identifies the configuration value.
        """
        if idx is None:
            return None
        return (str(side.dtypes[idx]), str(side.shapes[idx]), str(side.summaries[idx]))

    def _row_repr(self, side, idx):
        if idx is None:
            return '<missing row>'
        return f"{side.dtypes[idx]} {side.summaries[idx]}"

    def _emit_npu_side(self, records, unit, alias, direction, entry):
        """Every row of an npu-side unpaired entry stays visible - the
        native display shows npu-anchored rows with the bench columns N/A'd.
        Bench-side surplus entries never reach this path: the native display
        drops them, so they are reported with a logged reason instead.
        """
        for state in self._ordered_states((entry,)):
            for idx in entry.rows.get(state, []):
                records.append(self._make_record(unit, alias, direction, EXTRACT_UNPAIRED, idx, None))

    def _emit_param_grads(self, records, unit):
        npu_entries = self._npu_side.group(unit, Const.PARAMS_GRAD)
        bench_entries = self._bench_side.group(unit, Const.PARAMS_GRAD)
        if not npu_entries and not bench_entries:
            return
        npu_keyed = self._param_grad_rows(self._npu_side, npu_entries)
        bench_keyed = self._param_grad_rows(self._bench_side, bench_entries)
        dropped_bench_keys = []
        for key, npu_idxs in npu_keyed.items():
            bench_idxs = bench_keyed.pop(key, [])
            for npu_idx, bench_idx in zip(npu_idxs, bench_idxs):
                records.append(self._make_record(unit, unit, ROLE_PARAMETER_GRAD, EXTRACT_NAME, npu_idx, bench_idx))
            for npu_idx in npu_idxs[len(bench_idxs) :]:
                records.append(self._make_record(unit, unit, ROLE_PARAMETER_GRAD, EXTRACT_UNPAIRED, npu_idx, None))
            if len(bench_idxs) > len(npu_idxs):
                dropped_bench_keys.append(key)
        for key, bench_idxs in bench_keyed.items():
            if bench_idxs:
                dropped_bench_keys.append(key)
        if dropped_bench_keys:
            # bench-side surplus parameter gradients are not shown: the native
            # display is npu-anchored and drops them - log the reason instead.
            logger.warning(
                f"[tag compare] strict unit '{unit}': bench-side parameter gradients with no npu "
                f"counterpart are not shown (the compare table is npu-anchored, exactly like the "
                f"native display): {', '.join(f'{base}.{param}' for base, param in dropped_bench_keys)}. "
                f"Check that both dumps tagged the same parameters."
            )

    @staticmethod
    def _param_grad_rows(side, entries):
        """(module base, param name) -> [row indices], occurrence order kept."""
        keyed = {}
        for entry in entries:
            base = _param_grad_module_base(entry.name)
            for idx in entry.rows.get(Const.PARAMS_GRAD, []):
                param = side.op_names[idx].rsplit(Const.SEP, 1)[-1]
                keyed.setdefault((base, param), []).append(idx)
        return keyed

    # ----------------------------------------------------------------- record
    def _make_record(self, unit, alias, direction, extraction, npu_idx, bench_idx):
        record = {}
        npu_series = self._npu_side.df.iloc[npu_idx] if npu_idx is not None else None
        bench_series = self._bench_side.df.iloc[bench_idx] if bench_idx is not None else None
        for col in self.merge_cols:
            record[col + '_x'] = npu_series[col] if npu_series is not None else CompareConst.N_A
            record[col + '_y'] = bench_series[col] if bench_series is not None else CompareConst.N_A
        # dump row numbers of both sides: the caller sorts the merged result
        # back into the execution order with them. Every record is
        # npu-anchored (the native display never grows a bench-only row),
        # so the npu row number drives the sort; the bench number stays
        # for debugging/inspection.
        record[CompareConst.NPU_PARSE_ORDER] = npu_idx
        record[CompareConst.BENCH_PARSE_ORDER] = bench_idx if bench_idx is not None else float('nan')
        record[CompareConst.TAG] = alias
        record[CompareConst.ROLE] = direction
        record[CompareConst.TAG_EXTRACTION] = extraction
        record[CompareConst.TAG_MODE] = CompareConst.TAG_MODE_STRICT
        return record

    # --------------------------------------------------------------- warnings
    def _warn_nothing_found(self):
        all_aliases = sorted({alias for side in (self._npu_side, self._bench_side) for alias in side.aliases})
        if not all_aliases:
            logger.warning(
                "[tag compare] no entry with an 'alias' (custom tag) field was found in either dump.json. "
                "Dump the data inside 'with tag(...)' scopes and retry."
            )
