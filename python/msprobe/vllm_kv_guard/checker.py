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

"""Compare scheduler bindings with visible execution plans, without allocator state."""

from msprobe.vllm_kv_guard.events import emit


class KVGuardError(RuntimeError):
    """A visible KV boundary disagrees with its scheduling plan."""


class Guard:
    def __init__(self, action, rank=None):
        self.action, self.rank, self.failure = action, rank, None

    def fail(self, reason, **fields):
        if self.failure is None:
            self.failure = reason
        emit("kv.violation", reason=reason, rank=self.rank, evidence="boundary_plan", **fields)

    def check(self):
        if self.action == "raise" and self.failure is not None:
            raise KVGuardError(f"KV guard: {self.failure}")

    def spec(self, values, rows=None):
        """Validate native counts; older PP results have no current-step budget."""

        def fail(reason, **fields):
            self.fail(reason, trace_id=values.get("trace_id"), **fields)

        indices, sampled, rejected = (values[name] for name in ("idx_mapping", "num_sampled", "num_rejected"))
        starts = values["query_start_loc"]
        if starts is not None:
            starts = starts[: len(indices) + 1]
        if len(indices) != len(sampled) or len(indices) != len(rejected):
            fail("sample_count_shape_mismatch")
            return
        if starts is not None and (len(starts) != len(indices) + 1 or starts[0] != 0 or starts != sorted(starts)):
            fail("sample_query_range_mismatch")
            return
        for row, (index, produced, denied) in enumerate(zip(indices, sampled, rejected)):
            if index < 0:
                continue  # Native post_update skips masked rows, including their count values.
            if produced < 0 or denied < 0 or produced > values["sample_capacity"]:
                fail("sample_count_invalid", state_index=index)
            if starts is not None and denied > starts[row + 1] - starts[row]:
                fail("rejected_count_outside_query", state_index=index)
            if rows is not None:
                if index not in rows:
                    fail("sample_request_mismatch", state_index=index)
                elif starts is not None and starts[row + 1] - starts[row] > rows[index][1]:
                    fail("sample_query_exceeds_budget", request_id=rows[index][0])

    def audit(self, values, plan, layout):
        """Audit frozen host values; the worker raises only at its result boundary."""

        def fail(reason, **fields):
            self.fail(reason, trace_id=values.get("trace_id"), **fields)

        if layout.get("pcp_size", 1) != 1:
            return  # PCP local rows can repeat; parallel() audits the native global context.
        requests = {req: (computed, count, groups) for req, computed, count, groups in plan}
        ids, starts = values["req_ids"], values["query_start_loc"]
        positions, lengths = values["positions"], values["seq_lens"]
        tokens = values["num_tokens"]
        if len(ids) != len(set(ids)) or not set(ids).issubset(requests):
            fail("batch_request_mismatch")
            return
        if (
            len(starts) != len(ids) + 1  # pylint: disable=too-many-boolean-expressions
            or not starts
            or starts[0] != 0
            or not (0 <= starts[-1] <= tokens)
            or starts != sorted(starts)
            or (not layout.get("dynamic", False) and starts[-1] != tokens)
            or len(positions) != tokens
            or len(lengths) != len(ids)
        ):
            fail("batch_token_range_mismatch")
            return
        if set(ids) != set(requests) and not layout.get("dynamic", False):
            fail("batch_request_mismatch")
        if len(values["groups"]) != len(layout["groups"]):
            fail("cache_group_count_mismatch")
            return
        cp_size, rank = layout.get("cp_size", 1), layout.get("cp_rank", 0)
        interleave = layout.get("cp_interleave", 1)
        local_lengths = values.get("dcp_local_seq_lens")
        if local_lengths is not None and local_lengths != [
            length // (cp_size * interleave) * interleave
            + min(interleave, max(0, length % (cp_size * interleave) - rank * interleave))
            for length in lengths
        ]:
            fail("dcp_read_length_mismatch")
        for row, req in enumerate(ids):
            computed, budget, _ = requests[req]
            begin, end = starts[row : row + 2]
            count, actual = end - begin, positions[begin:end]
            if layout.get("dynamic", False):
                valid = count <= budget and (not actual or 0 <= actual[0] <= computed)
            else:
                valid = count == budget and (not actual or actual[0] == computed)
            if not valid or (actual and actual != list(range(actual[0], actual[0] + count))):
                fail("execution_range_mismatch", request_id=req)
            if actual and lengths[row] != actual[-1] + 1:
                fail("read_length_mismatch", request_id=req)
        for group, (data, spec) in enumerate(zip(values["groups"], layout["groups"])):
            size, kernel = spec["block_size"], spec["kernel_block_size"]
            if size <= 0 or kernel <= 0 or size % kernel:
                continue
            factor = size // kernel
            tables, slots = data["block_table"], data["slot_mapping"]
            if len(tables) != len(ids) or len(slots) != tokens:
                fail("cache_plan_shape_mismatch", group_id=group)
                continue
            if any(slot >= 0 for slot in slots[starts[-1] :]):
                fail("padded_write_target", group_id=group)
            for row, req in enumerate(ids):
                groups = requests[req][2]
                if group >= len(groups):
                    fail("scheduler_group_missing", request_id=req, group_id=group)
                    continue
                blocks, table = groups[group], tables[row]
                expanded = [block * factor + offset for block in blocks for offset in range(factor)]
                # Native window masking may replace retired prefix entries by zero.
                if len(table) < len(expanded) or any(value not in (0, want) for value, want in zip(table, expanded)):
                    fail("read_block_mapping_mismatch", request_id=req, group_id=group)
                begin, end = starts[row : row + 2]
                for token in range(begin, end):
                    position = positions[token]
                    virtual, offset = divmod(position, size * cp_size)
                    local = offset // interleave % cp_size == rank
                    offset = offset // (interleave * cp_size) * interleave + offset % interleave
                    if not spec.get("slot_mapping_enabled", True) or not local:
                        expected = -1
                    elif virtual < 0 or virtual >= len(blocks):
                        fail("write_outside_binding", request_id=req, group_id=group)
                        break
                    else:
                        block = blocks[virtual]
                        if (
                            block < 0
                            or block >= layout["num_blocks"]
                            or (block == 0 and not layout["allow_null_block"])
                        ):
                            fail(
                                "null_block_write" if block == 0 else "write_outside_cache",
                                request_id=req,
                                group_id=group,
                                token_index=token,
                                block_id=block,
                                num_blocks=layout["num_blocks"],
                            )
                            break
                        expected = blocks[virtual] * size + offset
                        column = (virtual * size + offset) // kernel
                        kernel_block = blocks[virtual] * factor + offset // kernel
                        if column >= len(table) or table[column] != kernel_block:
                            fail("write_block_mapping_mismatch", request_id=req, group_id=group)
                            break
                    if slots[token] != expected:
                        fail(
                            "write_target_mismatch",
                            request_id=req,
                            group_id=group,
                            token_index=token,
                            expected_slot=expected,
                            actual_slot=slots[token],
                        )
                        break

    def parallel(self, values, plan, layout):
        """Check native PCP index maps and gathered write frames, without repartitioning."""

        def fail(reason, **fields):
            self.fail(reason, trace_id=values.get("trace_id"), **fields)

        ids, tokens = values["global_req_ids"], values["global_num_tokens"]
        positions, starts = values["global_positions"], values["global_query_start_loc"]
        tables, slots = values["global_block_tables"], values["global_slot_mappings"]
        global_values = dict(
            req_ids=ids,
            num_tokens=tokens,
            positions=positions,
            query_start_loc=starts,
            seq_lens=values["global_seq_lens"],
            trace_id=values.get("trace_id"),
            groups=[dict(block_table=table, slot_mapping=slot[:tokens]) for table, slot in zip(tables, slots)],
        )
        self.audit(global_values, plan, {**layout, "pcp_size": 1})
        if (
            len(starts) != len(ids) + 1  # pylint: disable=too-many-boolean-expressions
            or not starts
            or starts[0] != 0
            or not 0 <= starts[-1] <= tokens
            or starts != sorted(starts)
            or len(positions) != tokens
            or len(set(ids)) != len(ids)
            or len(tables) != len(slots)
            or len(tables) != len(layout["groups"])
            or any(len(table) != len(ids) for table in tables)
        ):
            return
        local = values.get("local")
        indices = values["local_token_indices"]
        if local is None or indices is None:
            return
        local_ids, query = local["req_ids"], local["query_start_loc"]
        if (
            len(query) != len(local_ids) + 1  # pylint: disable=too-many-boolean-expressions
            or not query
            or query[0] != 0
            or (local["num_tokens"] != 0 and query[-1] != local["num_tokens"])
            or query != sorted(query)
            or len(indices) != local["num_tokens"]
            or len(local["positions"]) != len(indices)
            or len(local["seq_lens"]) != len(local_ids)
            or any(index < 0 or index >= values["global_num_tokens_after_padding"] for index in indices)
            or not set(local_ids).issubset(ids)
        ):
            fail("pcp_local_range_mismatch")
            return
        # FULL graph may give an empty rank a padded uniform query such as [0, 1].
        rows = {req: row for row, req in enumerate(ids)}
        owners = [req for row, req in enumerate(ids) for _ in range(starts[row + 1] - starts[row])]
        row_map = values["local_to_global_req_indices"]
        if row_map is not None and list(row_map) != [rows[req] for req in local_ids]:
            fail("pcp_request_mapping_mismatch")
        cp, rank, interleave = layout["cp_size"], layout["cp_rank"], layout["cp_interleave"]
        local_lengths = local["dcp_local_seq_lens"]
        if local_lengths is not None and local_lengths != [
            length // (cp * interleave) * interleave
            + min(interleave, max(0, length % (cp * interleave) - rank * interleave))
            for length in local["seq_lens"]
        ]:
            fail("dcp_read_length_mismatch")
        for row, req in enumerate(local_ids):
            begin, end = query[row : row + 2]
            selected = indices[begin:end]
            actual = local["positions"][begin:end]
            if any(index >= starts[-1] for index in selected):
                continue
            if actual != [positions[index] for index in selected] or any(owners[index] != req for index in selected):
                fail("pcp_token_mapping_mismatch", request_id=req)
            if local["seq_lens"][row] != (actual[-1] + 1 if actual else 0):
                fail("pcp_read_length_mismatch", request_id=req)
        gather, mask = values["padded_gather_idx"], values["gathered_kv_write_mask"]
        if gather is None:
            return
        if mask is not None and len(mask) != len(gather):
            fail("pcp_write_mask_shape_mismatch")
            return
        if local["groups"] is None:
            return
        if len(local["groups"]) != len(tables):
            fail("pcp_cache_group_count_mismatch")
            return
        bindings = {req: groups for req, _computed, _count, groups in plan}
        for group, data in enumerate(local["groups"]):
            if (
                group >= len(tables)
                or len(data["block_table"]) != len(local_ids)
                or any(index < 0 or index >= len(slots[group]) for index in gather)
            ):
                fail("pcp_gather_index_mismatch", group_id=group)
                continue
            for row, req in enumerate(local_ids):
                expected = tables[group][rows[req]]
                actual = data["block_table"][row]
                groups, spec = bindings.get(req, ()), layout["groups"][group]
                if group >= len(groups) or spec["kernel_block_size"] <= 0:
                    continue
                limit = len(groups[group]) * spec["block_size"] // spec["kernel_block_size"]
                if len(actual) < limit or any(
                    value not in (0, want) for value, want in zip(actual[:limit], expected[:limit])
                ):
                    fail("pcp_read_block_mapping_mismatch", request_id=req, group_id=group)
            if data["slot_mapping"] is not None and data["slot_mapping"] != [
                slots[group][index] if mask is None or mask[token] else -1 for token, index in enumerate(gather)
            ]:
                fail("pcp_write_frame_mismatch", group_id=group)


def bind(output, bindings, guard=None):
    """Materialize only public binding deltas; CoW destinations are already bound."""

    def fail(reason, **fields):
        if guard is not None:
            guard.fail(reason, trace_id=getattr(output, "_msprobe_kv_trace_id", None), **fields)

    if sum(output.num_scheduled_tokens.values()) != output.total_num_scheduled_tokens:
        fail("scheduled_token_total_mismatch")
    for req in set(output.finished_req_ids) | set(output.preempted_req_ids or ()):
        bindings.pop(req, None)
    computed = {}
    for req in output.scheduled_new_reqs:
        bindings[req.req_id] = tuple(tuple(group) for group in req.block_ids)
        computed[req.req_id] = req.num_computed_tokens
    cached = output.scheduled_cached_reqs
    if not (len(cached.req_ids) == len(cached.num_computed_tokens) == len(cached.new_block_ids)):
        fail("cached_request_array_length_mismatch")
    for req, start, added in zip(cached.req_ids, cached.num_computed_tokens, cached.new_block_ids):
        computed[req] = start
        if req in cached.resumed_req_ids:
            bindings[req] = tuple(tuple(group) for group in added) if added is not None else ()
        elif added is not None:
            old = bindings.get(req, ())
            if old and len(old) == len(added):
                bindings[req] = tuple(prior + tuple(extra) for prior, extra in zip(old, added))
            elif not old:
                bindings[req] = tuple(tuple(group) for group in added)
            else:
                fail("scheduler_group_count_mismatch", request_id=req)
    for req, count in output.num_scheduled_tokens.items():
        if computed.get(req, -1) < 0 or count <= 0:
            fail("scheduled_range_invalid", request_id=req)
        if req not in bindings:
            fail("scheduler_binding_missing", request_id=req)
    output._msprobe_kv_plan = tuple(
        (req, computed.get(req, -1), count, bindings.get(req, ())) for req, count in output.num_scheduled_tokens.items()
    )
