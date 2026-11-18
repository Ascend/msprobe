from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from msprobe.vllm_kv_guard import checker
from msprobe.vllm_kv_guard.checker import Guard, KVGuardError, bind


class TestGuard(TestCase):
    def setUp(self):
        patcher = patch.object(checker, "emit")
        self.mock_emit = patcher.start()
        self.addCleanup(patcher.stop)
        self.guard = Guard("log", rank=1)
        self.values = dict(req_ids=["a", "b"], query_start_loc=[0, 2, 4], positions=[2, 3, 0, 1],
                           seq_lens=[4, 2], num_tokens=4, trace_id="step",
                           groups=[dict(block_table=[[1], [2]], slot_mapping=[6, 7, 8, 9])])
        self.plan = (("a", 2, 2, ((1,),)), ("b", 0, 2, ((2,),)))
        self.layout = dict(groups=[dict(block_size=4, kernel_block_size=4)], num_blocks=8,
                           allow_null_block=False, cp_size=1, cp_rank=0, cp_interleave=1)
        self.parallel = {f"global_{key}": self.values[key] for key in
                         ("req_ids", "num_tokens", "positions", "query_start_loc", "seq_lens")}
        self.parallel.update(global_block_tables=[[[1], [2]]], global_slot_mappings=[[6, 7, 8, 9]],
                             global_num_tokens_after_padding=4,
                             local={**self.values, "dcp_local_seq_lens": None}, local_token_indices=[0, 1, 2, 3],
                             local_to_global_req_indices=[0, 1], padded_gather_idx=[0, 1, 2, 3],
                             gathered_kv_write_mask=[True] * 4)
        self.parallel_layout = {**self.layout, "pcp_size": 2}

    def reasons(self):
        self.assertTrue(all(call.args == ("kv.violation",) for call in self.mock_emit.call_args_list))
        return [call.kwargs["reason"] for call in self.mock_emit.call_args_list]

    def test_failure_latches_first_reason_and_only_raise_blocks(self):
        self.guard.check()
        self.guard.fail("first", request_id="a")
        self.guard.fail("second")
        self.guard.check()
        self.assertEqual(self.reasons(), ["first", "second"])
        self.assertEqual(self.mock_emit.call_args_list[0].kwargs["rank"], 1)
        self.assertEqual(self.mock_emit.call_args_list[0].kwargs["evidence"], "boundary_plan")
        self.guard.action = "raise"
        with self.assertRaisesRegex(KVGuardError, "first"):
            self.guard.check()

    def enable_slots(self):
        self.guard.slot = True
        self.values.update(num_tokens_after_padding=4, padded_slot_mappings=[[]])

    def test_slot_healthy_reordered_chunked_and_cross_block_writes(self):
        self.enable_slots()
        self.guard.audit(self.values, self.plan, self.layout)
        self.values.update(req_ids=["b", "a"], query_start_loc=[0, 2, 4], positions=[0, 1, 3, 4], seq_lens=[2, 5])
        self.values["groups"] = [dict(block_table=[[2, 0], [1, 3]], slot_mapping=[8, 9, 7, 12])]
        self.guard.audit(self.values, (("a", 3, 2, ((1, 3),)), self.plan[1]), self.layout)
        self.assertEqual(self.reasons(), [])

    def test_slot_rejects_self_consistent_cross_request_claims(self):
        self.enable_slots()
        self.values.update(positions=[0, 1, 0, 1], seq_lens=[2, 2])
        self.values["groups"] = [dict(block_table=[[1], [1]], slot_mapping=[4, 5, 4, 5])]
        plan = (("a", 0, 2, ((1,),)), ("b", 0, 2, ((1,),)))
        self.guard.audit(self.values, plan, self.layout)
        collisions = [call.kwargs for call in self.mock_emit.call_args_list]
        self.assertTrue(collisions)
        self.assertEqual(set(self.reasons()), {"slot_collision"})
        self.assertEqual({c["source"] for c in collisions}, {"scheduler", "execution"})
        self.assertTrue(all(c["request_id"] == "b" for c in collisions))
        self.assertTrue(all("other_request_id" not in c and "other_token_position" not in c for c in collisions))
        self.assertTrue(all(c["audit"] == "slot" for c in collisions))

    def test_slot_collision_is_checked_after_an_earlier_target_mismatch(self):
        self.enable_slots()
        self.values["groups"][0]["slot_mapping"] = [31, 7, 7, 9]
        self.guard.audit(self.values, self.plan, self.layout)
        self.assertIn("write_target_mismatch", self.reasons())
        collision = next(call.kwargs for call in self.mock_emit.call_args_list
                         if call.kwargs["reason"] == "slot_collision")
        self.assertEqual((collision["request_id"], collision["slot"]), ("b", 7))

    def test_slot_shared_prefix_reads_and_distinct_groups_do_not_collide(self):
        self.enable_slots()
        self.values.update(positions=[4, 5, 4, 5], seq_lens=[6, 6])
        group = dict(block_table=[[1, 2], [1, 3]], slot_mapping=[8, 9, 12, 13])
        self.values["groups"] = [deepcopy(group), deepcopy(group)]
        self.values["padded_slot_mappings"] = [[], []]
        self.layout["groups"] *= 2
        plan = (("a", 4, 2, ((1, 2), (1, 2))), ("b", 4, 2, ((1, 3), (1, 3))))
        self.guard.audit(self.values, plan, self.layout)
        self.assertEqual(self.reasons(), [])

    def test_slot_uses_scheduler_positions_even_when_execution_is_self_consistent(self):
        self.enable_slots()
        self.values.update(positions=[1, 2, 0, 1], seq_lens=[3, 2])
        self.values["groups"][0]["slot_mapping"] = [5, 6, 8, 9]
        self.guard.audit(self.values, self.plan, self.layout)
        self.assertIn("execution_range_mismatch", self.reasons())
        mismatch = next(c.kwargs for c in self.mock_emit.call_args_list if c.kwargs["reason"] == "write_target_mismatch")
        self.assertEqual((mismatch["expected_slot"], mismatch["actual_slot"]), (6, 5))

    def test_slot_capacity_and_tail_authorization(self):
        for slot in (-2, 32, 6):
            with self.subTest(slot=slot):
                self.mock_emit.reset_mock()
                self.guard = Guard("log", rank=1)
                self.enable_slots()
                self.values["groups"][0]["slot_mapping"][1] = slot
                self.guard.audit(self.values, self.plan, self.layout)
                self.assertIn("write_target_mismatch", self.reasons())
                self.assertEqual("slot_out_of_capacity" in self.reasons(), slot in (-2, 32))

    def test_slot_repeated_block_within_request_is_duplicate_write(self):
        self.enable_slots()
        self.values.update(req_ids=["a"], query_start_loc=[0, 6], positions=list(range(6)), seq_lens=[6],
                           num_tokens=6, num_tokens_after_padding=6)
        self.values["groups"] = [dict(block_table=[[1, 1]], slot_mapping=[4, 5, 6, 7, 4, 5])]
        self.guard.audit(self.values, (("a", 0, 6, ((1, 1),)),), self.layout)
        self.assertIn("duplicate_slot_write", self.reasons())
        self.assertNotIn("slot_collision", self.reasons())

    def test_slot_full_padding_checks_every_bucket_row(self):
        self.enable_slots()
        self.values.update(num_tokens_after_padding=8, padded_slot_mappings=[[-1, -1, -1, -1]])
        self.guard.audit(self.values, self.plan, self.layout)
        self.assertEqual(self.reasons(), [])
        self.values["padded_slot_mappings"][0][-1] = 7
        self.guard.audit(self.values, self.plan, self.layout)
        self.assertEqual(self.reasons(), ["padded_write_target"])
        self.assertEqual(self.mock_emit.call_args.kwargs["slot_mapping_scope"], "graph_padding")

    def test_slot_metadata_and_layout_failures_follow_action_at_check(self):
        cases = [({}, {"allow_null_block": True}), ({}, {"num_blocks": 0}), ({}, {"groups": []}),
                 ({}, {"groups": [dict(block_size=4, kernel_block_size=3)]}),
                 ({}, {"groups": [dict(block_size=4, kernel_block_size=4, slot_mapping_enabled=False)]}),
                 ({"padded_slot_mappings": None}, {}), ({"num_tokens_after_padding": 3}, {})]
        for action in ("log", "raise"):
            for values, layout in cases:
                with self.subTest(action=action, values=values, layout=layout):
                    self.enable_slots()
                    self.guard = Guard(action, slot=True)
                    self.guard.audit({**self.values, **values}, self.plan, {**self.layout, **layout})
                    self.assertIsNotNone(self.guard.failure)
                    if action == "raise":
                        for _ in range(2):
                            with self.assertRaisesRegex(KVGuardError, self.guard.failure):
                                self.guard.check()
                    else:
                        self.guard.check()

    def test_slot_raise_latches_and_log_continues(self):
        self.enable_slots()
        self.values["groups"][0]["slot_mapping"][0] = 9
        self.guard.audit(self.values, self.plan, self.layout)
        first = self.guard.failure
        self.guard.check()
        self.guard.action = "raise"
        for _ in range(2):
            with self.assertRaisesRegex(KVGuardError, first):
                self.guard.check()

    def test_slot_dcp_checks_rank_local_targets_capacity_and_collisions(self):
        for rank, slots in ((0, [5, -1, 8, -1]), (1, [-1, 5, -1, 8])):
            with self.subTest(rank=rank):
                self.mock_emit.reset_mock()
                self.enable_slots()
                self.guard = Guard("log", slot=True)
                values = deepcopy(self.values)
                values["groups"][0]["slot_mapping"] = slots.copy()
                layout = {**self.layout, "cp_size": 2, "cp_rank": rank, "dcp_size": 2}
                self.guard.audit(values, self.plan, layout)
                self.assertEqual(self.reasons(), [])
                values["groups"][0]["slot_mapping"][2 + rank] = 5
                self.guard.audit(values, self.plan, layout)
                self.assertIn("slot_collision", self.reasons())
                values["groups"][0]["slot_mapping"][rank] = 32
                self.guard.audit(values, self.plan, layout)
                self.assertIn("slot_out_of_capacity", self.reasons())

    def test_slot_pcp_reuses_global_local_gather_and_padding_checks(self):
        for rank, slots, indices, positions, lengths in (
            (0, [5, -1, 8, -1], [0, 2], [2, 0], [3, 1]),
            (1, [-1, 5, -1, 8], [1, 3], [3, 1], [4, 2]),
        ):
            with self.subTest(rank=rank):
                self.mock_emit.reset_mock()
                self.guard = Guard("log", slot=True)
                values = deepcopy(self.parallel)
                values.update(global_slot_mappings=[slots + [-1, -1]], global_num_tokens_after_padding=6,
                              local_token_indices=indices, local_to_global_req_indices=[0, 1],
                              padded_gather_idx=indices, gathered_kv_write_mask=[True, False])
                values["local"].update(num_tokens=2, query_start_loc=[0, 1, 2], positions=positions,
                                       seq_lens=lengths, dcp_local_seq_lens=[2, 1])
                values["local"]["groups"][0]["slot_mapping"] = [5, -1]
                layout = {**self.parallel_layout, "cp_size": 2, "cp_rank": rank, "dcp_size": 2}
                self.guard.parallel(values, self.plan, layout)
                self.assertEqual(self.reasons(), [])
                values["local"]["groups"][0]["slot_mapping"][0] = 9
                self.guard.parallel(values, self.plan, layout)
                self.assertIn("pcp_write_frame_mismatch", self.reasons())
                values["global_slot_mappings"][0][-1] = 5
                self.guard.parallel(values, self.plan, layout)
                self.assertIn("padded_write_target", self.reasons())

    def test_slot_finished_preempted_resume_bindings_keep_frozen_plans(self):
        bindings = {}
        self.guard.slot = True
        empty = SimpleNamespace(req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set())
        first = scheduler_output(scheduled_new_reqs=[SimpleNamespace(req_id="a", num_computed_tokens=0,
                                                                     block_ids=[[1]])],
                                 scheduled_cached_reqs=empty)
        bind(first, bindings, self.guard)
        frozen = first._msprobe_kv_plan
        resumed = scheduler_output(preempted_req_ids=["a"], scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"], num_computed_tokens=[0], new_block_ids=[[[3]]], resumed_req_ids={"a"}))
        bind(resumed, bindings, self.guard)
        self.assertEqual(frozen, (("a", 0, 2, ((1,),)),))
        self.assertEqual(bindings, {"a": ((3,),)})
        values = dict(req_ids=["a"], query_start_loc=[0, 2], positions=[0, 1], seq_lens=[2], num_tokens=2,
                      num_tokens_after_padding=2, padded_slot_mappings=[[]], trace_id="resumed",
                      groups=[dict(block_table=[[3]], slot_mapping=[12, 13])])
        self.guard.audit(values, resumed._msprobe_kv_plan, self.layout)
        self.assertEqual(self.reasons(), [])
        values["groups"][0]["slot_mapping"] = [4, 5]
        self.guard.audit(values, resumed._msprobe_kv_plan, self.layout)
        self.assertEqual(self.reasons(), ["write_target_mismatch"] * 2)
        finished = scheduler_output(num_scheduled_tokens={}, total_num_scheduled_tokens=0,
                                    finished_req_ids=["a"], scheduled_cached_reqs=empty)
        bind(finished, bindings, self.guard)
        self.assertEqual(bindings, {})
        self.assertEqual(frozen, (("a", 0, 2, ((1,),)),))

    def test_spec_actual_row_constraints(self):
        cases = [
            ("num_sampled", [1], "sample_count_shape_mismatch"),
            ("query_start_loc", [0, 2], "sample_query_range_mismatch"),
            ("query_start_loc", [1, 2, 4, 6], "sample_query_range_mismatch"),
            ("query_start_loc", [0, 3, 2, 6], "sample_query_range_mismatch"),
            ("num_sampled", [-1, 1, 1], "sample_count_invalid"),
            ("num_sampled", [3, 1, 1], "sample_count_invalid"),
            ("num_rejected", [-1, 1, 1], "sample_count_invalid"),
            ("num_rejected", [3, 1, 1], "rejected_count_outside_query"),
            ("idx_mapping", [7, 1, 2], "sample_request_mismatch"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field, value=value):
                self.mock_emit.reset_mock()
                values = dict(idx_mapping=[0, 1, 2], num_sampled=[1, 1, 1], num_rejected=[1, 1, 1],
                              query_start_loc=[0, 2, 4, 6, 8], sample_capacity=2, trace_id="step")
                values[field] = value
                self.guard.spec(values, {i: (str(i), 2) for i in range(3)})
                self.assertEqual(self.reasons(), [reason])
                self.assertEqual(self.mock_emit.call_args.kwargs["trace_id"], "step")

    def test_spec_full_padding_masked_rows_and_deferred_pp(self):
        values = dict(idx_mapping=[0, 1, 2], num_sampled=[1, 1, 1], num_rejected=[1, 1, 1],
                      query_start_loc=[0, 2, 4, 6, 8], sample_capacity=2)
        original = deepcopy(values)
        self.guard.spec(values, {i: (str(i), 2) for i in range(3)})
        self.assertEqual(values, original)
        self.mock_emit.assert_not_called()
        values.update(idx_mapping=[-1], num_sampled=[-999], num_rejected=[999], query_start_loc=None)
        self.guard.spec(values)
        self.mock_emit.assert_not_called()
        values.update(idx_mapping=[0], num_sampled=[1], num_rejected=[0], query_start_loc=[0, 2])
        self.guard.spec(values, {0: ("a", 1)})
        self.assertEqual(self.reasons(), ["sample_query_exceeds_budget"])

    def test_audit_request_ranges(self):
        cases = [
            ("req_ids", ["a", "a"], "batch_request_mismatch"),
            ("req_ids", ["a", "unknown"], "batch_request_mismatch"),
            ("query_start_loc", [0, 4, 3], "batch_token_range_mismatch"),
            ("groups", [], "cache_group_count_mismatch"),
            ("dcp_local_seq_lens", [0, 0], "dcp_read_length_mismatch"),
            ("positions", [2, 4, 0, 1], "execution_range_mismatch"),
            ("seq_lens", [5, 2], "read_length_mismatch"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field, value=value):
                self.mock_emit.reset_mock()
                values = {**self.values, field: value}
                self.guard.audit(values, self.plan, self.layout)
                self.assertIn(reason, self.reasons())

    def test_audit_read_and_cross_request_write_routing(self):
        cases = [
            ([[1]], [6, 7, 8, 9], "cache_plan_shape_mismatch"),
            ([[3], [2]], [6, 7, 8, 9], "read_block_mapping_mismatch"),
            ([[0], [2]], [6, 7, 8, 9], "write_block_mapping_mismatch"),
            ([[1], [2]], [8, 9, 6, 7], "write_target_mismatch"),
        ]
        for tables, slots, reason in cases:
            with self.subTest(reason=reason):
                self.mock_emit.reset_mock()
                self.values["groups"] = [dict(block_table=tables, slot_mapping=slots)]
                self.guard.audit(self.values, self.plan, self.layout)
                self.assertIn(reason, self.reasons())

    def test_audit_cache_capacity_and_null_block(self):
        for block, reason in [(0, "null_block_write"), (8, "write_outside_cache"), (-1, "write_outside_cache")]:
            with self.subTest(block=block):
                self.mock_emit.reset_mock()
                plan = (("a", 2, 2, ((block,),)), self.plan[1])
                self.values["groups"][0].update(block_table=[[block], [2]],
                                                slot_mapping=[block * 4 + 2, block * 4 + 3, 8, 9])
                self.guard.audit(self.values, plan, self.layout)
                self.assertEqual(self.reasons(), [reason])

    def test_audit_native_null_exception_dcp_and_disabled_slots(self):
        self.guard.audit(self.values, self.plan, self.layout)
        self.layout.update(cp_size=2)
        self.values["groups"][0]["slot_mapping"] = [5, -1, 8, -1]
        self.values["dcp_local_seq_lens"] = [2, 1]
        self.guard.audit(self.values, self.plan, self.layout)
        self.layout["groups"][0]["slot_mapping_enabled"] = False
        self.values["groups"][0]["slot_mapping"] = [-1] * 4
        self.guard.audit(self.values, self.plan, self.layout)
        self.layout.update(cp_size=1, allow_null_block=True)
        self.layout["groups"][0]["slot_mapping_enabled"] = True
        self.values.pop("dcp_local_seq_lens")
        self.values["groups"][0].update(block_table=[[0], [2]], slot_mapping=[2, 3, 8, 9])
        self.guard.audit(self.values, (("a", 2, 2, ((0,),)), self.plan[1]), self.layout)
        self.mock_emit.assert_not_called()

    def test_audit_kernel_block_expansion_and_retired_window_mask(self):
        self.layout["groups"][0]["kernel_block_size"] = 2
        self.values["groups"][0]["block_table"] = [[0, 3], [4, 5]]
        self.guard.audit(self.values, self.plan, self.layout)
        self.mock_emit.assert_not_called()

    def test_audit_dynamic_padding_and_scheduler_group_errors(self):
        self.values.update(req_ids=["a"], query_start_loc=[0, 2], seq_lens=[4])
        self.values["groups"][0].update(block_table=[[1]], slot_mapping=[6, 7, -1, -1])
        self.layout["dynamic"] = True
        self.guard.audit(self.values, self.plan, self.layout)
        self.mock_emit.assert_not_called()
        self.values["groups"][0]["slot_mapping"][-1] = 1
        self.guard.audit(self.values, (("a", 2, 2, ()),), self.layout)
        self.assertEqual(self.reasons(), ["padded_write_target", "scheduler_group_missing"])
    def test_pcp_visible_mapping_constraints(self):
        cases = [
            ("global", "local_token_indices", [0, 1, 2, 4], "pcp_local_range_mismatch"),
            ("global", "local_to_global_req_indices", [1, 0], "pcp_request_mapping_mismatch"),
            ("local", "positions", [1, 3, 0, 1], "pcp_token_mapping_mismatch"),
            ("local", "seq_lens", [5, 2], "pcp_read_length_mismatch"),
            ("local", "dcp_local_seq_lens", [0, 0], "dcp_read_length_mismatch"),
            ("global", "gathered_kv_write_mask", [], "pcp_write_mask_shape_mismatch"),
            ("local", "groups", [], "pcp_cache_group_count_mismatch"),
            ("global", "padded_gather_idx", [99, 1, 2, 3], "pcp_gather_index_mismatch"),
            ("local", "groups", [dict(block_table=[[3], [2]], slot_mapping=[6, 7, 8, 9])],
             "pcp_read_block_mapping_mismatch"),
            ("local", "groups", [dict(block_table=[[1], [2]], slot_mapping=[8, 9, 6, 7])],
             "pcp_write_frame_mismatch"),
        ]
        for target, field, value, reason in cases:
            with self.subTest(field=field, value=value):
                self.mock_emit.reset_mock()
                values = deepcopy(self.parallel)
                (values if target == "global" else values["local"])[field] = value
                self.guard.parallel(values, self.plan, self.parallel_layout)
                self.assertEqual(self.reasons(), [reason])

    def test_pcp_gather_mask_and_missing_native_context(self):
        self.guard.audit(self.values, self.plan, self.parallel_layout)
        self.guard.parallel(self.parallel, self.plan, self.parallel_layout)
        self.parallel["gathered_kv_write_mask"][1] = False
        self.values["groups"][0]["slot_mapping"][1] = -1
        self.guard.parallel(self.parallel, self.plan, self.parallel_layout)
        for field in ("local", "local_token_indices", "padded_gather_idx"):
            with self.subTest(field=field):
                self.guard.parallel({**self.parallel, field: None}, self.plan, self.parallel_layout)
        self.mock_emit.assert_not_called()

    def test_bind_new_incremental_resume_preempt_finished_and_shared_prefix(self):
        bindings = {"finished": ((7,),), "preempted": ((6,),)}
        first = scheduler_output(finished_req_ids=["finished"], preempted_req_ids=["preempted"],
                                 scheduled_new_reqs=[SimpleNamespace(req_id="a", num_computed_tokens=0, block_ids=[[1]])])
        bind(first, bindings, self.guard)
        saved = first._msprobe_kv_plan
        second = scheduler_output(scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"], num_computed_tokens=[2], new_block_ids=[[[2]]], resumed_req_ids=[]))
        bind(second, bindings, self.guard)
        self.assertEqual(bindings, {"a": ((1, 2),)})
        self.assertEqual(saved, (("a", 2, 2, ((1,),)),))
        resumed = scheduler_output(scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"], num_computed_tokens=[2], new_block_ids=[[[3]]], resumed_req_ids=["a"]))
        bind(resumed, bindings, self.guard)
        self.assertEqual(bindings, {"a": ((3,),)})
        shared = scheduler_output(scheduled_new_reqs=[SimpleNamespace(
            req_id="b", num_computed_tokens=2, block_ids=[[3]])])
        bind(shared, bindings, self.guard)
        self.assertEqual(bindings["a"], bindings["b"])
        self.mock_emit.assert_not_called()

    def test_bind_rejects_inconsistent_public_deltas(self):
        cases = [
            ({"total_num_scheduled_tokens": 3}, {"a": ((1,),)}, "scheduled_token_total_mismatch"),
            ({"scheduled_cached_reqs": SimpleNamespace(req_ids=[], num_computed_tokens=[2],
                                                       new_block_ids=[], resumed_req_ids=[])},
             {"a": ((1,),)}, "cached_request_array_length_mismatch"),
            ({"num_scheduled_tokens": {"a": 0}, "total_num_scheduled_tokens": 0}, {"a": ((1,),)},
             "scheduled_range_invalid"),
            ({}, {}, "scheduler_binding_missing"),
        ]
        for change, bindings, reason in cases:
            with self.subTest(reason=reason):
                self.mock_emit.reset_mock()
                bind(scheduler_output(**change), bindings, self.guard)
                self.assertIn(reason, self.reasons())
                bind(scheduler_output(**change), dict(bindings))

    def test_bind_cached_first_seen_and_mismatched_group_count(self):
        bindings = {}
        cached = scheduler_output(scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"], num_computed_tokens=[2], new_block_ids=[[[1]]], resumed_req_ids=[]))
        bind(cached, bindings, self.guard)
        self.assertEqual(bindings, {"a": ((1,),)})
        self.mock_emit.assert_not_called()
        cached.scheduled_cached_reqs.new_block_ids = [[[2], [3]]]
        bind(cached, bindings, self.guard)
        self.assertEqual(bindings, {"a": ((1,),)})
        self.assertEqual(self.reasons(), ["scheduler_group_count_mismatch"])


def scheduler_output(**kwargs):
    fields = dict(num_scheduled_tokens={"a": 2}, total_num_scheduled_tokens=2, finished_req_ids=[],
                  preempted_req_ids=[], scheduled_new_reqs=[], _msprobe_kv_trace_id="step",
                  scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[2],
                                                       new_block_ids=[None], resumed_req_ids=[]))
    return SimpleNamespace(**{**fields, **kwargs})
