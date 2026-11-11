import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch

import torch


def _load_model_runner_module():
    module_names = ("msprobe", "msprobe.vllm_kv_guard", "msprobe.vllm_kv_guard.events")
    fake_modules = {name: ModuleType(name) for name in module_names}
    fake_modules[module_names[-1]].emit = Mock()
    module_path = Path(__file__).resolve().parents[3] / "python/msprobe/vllm_kv_guard/model_runner.py"
    spec = importlib.util.spec_from_file_location("msprobe_model_runner_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


model_runner = _load_model_runner_module()


def guard():
    return Mock(spec=["audit", "spec", "parallel", "fail", "check"])


def layout():
    return dict(
        groups=[dict(block_size=2, kernel_block_size=2, slot_mapping_enabled=True)],
        pcp_size=1,
        pcp_rank=0,
        dcp_size=1,
        dcp_rank=0,
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
        dynamic=False,
        num_blocks=8,
        dp_rank=0,
        allow_null_block=False,
    )


def batch(**changes):
    values = dict(
        req_ids=["a"],
        num_reqs=1,
        num_tokens=2,
        num_tokens_after_padding=2,
        query_start_loc=torch.tensor([0, 2, 999]),
        positions=torch.tensor([0, 1, 999]),
        seq_lens=torch.tensor([2, 999]),
        dcp_local_seq_lens=None,
        is_dummy=False,
        is_cpp_profile=False,
    )
    return NS(**(values | changes))


def runner(pcp=None):
    return NS(
        initialize_kv_cache=Mock(return_value="initialized"),
        postprocess_sampled=Mock(return_value="sampled"),
        _dummy_run=Mock(return_value="dummy"),
        model_state=NS(prepare_attn=Mock(return_value="attention")),
        block_tables=NS(
            block_sizes=[2],
            kernel_block_sizes=[2],
            slot_mapping_enabled=[True],
            cp_size=1,
            cp_rank=0,
            cp_interleave=1,
            redirect_writes_to_null_block=False,
            slot_mappings=torch.tensor([[-1, -1]]),
        ),
        pcp_manager=pcp,
        dcp_size=1,
        dcp_rank=0,
        kv_cache_config=NS(num_blocks=8),
        vllm_config=NS(parallel_config=NS(data_parallel_rank=0)),
        is_last_pp_rank=True,
        speculator=None,
        sync_spec_pp_cpu_counts=False,
        num_speculative_steps=0,
        req_states=NS(req_id_to_index={"a": 0, "b": 1, "c": 2}),
    )


def recorder():
    result = model_runner.PlanRecorder(0, guard())
    result.trace_id, result.plan = ("step", (("a", 0, 2, ((1,),)),))
    return result


def parallel_inputs():
    global_batch = batch(positions=torch.tensor([0, 1]), query_start_loc=torch.tensor([0, 2]))
    local_batch = batch(
        num_tokens=1,
        positions=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1]),
        dcp_local_seq_lens=torch.tensor([1]),
    )
    tables, slots = ([torch.tensor([[1]])], torch.tensor([[2]]))
    context = NS(
        global_batch=global_batch,
        global_block_tables=tables,
        global_slot_mappings=torch.tensor([[2, -1]]),
        local_to_global_req_indices=torch.tensor([0]),
        local_token_indices=torch.tensor([0]),
        padded_gather_idx=torch.tensor([0]),
        gathered_kv_write_mask=torch.tensor([True]),
        global_block_table_num_blocks=torch.tensor([[1]]),
    )
    return (context, local_batch, tables, slots)


class TestModelRunner(TestCase):
    def setUp(self):
        self.event, self.stream, self.records = (Mock(), object(), [])
        npu = NS(Event=Mock(return_value=self.event), current_stream=Mock(return_value=self.stream))
        npu_patcher = patch.object(torch, "npu", npu, create=True)
        npu_patcher.start()
        self.addCleanup(npu_patcher.stop)
        emit_patcher = patch.object(
            model_runner, "emit", side_effect=lambda name, **fields: self.records.append((name, fields))
        )
        emit_patcher.start()
        self.addCleanup(emit_patcher.stop)

    def test_snapshot_freezes_values_flushes_previous_and_finishes_once(self):
        current, output = (recorder(), [])
        tensor, names = (torch.tensor([1, 2]), ["a"])
        current.snapshot(
            {"values": tensor, "names": names, "nested": {"values": tensor}},
            lambda trace, values: output.append((trace, values)),
        )
        tensor.fill_(9)
        names.append("b")
        self.assertEqual(output, [])
        current.snapshot({"values": torch.tensor([3])}, lambda trace, values: output.append((trace, values)))
        self.assertEqual(output, [("step", {"values": [1, 2], "names": ["a"], "nested": {"values": [1, 2]}})])
        current.pending()
        current.pending()
        self.assertEqual(output[-1], ("step", {"values": [3]}))
        self.assertEqual(len(output), self.event.synchronize.call_count)
        self.assertEqual(self.event.synchronize.call_count, 2)
        self.event.record.assert_called_with(self.stream)

    def test_snapshot_caches_callback_failure_and_synchronizes_partial_copy(self):
        current, failure = (recorder(), RuntimeError("callback"))
        callback = Mock(side_effect=failure)
        current.snapshot(torch.tensor([1]), callback)
        for _ in range(2):
            with self.assertRaises(RuntimeError) as raised:
                current.pending()
            self.assertIs(raised.exception, failure)
        callback.assert_called_once()
        self.assertEqual(self.event.synchronize.call_count, 1)
        with patch.object(torch.Tensor, "clone", side_effect=MemoryError("host copy")):
            with self.assertRaisesRegex(MemoryError, "host copy"):
                recorder().snapshot(torch.tensor([1]), Mock())
        self.assertEqual(self.event.synchronize.call_count, 2)

    def test_device_copy_reuses_matching_host_buffers_and_reallocates_shape(self):

        class DeviceTensor(torch.Tensor):
            @property
            def device(self):
                return NS(type="npu")

        def allocate(value, **kwargs):
            self.assertEqual(kwargs, dict(device="cpu", pin_memory=True))
            return torch.empty(value.shape, dtype=value.dtype)

        allocation = Mock(side_effect=allocate)
        with patch.object(torch, "empty_like", allocation):
            current, output = (recorder(), [])
            for values in ([1, 2], [3, 4], [5]):
                current.snapshot(
                    torch.tensor(values).as_subclass(DeviceTensor), lambda trace, data: output.append(data)
                )
                current.pending()
            self.assertEqual(output, [[1, 2], [3, 4], [5]])
            self.assertEqual(allocation.call_count, 2)
            self.assertEqual(len(current.buffers), 1)

    def test_capture_preserves_address_plan_before_native_buffers_change(self):
        records = self.records
        current, input_batch = (recorder(), batch())
        tables, slots = ([torch.tensor([[1], [7]])], torch.tensor([[2, 3, 999]]))
        current.capture(input_batch, tables, slots, layout(), "FULL")
        input_batch.positions.fill_(99)
        tables[0].fill_(7)
        slots.fill_(7)
        current.pending()
        current.guard.audit.assert_called_once()
        values, plan, spec = current.guard.audit.call_args.args
        self.assertEqual(plan, current.plan)
        self.assertEqual(spec, layout())
        self.assertEqual(values["trace_id"], "step")
        self.assertEqual(values["positions"], [0, 1])
        self.assertEqual(values["groups"], [{"block_table": [[1]], "slot_mapping": [2, 3]}])
        self.assertEqual([name for name, _ in records], ["kv.write", "kv.read"])
        write, read = (fields for _, fields in records)
        self.assertEqual(write["positions"], [0, 1])
        self.assertEqual(write["slot_mapping"], [2, 3])
        self.assertEqual(write["query_start_loc"], [0, 2])
        self.assertEqual(write["execution_mode"], "FULL")
        self.assertEqual(read["block_table"], [[1]])
        self.assertEqual(read["seq_lens"], [2])

    def test_dummy_only_writes_padding_or_explicit_native_null_slots(self):
        for allow, slots, persistent, invalid in [
            (False, [-1, -1], [-1], False),
            (True, [0, 1], [-1], False),
            (False, [0], [-1], True),
            (True, [2], [-1], True),
            (False, [-1], [2], True),
        ]:
            with self.subTest(allow=allow, slots=slots, persistent=persistent, invalid=invalid):
                self.records.clear()
                current = recorder()
                spec = layout()
                spec["allow_null_block"] = allow
                current.capture_dummy(torch.tensor([slots]), torch.tensor([persistent]), spec, "FULL")
                current.pending()
                if invalid:
                    scope, values = (
                        ("input", slots) if any((slot != -1 for slot in slots)) else ("persistent", persistent)
                    )
                    index = next((index for index, slot in enumerate(values) if slot != -1))
                    current.guard.fail.assert_called_once_with(
                        "dummy_write_target",
                        trace_id="step",
                        dp_rank=0,
                        group_id=0,
                        token_index=index,
                        actual_slot=values[index],
                        slot_mapping_scope=scope,
                        boundary="model_state.prepare_attn.input",
                    )
                else:
                    current.guard.fail.assert_not_called()
                self.assertEqual([fields["slot_mapping_scope"] for _, fields in self.records], ["input", "persistent"])

    def test_parallel_capture_audits_global_and_local_maps_then_logs_global_frame(self):
        current = recorder()
        audits = []
        current.guard.parallel.side_effect = lambda *args: audits.append(deepcopy(args))
        context, local_batch, tables, slots = parallel_inputs()
        spec = layout()
        spec.update(pcp_size=2, cp_size=2, dcp_size=2)
        current.capture_parallel(context, spec, local_batch, tables, slots)
        context.local_token_indices.fill_(99)
        current.pending()
        values, plan, checked_layout = audits[0]
        self.assertEqual(plan, current.plan)
        self.assertEqual(checked_layout, spec)
        self.assertEqual(values["trace_id"], "step")
        self.assertEqual(values["local_token_indices"], [0])
        self.assertEqual(values["local"]["dcp_local_seq_lens"], [1])
        self.assertEqual(values["local"]["groups"], [{"block_table": [[1]], "slot_mapping": [2]}])
        fields = self.records[-1][1]
        self.assertEqual(fields["local_token_indices"], [0])
        self.assertEqual(fields["global_slot_mappings"], [[2, -1]])
        self.assertNotIn("local", fields)
        self.assertNotIn("layout", fields)
        current.capture_parallel(parallel_inputs()[0], spec, None, None, None)
        current.pending()
        self.assertEqual(len(audits), 2)
        self.assertIsNone(audits[1][0]["local"])

    def test_install_forwards_initialization_and_prepare_arguments(self):
        native = runner()
        initialize, prepare = (native.initialize_kv_cache, native.model_state.prepare_attn)
        current = model_runner.install(native, 0, guard())
        self.assertEqual(native.initialize_kv_cache("cache", mode=1), "initialized")
        initialize.assert_called_once_with("cache", mode=1)
        current.trace_id, current.plan = ("step", (("a", 0, 2, ((1,),)),))
        args = (batch(), NS(name="FULL"), [torch.tensor([[1]])], torch.tensor([[2, 3]]), "groups", "config", False)
        self.assertEqual(native.model_state.prepare_attn(*args, "extra", flag=True), "attention")
        self.assertEqual(prepare.call_args.args, (*args, "extra"))
        self.assertEqual(prepare.call_args.kwargs, {"flag": True})
        current.pending()
        for changes, for_capture in (({}, True), ({"is_cpp_profile": True}, False)):
            native.model_state.prepare_attn(batch(**changes), *args[1:6], for_capture)
        self.assertEqual(len([name for name, _ in self.records if name == "kv.write"]), 1)

    def test_postprocess_records_full_padding_and_preserves_native_arguments(self):
        for last_rank in [True, False]:
            with self.subTest(last_rank=last_rank):
                native = runner()
                native.is_last_pp_rank, native.num_speculative_steps = (last_rank, 1)
                delegate = native.postprocess_sampled
                current = model_runner.install(native, 0, guard())
                current.trace_id = "step"
                current.plan = tuple(((req, 0, 2, ((index + 1,),)) for index, req in enumerate(("a", "b", "c"))))
                indices, sampled, rejected = (torch.tensor([0, 1, 2]), torch.tensor([1, 1, 1]), torch.tensor([0, 0, 0]))
                starts = torch.tensor([0, 2, 4, 6, 8]) if last_rank else None
                args = (indices, torch.zeros(3, 2), sampled, rejected, starts, "draft")
                delegate.side_effect = lambda *args: native.req_states.req_id_to_index.clear() or "sampled"
                self.assertEqual(native.postprocess_sampled(*args), "sampled")
                self.assertEqual(delegate.call_args.args, args)
                rejected.fill_(99)
                current.pending()
                fields = self.records[-1][1]
                self.assertEqual(fields["num_rejected"], [0, 0, 0])
                self.assertEqual(fields["query_start_loc"], [0, 2, 4, 6, 8] if last_rank else None)
                self.assertEqual(fields["trace_id"], "step" if last_rank else None)
                self.assertEqual(fields["association"], "current_step" if last_rank else "observed_call_only")
                values, rows = current.guard.spec.call_args.args
                self.assertEqual(values["num_rejected"], [0, 0, 0])
                self.assertEqual(values["query_start_loc"], [0, 2, 4, 6, 8] if last_rank else None)
                self.assertEqual(rows, {0: ("a", 2), 1: ("b", 2), 2: ("c", 2)} if last_rank else None)

    def test_postprocess_observes_only_active_speculative_calls(self):
        for traced, speculator, sync, steps in [
            (False, True, False, 0),
            (True, False, False, 0),
            (True, True, False, 0),
            (True, False, True, 0),
        ]:
            with self.subTest(traced=traced, speculator=speculator, sync=sync, steps=steps):
                self.records.clear()
                native = runner()
                native.speculator = object() if speculator else None
                native.sync_spec_pp_cpu_counts, native.num_speculative_steps = (sync, steps)
                current = model_runner.install(native, 0, guard())
                current.trace_id, current.plan = ("step" if traced else None, (("a", 0, 1, ((1,),)),))
                native.postprocess_sampled(
                    torch.tensor([0]), torch.tensor([[9]]), torch.tensor([1]), torch.tensor([0]), torch.tensor([0, 1])
                )
                current.pending()
                self.assertEqual(bool(self.records), traced and (speculator or sync or steps > 0))
                self.assertEqual(current.guard.spec.call_count, int(traced and (speculator or sync or steps > 0)))

    def test_postprocess_forwards_active_counts_and_propagates_guard_callback_failure(self):
        native = runner()
        native.num_speculative_steps = 1
        current = model_runner.install(native, 0, guard())
        current.trace_id, current.plan = ("step", (("a", 0, 2, ((1,),)),))
        native.postprocess_sampled(
            torch.tensor([0]), torch.tensor([[9, 8]]), torch.tensor([1]), torch.tensor([3]), torch.tensor([0, 2, 4])
        )
        failure = RuntimeError("guard callback")
        current.guard.spec.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            current.pending()
        self.assertIs(raised.exception, failure)
        values, rows = current.guard.spec.call_args.args
        self.assertEqual(values["num_rejected"], [3])
        self.assertEqual(values["query_start_loc"], [0, 2, 4])
        self.assertEqual(rows, {0: ("a", 2)})
        self.assertEqual(self.records, [])

    def test_dummy_flushes_guard_before_return_and_restores_trace_on_failure(self):
        native = runner()
        delegate = native._dummy_run
        current = model_runner.install(native, 0, guard())
        native.initialize_kv_cache()
        current.trace_id = "live-step"

        def dummy_body(*args, **kwargs):
            native.model_state.prepare_attn(
                batch(is_dummy=True), NS(name="FULL"), [torch.tensor([[1]])], torch.tensor([[-1, -1]]), None, None, True
            )
            return "dummy"

        delegate.side_effect = dummy_body
        self.assertEqual(native._dummy_run("arg", flag=True), "dummy")
        delegate.assert_called_once_with("arg", flag=True)
        current.guard.check.assert_called_once()
        self.assertEqual(current.trace_id, "live-step")
        native.block_tables.slot_mappings.fill_(2)
        current.guard.check.side_effect = RuntimeError("guard check")
        with self.assertRaisesRegex(RuntimeError, "guard check"):
            native._dummy_run()
        self.assertEqual(current.guard.fail.call_args.args, ("dummy_write_target",))
        self.assertEqual(current.guard.fail.call_args.kwargs["slot_mapping_scope"], "persistent")
        self.assertEqual(current.trace_id, "live-step")
        self.assertEqual(
            len([fields for _, fields in self.records if fields.get("boundary") == "runner._dummy_run.return"]), 2
        )

    def test_native_exception_identity_is_preserved(self):
        for boundary in ["initialize", "prepare", "sample", "dummy"]:
            with self.subTest(boundary=boundary):
                native, failure = (runner(), RuntimeError("native failure"))
                delegates = dict(
                    initialize=native.initialize_kv_cache,
                    prepare=native.model_state.prepare_attn,
                    sample=native.postprocess_sampled,
                    dummy=native._dummy_run,
                )
                current = model_runner.install(native, 0, guard())
                if boundary == "prepare":
                    native.initialize_kv_cache()
                current.trace_id = "step"
                delegates[boundary].side_effect = failure
                calls = dict(
                    initialize=lambda: native.initialize_kv_cache(),
                    prepare=lambda: native.model_state.prepare_attn(
                        batch(), NS(name="FULL"), [], torch.tensor([]), None, None, True
                    ),
                    sample=lambda: native.postprocess_sampled(
                        torch.tensor([]), torch.tensor([[]]), torch.tensor([]), torch.tensor([])
                    ),
                    dummy=lambda: native._dummy_run(),
                )
                with self.assertRaises(RuntimeError) as raised:
                    calls[boundary]()
                self.assertIs(raised.exception, failure)
                self.assertEqual(current.trace_id, "step")

    def test_pcp_install_captures_live_context_skips_profiles_and_uses_persistent_dummy_slots(self):
        context, local_batch, tables, slots = parallel_inputs()
        pcp = NS(
            pcp_world_size=2,
            pcp_rank=0,
            build_attention_context=Mock(return_value=context),
            _gathered_kv_slot_mappings=torch.tensor([[-1, -1]]),
        )
        native = runner(pcp)
        native.block_tables.cp_size = 2
        current = model_runner.install(native, 0, guard())
        native.initialize_kv_cache()
        current.trace_id, current.plan = ("step", (("a", 0, 2, ((1,),)),))
        self.assertIs(pcp.build_attention_context(local_batch, tables, slots), context)
        current.pending()
        self.assertEqual(self.records[-1][1]["evidence"], "parallel_context")
        for changes in ({"is_dummy": True}, {"is_cpp_profile": True}):
            context.global_batch = batch(**changes)
            pcp.build_attention_context()
        current.trace_id = None
        context.global_batch = batch()
        pcp.build_attention_context()
        native.model_state.prepare_attn(batch(), NS(name="FULL"), tables, slots, None, None)
        self.assertEqual(len(self.records), 1)
        native.block_tables.slot_mappings.fill_(9)
        native.model_state.prepare_attn(batch(is_dummy=True), NS(name="FULL"), tables, torch.tensor([[-1]]), None, None)
        current.pending()
        self.assertEqual(self.records[-1][1]["non_pad_slots"], [])
