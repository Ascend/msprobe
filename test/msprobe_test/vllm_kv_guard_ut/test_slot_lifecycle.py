from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch
import sys

from msprobe.vllm_kv_guard import checker, config, scheduler, transfer
from msprobe.vllm_kv_guard.checker import Guard, KVGuardError


class TestSlotLifecycle(TestCase):
    def fixture(self, action="raise", pd=False, retained=False):
        owned = {"a": ([1],)}
        manager = NS(get_block_ids=Mock(side_effect=lambda req: owned.get(req, ([],))))
        manager.free = Mock(side_effect=None if retained else lambda req, **kw: owned.pop(req.request_id, None))
        native = manager.free
        params = {"remote_request_id": "a", "remote_block_ids": ([1],)}
        connector = NS(request_finished=Mock(return_value=(True, params)),
                       _get_transfer_block_ids=Mock(side_effect=lambda ids, count: ids))
        host = NS(kv_cache_manager=manager, connector=NS(connector_scheduler=connector))
        guard, pending = Guard(action, slot=True), set()
        transfer.lifecycle(host, guard, {"pd_slot": pd}, pending)
        return host, guard, pending, native, params, owned

    def test_release_and_inflight_release_follow_action(self):
        for action in ("log", "raise"):
            for retained, inflight, reason in ((False, False, None), (True, False, "slot_release_binding_stale"),
                                               (False, True, "slot_release_inflight")):
                with self.subTest(action=action, reason=reason):
                    host, guard, pending, native, _, _ = self.fixture(action, retained=retained)
                    if inflight:
                        pending.add("a")
                    with self.assertRaisesRegex(KVGuardError, reason) if reason and action == "raise" else nullcontext():
                        host.kv_cache_manager.free(NS(request_id="a"))
                    self.assertEqual(guard.failure, reason)
                    self.assertEqual(native.call_count, int(not (inflight and action == "raise")))

    def test_optional_native_connector_result_preserves_update_output(self):
        host, _, _, _, _, _ = self.fixture()
        value = {0: NS(outputs=[])}
        host.schedule, host.update_from_output = Mock(), Mock(return_value=value)
        host.vllm_config = NS(speculative_config=None)
        scheduler._observe(host, {"slot": True, "pd_slot": True, "action": "raise"})
        output = NS(_msprobe_kv_trace_id="trace", num_scheduled_tokens={"a": 1})
        result = NS(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[1]], kv_connector_output=None)
        self.assertIs(host.update_from_output(output, result), value)

    def test_empty_receive_can_release_without_an_inflight_slot(self):
        host, guard, pending, native, _, owned = self.fixture()
        owned["a"] = ([],)
        pending.add("a")
        host.kv_cache_manager.free(NS(request_id="a"))
        self.assertIsNone(guard.failure)
        native.assert_called_once()

    def test_producer_export_and_native_tail_clipping(self):
        for action in ("log", "raise"):
            for bad in (False, True):
                host, guard, _, _, params, owned = self.fixture(action, pd=True)
                owned["a"] = ([1, 2],)
                connector = host.connector.connector_scheduler
                connector._get_transfer_block_ids.side_effect = None
                connector._get_transfer_block_ids.return_value = ([1],)
                if bad:
                    params["remote_block_ids"] = ([2],)
                with self.assertRaisesRegex(KVGuardError, "pd_slot_source_plan_mismatch") if bad and action == "raise" else nullcontext():
                    connector.request_finished(NS(request_id="a", num_prompt_tokens=1), owned["a"])
                self.assertEqual(guard.failure, "pd_slot_source_plan_mismatch" if bad else None)
                connector._get_transfer_block_ids.assert_called_once_with(owned["a"], 1)


class TestMooncakeBatchSlots(TestCase):
    def fixture(self, action="raise"):
        guard = Guard(action, slot=True)
        guard.plans["d"] = ((3, 4), (8, 9))
        backend = Mock(return_value=0)
        class Engine:
            def batch_transfer_sync_read(self, *args):
                return backend(*args)
        class Connector:
            kv_group2layeridx = {0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])}
            def register_kv_caches(self, *args, **kwargs):
                pass
        class Receiver:
            def __init__(self):
                self.engine = Engine()
                self.local_engine_id, self.local_handshake_port = "local", 1
                self.kv_caches_base_addr = {"local": {1: [[1000]]}, "remote": {2: [[5000]]}}
                self.remote_block_stride_per_addr = {"remote": {2: [[10]]}}
                self.remote_kv_group2layeridx, self.remote_metadata_lock = {}, nullcontext()
                self.kv_group2layeridx = Connector.kv_group2layeridx
                self.block_len_per_addr = self.block_stride_per_addr = [[10]]
            def _transfer_kv_cache_all_groups(self, request):
                if request.get("error"):
                    raise request["error"]
                if request.get("discover"):
                    self.kv_caches_base_addr["remote"] = {2: [[5000]]}
                for rows in request["batches"]:
                    self.engine.batch_transfer_sync_read("session", *rows)
                return "done"
        self.connector_class = Connector
        self.names, self.resolve = Mock(return_value={}), Mock(side_effect=lambda layer, *args: layer)
        module = NS(KVCacheRecvingThread=Receiver, MooncakeConnectorWorker=Connector, TransferEngine=Engine,
                    build_layer_name_to_metadata_idx=self.names, resolve_remote_layer_idx=self.resolve)
        parent = "vllm_ascend.distributed.kv_transfer.kv_p2p"
        with patch.dict(sys.modules, {parent: NS(mooncake_connector=module), parent + ".mooncake_connector": module}):
            transfer.install(guard, True)
            Connector().register_kv_caches({})
        request = dict(request_id="d", remote_request_id="p", remote_engine_id="remote", remote_handshake_port=2,
                       local_block_ids=([3, 4],), remote_block_ids=([8, 9],))
        return Receiver(), guard, backend, request

    def test_cache_layout_is_rejected_during_native_registration_for_both_actions(self):
        for action in ("log", "raise"):
            self.fixture(action)
            self.connector_class.kv_group2layeridx = {0: ({"kv_cache_spec_type": "MambaSpec"}, [0])}
            with self.assertRaisesRegex(ValueError, "single full-attention"):
                self.connector_class().register_kv_caches({})

    def test_whole_range_comparison_accepts_merge_split_reordering_and_byte_fragments(self):
        batches = [([([1030], [5080], [20])]),
                   [([1040], [5090], [10]), ([1030], [5080], [10])],
                   [([1030], [5080], [6]), ([1036], [5086], [14])]]
        for rows in batches:
            receiver, guard, backend, req = self.fixture()
            req["batches"] = rows
            with patch.object(guard, "check", wraps=guard.check) as checked:
                self.assertEqual(receiver._transfer_kv_cache_all_groups(req), "done")
            self.assertIsNone(guard.failure)
            self.assertTrue(any(call.args[1] == "pd_slot_transfer_mismatch" and bool(call.args[0])
                                for call in checked.call_args_list if len(call.args) > 1))
            self.assertEqual(backend.call_count, len(rows))

    def test_whole_range_comparison_detects_shift_loss_duplicate_wrong_pairs_and_shape(self):
        cases = [[], [([1030], [5080], [10])], [([1040], [5080], [20])],
                 [([1030, 1030], [5080, 5080], [10, 10])],
                 [([1040, 1030], [5080, 5090], [10, 10])], [([1030], [], [20])],
                 [([1030], [5080], [0])]]
        for action in ("log", "raise"):
            for rows in cases:
                with self.subTest(action=action, rows=rows):
                    receiver, guard, backend, req = self.fixture(action)
                    req["batches"] = rows
                    with self.assertRaisesRegex(KVGuardError, "pd_slot_transfer_mismatch") if action == "raise" else nullcontext():
                        receiver._transfer_kv_cache_all_groups(req)
                    self.assertEqual(guard.failure, "pd_slot_transfer_mismatch")
                    self.assertEqual(backend.call_count, len(rows))

    def test_request_snapshot_replacement_or_retirement_detects_stale_transfer(self):
        for action in ("log", "raise"):
            for replace in (False, True):
                receiver, guard, backend, req = self.fixture(action)
                req["batches"] = [([1030], [5080], [20])]
                def completion(*args):
                    old = guard.plans.pop("d")
                    if replace:
                        guard.plans["d"] = tuple(list(old))
                    return 0
                backend.side_effect = completion
                with self.assertRaisesRegex(KVGuardError, "pd_slot_stale_transfer") if action == "raise" else nullcontext():
                    receiver._transfer_kv_cache_all_groups(req)
                self.assertEqual(guard.failure, "pd_slot_stale_transfer")

    def test_native_error_identity_and_context_cleanup(self):
        receiver, guard, backend, req = self.fixture()
        req.update(error=RuntimeError("backend failure"), batches=[])
        with self.assertRaises(RuntimeError) as caught:
            receiver._transfer_kv_cache_all_groups(req)
        self.assertIs(caught.exception, req.pop("error"))
        req["batches"] = [([1030], [5080], [20])]
        self.assertEqual(receiver._transfer_kv_cache_all_groups(req), "done")
        self.assertIsNone(guard.failure)

    def test_peer_discovery_and_strided_planes(self):
        receiver, guard, backend, req = self.fixture()
        receiver.kv_caches_base_addr.pop("remote")
        receiver.block_stride_per_addr = [[16]]
        receiver.remote_block_stride_per_addr["remote"][2] = [[32]]
        req.update(discover=True, batches=[([1048, 1064], [5256, 5288], [10, 10])])
        receiver._transfer_kv_cache_all_groups(req)
        self.assertIsNone(guard.failure)

    def test_native_layer_name_resolution(self):
        receiver, guard, backend, req = self.fixture()
        receiver.kv_group2layeridx = {0: ({"layer_names": ["k", "v"]}, [0, 1])}
        receiver.kv_caches_base_addr["local"][1] = [[1000], [2000]]
        receiver.kv_caches_base_addr["remote"][2] = [[7000], [5000]]
        receiver.remote_block_stride_per_addr["remote"][2] = [[10], [10]]
        receiver.block_len_per_addr = receiver.block_stride_per_addr = [[10], [10]]
        self.names.return_value = {"k": 1, "v": 0}
        self.resolve.side_effect = lambda layer, *args: 1 - layer
        req["batches"] = [([1030, 2030], [5080, 7080], [20, 20])]
        receiver._transfer_kv_cache_all_groups(req)
        self.assertIsNone(guard.failure)
        self.assertEqual(self.resolve.call_count, 2)


class TestSwapBatchSlots(TestCase):
    def test_native_idempotent_load_retires_only_completed_plans(self):
        for completed in (False, True):
            guard = Guard("raise", slot=True)
            guard.plans[False, 1] = ((0,), (1,))
            class Worker:
                _submitted_load_event_indices = {1}
                _load_hwm = 1 if completed else -1
                def _submit_transfer(self, *args, **kwargs):
                    pass
            module = "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.worker"
            with patch.dict(sys.modules, {module: NS(PreemptOffloadWorker=Worker)}):
                transfer.install(guard, False)
            Worker()._submit_transfer([0], [1], 1, False, sync=True)
            self.assertEqual(bool(guard.plans), not completed)

    def test_plan_capacity_and_native_completion_follow_action(self):
        for action in ("log", "raise"):
            for store in (False, True):
                for fault in (None, "plan", "bounds", "completion"):
                    with self.subTest(action=action, store=store, fault=fault):
                        guard = Guard(action, slot=True)
                        sources, targets = [0, 1], [1, 2]
                        guard.plans[store, 1] = (tuple(sources), tuple(targets))
                        if fault == "plan":
                            targets = [2, 3]
                        if fault == "bounds":
                            targets = [1, 4]
                            guard.plans[store, 1] = (tuple(sources), tuple(targets))
                        class Worker:
                            num_gpu_blocks = num_cpu_blocks = 4
                            def __init__(self):
                                self._submitted_load_event_indices = set()
                                self._completed_store_events, self._load_hwm = {}, -1
                            def _submit_transfer(self, src_block_ids, dst_block_ids, event_idx, is_store, sync=False):
                                if fault != "completion":
                                    self._completed_store_events[event_idx] = 1
                                    self._load_hwm = event_idx
                        module = "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.worker"
                        with patch.dict(sys.modules, {module: NS(PreemptOffloadWorker=Worker)}):
                            transfer.install(guard, False)
                        reason = "swap_slot_completion_missing" if fault == "completion" else "swap_slot_plan_mismatch" if fault else None
                        with self.assertRaisesRegex(KVGuardError, reason) if reason and action == "raise" else nullcontext():
                            Worker()._submit_transfer(sources, targets, 1, store, sync=True)
                        self.assertEqual(guard.failure, reason)
