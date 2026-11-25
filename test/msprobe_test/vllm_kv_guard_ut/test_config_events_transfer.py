import json
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from msprobe.vllm_kv_guard import config, events, transfer


def parallel(**fields):
    return SimpleNamespace(**({"tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                              "prefill_context_parallel_size": 1, "decode_context_parallel_size": 1} | fields))


class TestConfig(TestCase):
    def test_slot_configuration_and_disabled_compatibility(self):
        self.assertEqual(config.parse_options({"slot": True}),
                         {"enabled": True, "action": "log", "slot": True})
        self.assertEqual(config.parse_options({"slot": False}), {"enabled": True, "action": "log"})
        for options in ({"slot": 1}, {"slot": "true"}, {"slot": None}, {"slot_audit": True},
                        {"slot": True, "enabled": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                config.parse_options(options)

    def test_slot_rejects_unadapted_paths_for_both_actions(self):
        for action in ("log", "raise"):
            for field, change, message in (
                ("parallel_config", {"enable_dbo": True}, "DBO"),
                ("model_config", {"is_encoder_decoder": True}, "encoder-decoder"),
                ("speculative_config", object(), "speculative decoding"),
                ("kv_transfer_config", object(), "KV transfer"),
                ("additional_config", {"sparse_kv_offload_config": {"enabled": True}}, "sparse_kv_offload_config"),
                ("additional_config", {"sparse_kv_offload_config": True}, "sparse_kv_offload_config"),
                ("additional_config", {"sparse_kv_offload_config": []}, "sparse_kv_offload_config"),
            ):
                with self.subTest(action=action, field=field, change=change):
                    options = {"slot": True, "action": action}
                    supplied = SimpleNamespace(additional_config={"kv_trace": options},
                                               parallel_config=parallel(), model_config=SimpleNamespace(),
                                               speculative_config=None, kv_transfer_config=None)
                    if field == "additional_config":
                        supplied.additional_config.update(change)
                    else:
                        setattr(supplied, field, SimpleNamespace(**change) if isinstance(change, dict) else change)
                    with self.assertRaisesRegex(ValueError, message):
                        config.get_options(supplied)
                    options["slot"] = False
                    self.assertEqual(config.get_options(supplied), {"enabled": True, "action": action})

    def test_slot_supported_config_is_retained(self):
        supplied = SimpleNamespace(additional_config={"kv_trace": {"slot": True}},
                                   parallel_config=parallel(), model_config=SimpleNamespace(),
                                   speculative_config=None, kv_transfer_config=None)
        self.assertTrue(config.get_options(supplied)["slot"])
        for offload in (None, {"enabled": False}):
            with self.subTest(offload=offload):
                supplied.additional_config["sparse_kv_offload_config"] = offload
                self.assertTrue(config.get_options(supplied)["slot"])

    def test_slot_allows_only_the_supported_mooncake_transfer_configuration(self):
        supplied = SimpleNamespace(
            additional_config={"kv_trace": {"slot": True}}, parallel_config=parallel(),
            model_config=SimpleNamespace(), speculative_config=None,
            kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnectorV1", kv_role="kv_consumer"),
        )
        self.assertTrue(config.get_options(supplied)["pd_slot"])
        supplied.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "KV transfer"):
            config.get_options(supplied)

    def test_slot_retains_pcp_and_dcp_configurations(self):
        for action in ("log", "raise"):
            for pcp, dcp in ((2, 1), (1, 2), (2, 2)):
                with self.subTest(action=action, pcp=pcp, dcp=dcp):
                    supplied = SimpleNamespace(
                        additional_config={"kv_trace": {"slot": True, "action": action}},
                        parallel_config=parallel(prefill_context_parallel_size=pcp,
                                                        decode_context_parallel_size=dcp),
                        model_config=SimpleNamespace(), speculative_config=None, kv_transfer_config=None)
                    self.assertEqual(config.get_options(supplied), {"enabled": True, "action": action, "slot": True})

    def test_config_options(self):
        cases = [(None, None), ({}, {"enabled": True, "action": "log"}), ({"enabled": False}, None),
                 ({"action": "raise"}, {"enabled": True, "action": "raise"}),
                 ({"enabled": True, "action": "log"}, {"enabled": True, "action": "log"}),
                 ({"enabled": True, "action": "log", "kv_stats_enable": True},
                  {"enabled": True, "action": "log"})]
        for options, expected in cases:
            with self.subTest(options=options):
                original = deepcopy(options)
                self.assertEqual(config.parse_options(options), expected)
                self.assertEqual(config.get_options(SimpleNamespace(additional_config={"kv_trace": options})), expected)
                self.assertEqual(options, original)

    def test_invalid_config(self):
        cases = [(False, TypeError), ([], TypeError), ("log", TypeError), ({"enabled": 1}, ValueError),
                 ({"action": "stop"}, ValueError), ({"action": None}, ValueError)]
        for options, error in cases:
            with self.subTest(options=options):
                with self.assertRaises(error):
                    config.parse_options(options)

    def test_config_disabled_by_default(self):
        for additional in [None, {}, {"other": 1}]:
            with self.subTest(additional=additional):
                self.assertIsNone(config.get_options(SimpleNamespace(additional_config=additional)))


class TestEvents(TestCase):
    def setUp(self):
        patcher = patch.object(events, "logger")
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

    def test_emit_json(self):
        for event, level in [("kv.write", "info"), ("kv.violation", "error"), ("kv.error", "error")]:
            with self.subTest(event=event):
                self.mock_logger.reset_mock()
                fields = {"req_id": "请求", "slots": [17, -1]}
                events.emit(event, **fields)
                prefix = f"[KV_GUARD] {event} "
                message = getattr(self.mock_logger, level).call_args.args[0]
                self.assertTrue(message.startswith(prefix))
                self.assertIn("请求", message)
                self.assertEqual(json.loads(message[len(prefix):]), fields)
                self.assertEqual(len(self.mock_logger.mock_calls), 1)

    def test_unserializable_event_is_silent(self):
        for circular in [False, True]:
            with self.subTest(circular=circular):
                value = [] if circular else object()
                if circular:
                    value.append(value)
                events.emit("kv.write", value=value)
                self.mock_logger.assert_not_called()
                self.assertEqual(self.mock_logger.mock_calls, [])

    def test_result_fields(self):
        for sampled, expected in [(None, {}), ([], {}), ([[1, 2], []], {"a": 2, "b": 0, "bad": None})]:
            with self.subTest(sampled=sampled):
                connector = SimpleNamespace(finished_sending={"b", "a"}, finished_recving=None,
                                            failed_recving=set(), invalid_block_ids={9, 2})
                output = SimpleNamespace(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1, "bad": -1},
                                         sampled_token_ids=sampled, kv_connector_output=connector)
                original = deepcopy(output)
                fields = events.result_fields(output)
                self.assertEqual(fields, {"req_ids": ["a", "b"], "req_id_to_index": output.req_id_to_index,
                                          "num_output_tokens": expected, "kv_connector_output": {
                                              "finished_sending": ["a", "b"], "finished_recving": None,
                                              "failed_recving": [], "invalid_block_ids": [2, 9]}})
                self.assertEqual(output, original)
                fields["req_ids"].clear()
                fields["req_id_to_index"].clear()
                self.assertEqual(output, original)
                output.kv_connector_output = None
                self.assertNotIn("kv_connector_output", events.result_fields(output))


class TestTransfer(TestCase):
    def setUp(self):
        patcher = patch.object(transfer, "emit")
        self.mock_emit = patcher.start()
        self.addCleanup(patcher.stop)

    def test_connector_public_projection(self):
        for as_mapping in [True, False]:
            with self.subTest(as_mapping=as_mapping):
                self.mock_emit.reset_mock()
                public = dict(local_block_ids=[1, 2], remote_block_ids=[4, 5], num_computed_tokens=0,
                              num_external_tokens=16, remote_engine_id="peer", remote_request_id="remote")
                request = {**public, "private": "not logged"}
                requests = {"a": request if as_mapping else SimpleNamespace(**request), "unknown": {}}
                original = deepcopy(requests)
                poll = Mock(side_effect=AssertionError("completion queues must not be consumed"))
                metadata = SimpleNamespace(requests=requests, requests_to_send={"a": object()}, get_finished=poll)
                transfer.record_plan(metadata, "trace", "scheduler.schedule", rank=3)
                self.mock_emit.assert_called_once()
                self.assertEqual(self.mock_emit.call_args.args, ("boundary",))
                fields = self.mock_emit.call_args.kwargs
                self.assertEqual(fields["coverage"], "selected_public_fields")
                self.assertEqual(fields["trace_id"], "trace")
                self.assertEqual(fields["boundary"], "scheduler.schedule")
                self.assertEqual(fields["rank"], 3)
                self.assertEqual(fields["evidence"], "connector_plan")
                self.assertEqual(fields["metadata_type"], "types.SimpleNamespace")
                self.assertEqual(fields["requests"], {"a": public})
                self.assertEqual(fields["unexpanded_req_ids"], ["unknown"])
                self.assertEqual(fields["send_req_ids"], ["a"])
                self.assertEqual(requests, original)
                poll.assert_not_called()

    def test_connector_absent_and_unknown(self):
        transfer.record_plan(None, "trace", "worker")
        self.mock_emit.assert_not_called()
        transfer.record_plan(object(), "trace", "worker")
        self.mock_emit.assert_called_once_with("boundary", trace_id="trace", boundary="worker",
                                               evidence="connector_plan", metadata_type="builtins.object",
                                               coverage="not_expanded")
