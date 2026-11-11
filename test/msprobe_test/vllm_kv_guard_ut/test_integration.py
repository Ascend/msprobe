import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, call, patch


def _load_wiring_modules():
    source = Path(__file__).resolve().parents[3] / "python/msprobe/vllm_kv_guard"
    name = "_kv_guard_wiring_ut"
    helpers = {}
    for module, fields in {
        "config": {"get_options": Mock(return_value=None), "parse_options": Mock()},
        "checker": {"Guard": Mock(return_value=Mock()), "bind": Mock()},
        "events": {"emit": Mock(), "result_fields": Mock(return_value={"result_marker": "native"})},
        "transfer": {"record_plan": Mock()},
        "model_runner": {"install": Mock()},
    }.items():
        helper = ModuleType(f"{name}.{module}")
        helper.__dict__.update(fields)
        sys.modules[helper.__name__] = helper
        helpers[module] = helper
    for module in ("", "scheduler", "worker"):
        fullname = f"{name}.{module}" if module else name
        spec = importlib.util.spec_from_file_location(fullname, source / (f"{module}.py" if module else "__init__.py"))
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = loaded
        spec.loader.exec_module(loaded)
        helpers[module or "plugin"] = loaded
    return SimpleNamespace(**helpers, guard=helpers["checker"].Guard.return_value)


def _output(ids=("a",), index=None):
    return SimpleNamespace(req_ids=list(ids), req_id_to_index=index or {req: i for i, req in enumerate(ids)})


def _scheduled():
    copy_type = namedtuple("Copy", "source destination")
    return SimpleNamespace(
        num_scheduled_tokens={"a": 1}, total_num_scheduled_tokens=1,
        finished_req_ids=set(), preempted_req_ids=None,
        scheduled_new_reqs=[SimpleNamespace(req_id="a", block_ids=([2],), num_computed_tokens=0, replay_start=None)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
        new_block_ids_to_zero=[2], kv_cache_block_copies=[copy_type(1, 2)], kv_connector_metadata=object(),
        scheduled_spec_decode_tokens={"a": [7]}, num_spec_tokens_to_schedule=1,
        skip_drafts=False, num_invalid_spec_tokens=None,
    )


class TestKVGuardWiring(TestCase):
    def setUp(self):
        modules_patcher = patch.dict(sys.modules)
        modules_patcher.start()
        self.addCleanup(modules_patcher.stop)
        self.wiring = _load_wiring_modules()

    def _expose(self, name, **fields):
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            path = ".".join(parts[:end])
            if path not in sys.modules:
                module = ModuleType(path)
                module.__path__ = []
                sys.modules[path] = module
        module = sys.modules[name]
        for key, value in fields.items():
            attribute_patcher = patch.object(module, key, value, create=True)
            attribute_patcher.start()
            self.addCleanup(attribute_patcher.stop)
        return module

    def _observed_worker(self, action="log", execute=None, sample=None):
        recorder = SimpleNamespace(trace_id=None, plan=(), pending=Mock())
        self.wiring.model_runner.install.return_value = recorder
        native = SimpleNamespace(rank=2, use_v2_model_runner=True, model_runner=object(),
                                 execute_model=Mock(return_value=execute), sample_tokens=Mock(return_value=sample))
        self.wiring.worker._observe(native, {"action": action})
        self.wiring.model_runner.install.assert_called_once_with(native.model_runner, 2, self.wiring.guard)
        plan = _scheduled()
        plan._msprobe_kv_trace_id, plan._msprobe_kv_plan = "trace", object()
        return native, recorder, plan

    def test_scheduler_preserves_schedule_and_result_boundaries(self):
        for speculative in (True, False):
            with self.subTest(speculative=speculative):
                wiring = _load_wiring_modules()
                plan, result = _scheduled(), _output()
                if not speculative:
                    plan.preempted_req_ids, plan.kv_cache_block_copies = {"old"}, None
                native = SimpleNamespace(vllm_config=SimpleNamespace(
                    speculative_config=object() if speculative else None),
                    schedule=Mock(return_value=plan), update_from_output=Mock(return_value="processed"))
                wiring.scheduler._observe(native, {"action": "raise"})
                self.assertIs(native.schedule("schedule", mode=True), plan)
                native.schedule.__wrapped__.assert_called_once_with("schedule", mode=True)
                wiring.checker.bind.assert_called_once_with(plan, {}, wiring.guard)
                binding = next(item.kwargs for item in wiring.events.emit.call_args_list if item.args == ("kv.bind",))
                self.assertEqual("speculative" in binding, speculative)
                if speculative:
                    self.assertEqual(binding["speculative"]["draft_counts"], {"a": 1})
                    self.assertEqual(binding["kv_cache_block_copies"], [{"source": 1, "destination": 2}])
                wiring.transfer.record_plan.assert_called_once_with(
                    plan.kv_connector_metadata, plan._msprobe_kv_trace_id, "scheduler.schedule")
                self.assertEqual(native.update_from_output(plan, result, "update", mode=False), "processed")
                native.update_from_output.__wrapped__.assert_called_once_with(plan, result, "update", mode=False)
                wiring.events.result_fields.assert_called_once_with(result)
                self.assertEqual(wiring.events.emit.call_args.kwargs["result_marker"], "native")
                self.assertEqual(wiring.guard.check.call_count, 2)

    def test_scheduler_delegates_invalid_result_routes(self):
        for ids, index, reason in (
            (("a", "a"), {"a": 0}, "result_request_index_mismatch"),
            (("a",), {"a": 2}, "result_request_index_mismatch"),
            (("b",), {"b": 0}, "result_request_not_scheduled"),
        ):
            with self.subTest(ids=ids, index=index):
                wiring = _load_wiring_modules()
                plan, update = _scheduled(), Mock()
                native = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=None),
                                         schedule=Mock(return_value=plan), update_from_output=update)
                wiring.scheduler._observe(native, {"action": "raise"})
                native.schedule()
                error = RuntimeError("guard rejected route")
                wiring.guard.check.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    native.update_from_output(plan, _output(ids, index))
                self.assertIs(caught.exception, error)
                wiring.guard.fail.assert_called_once_with(
                    reason, trace_id=plan._msprobe_kv_trace_id, boundary="scheduler.update_from_output")
                update.assert_not_called()

    def test_scheduler_preserves_native_exceptions(self):
        for boundary in ("schedule", "update_from_output"):
            with self.subTest(boundary=boundary):
                wiring = _load_wiring_modules()
                plan, error = _scheduled(), RuntimeError("native")
                native = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=None),
                                         schedule=Mock(return_value=plan), update_from_output=Mock())
                getattr(native, boundary).side_effect = error
                wiring.scheduler._observe(native, {"action": "log"})
                with self.assertRaises(RuntimeError) as caught:
                    if boundary == "schedule":
                        native.schedule()
                    else:
                        native.update_from_output(native.schedule(), _output())
                self.assertIs(caught.exception, error)
                self.assertEqual(wiring.events.emit.call_args.args, ("kv.error",))

    def test_scheduler_install_respects_config(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                observe, config = Mock(), object()

                class Native:
                    def __init__(self, supplied):
                        self.vllm_config = supplied

                self._expose("vllm.v1.core.sched.scheduler", Scheduler=Native)
                options = {"action": "log"}
                with patch.object(self.wiring.scheduler, "_observe", observe), \
                        patch.object(self.wiring.scheduler, "get_options", return_value=options if enabled else None) as get:
                    self.wiring.scheduler.install()
                    instance = Native(config)
                self.assertIs(instance.vllm_config, config)
                get.assert_called_once_with(config)
                self.assertEqual(observe.call_args_list, [call(instance, options)] if enabled else [])

    def test_worker_observes_direct_output(self):
        wiring, result = self.wiring, _output()
        native, recorder, plan = self._observed_worker(execute=result)
        timeline = Mock()
        for name, target in (("execute", native.execute_model.__wrapped__), ("pending", recorder.pending),
                             ("fields", wiring.events.result_fields), ("emit", wiring.events.emit),
                             ("check", wiring.guard.check), ("plan", wiring.transfer.record_plan)):
            timeline.attach_mock(target, name)
        self.assertIs(native.execute_model(plan, mode=True), result)
        native.execute_model.__wrapped__.assert_called_once_with(plan, mode=True)
        recorder.pending.assert_called_once_with()
        self.assertIsNone(recorder.trace_id)
        self.assertIs(recorder.plan, plan._msprobe_kv_plan)
        wiring.events.result_fields.assert_called_once_with(result)
        self.assertEqual(wiring.events.emit.call_args.kwargs["phase"], "observed")
        wiring.guard.check.assert_called_once_with()
        self.assertEqual([item[0] for item in timeline.mock_calls],
                         ["emit", "plan", "execute", "pending", "fields", "emit", "check"])

    def test_worker_async_output_is_not_unpacked_early(self):
        result = _output()
        lazy = SimpleNamespace(get_output=Mock(return_value=result))
        original = lazy.get_output
        native, recorder, plan = self._observed_worker(execute=lazy)
        self.assertIs(native.execute_model(plan), lazy)
        original.assert_not_called()
        recorder.pending.assert_not_called()
        self.wiring.guard.check.assert_not_called()
        self.assertIs(lazy.get_output("result", wait=True), result)
        original.assert_called_once_with("result", wait=True)
        recorder.pending.assert_called_once_with()
        self.wiring.events.result_fields.assert_called_once_with(result)
        self.assertEqual(self.wiring.events.emit.call_args.kwargs["trace_id"], "trace")

    def test_worker_sampling_finishes_deferred_plan(self):
        result = _output()
        native, recorder, plan = self._observed_worker(sample=result)
        self.assertIsNone(native.execute_model(plan))
        self.assertEqual(recorder.trace_id, "trace")
        recorder.pending.assert_not_called()
        self.assertIs(native.sample_tokens("sample", mode=True), result)
        native.sample_tokens.__wrapped__.assert_called_once_with("sample", mode=True)
        recorder.pending.assert_called_once_with()
        self.assertIsNone(recorder.trace_id)
        self.wiring.guard.check.assert_called_once_with()

    def test_worker_propagates_native_failures(self):
        for boundary in ("execute_model", "sample_tokens", "get_output"):
            with self.subTest(boundary=boundary):
                self.wiring = _load_wiring_modules()
                error = RuntimeError("native")
                lazy = SimpleNamespace(get_output=Mock(side_effect=error))
                native, recorder, plan = self._observed_worker(execute=lazy if boundary == "get_output" else None)
                if boundary == "get_output":
                    native.execute_model(plan)
                    target = lazy.get_output
                else:
                    target = getattr(native, boundary)
                    target.__wrapped__.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    target(plan) if boundary == "execute_model" else target()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.wiring.events.emit.call_args.args, ("kv.error",))
                if boundary != "get_output":
                    self.assertIsNone(recorder.trace_id)

    def test_worker_raise_happens_at_result_boundary(self):
        lazy, error = SimpleNamespace(get_output=Mock(return_value=_output())), RuntimeError("bad plan")
        native, recorder, plan = self._observed_worker(action="raise", execute=lazy)
        recorder.pending.side_effect = error
        self.assertIs(native.execute_model(plan), lazy)
        with self.assertRaises(RuntimeError) as caught:
            lazy.get_output()
        self.assertIs(caught.exception, error)
        self.wiring.events.result_fields.assert_not_called()
        self.wiring.guard.check.assert_not_called()

    def test_worker_requires_v2(self):
        with self.assertRaisesRegex(ValueError, "Model Runner V2"):
            self.wiring.worker._observe(SimpleNamespace(use_v2_model_runner=False), {"action": "log"})
        self.wiring.model_runner.install.assert_not_called()

    def test_worker_install_respects_config(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                observe, config = Mock(), object()

                class Wrapper:
                    worker = object()
                    vllm_config = config

                    def init_device(self, *args, **kwargs):
                        return args, kwargs

                self._expose("vllm.v1.worker.worker_base", WorkerWrapperBase=Wrapper)
                options = {"action": "log"}
                with patch.object(self.wiring.worker, "_observe", observe), \
                        patch.object(self.wiring.worker, "get_options", return_value=options if enabled else None) as get:
                    self.wiring.worker.install()
                    instance = Wrapper()
                    self.assertEqual(instance.init_device("device", ordinal=2), (("device",), {"ordinal": 2}))
                get.assert_called_once_with(config)
                self.assertEqual(observe.call_args_list, [call(instance.worker, options)] if enabled else [])

    def test_plugin_ignores_other_platforms(self):
        self._expose("vllm.platforms", current_platform=SimpleNamespace(device_type="cuda"))
        self.wiring.plugin.register()
        self.wiring.config.parse_options.assert_not_called()

    def test_plugin_filters_only_tool_config(self):
        load, global_patch, schedule, run_worker = Mock(), Mock(), Mock(), Mock()

        class AscendConfig:
            def __init__(self, *args, **kwargs):
                self.arguments = args, kwargs

        class Engine:
            run_engine_core = staticmethod(Mock(return_value="engine"))

        self._expose("vllm.platforms", current_platform=SimpleNamespace(device_type="npu"))
        self._expose("vllm.plugins", load_general_plugins=load)
        self._expose("vllm_ascend", _ensure_global_patch=global_patch)
        self._expose("vllm_ascend.ascend_config", AscendConfig=AscendConfig)
        self._expose("vllm.v1.engine.core", EngineCoreProc=Engine)
        with patch.object(self.wiring.scheduler, "install", schedule), \
                patch.object(self.wiring.worker, "install", run_worker):
            self.wiring.plugin.register()
        options = {"enabled": True, "action": "raise"}
        self.assertEqual(AscendConfig("arg", kv_trace=options, native=4).arguments, (("arg",), {"native": 4}))
        self.wiring.config.parse_options.assert_called_once_with(options)
        self.assertEqual(options, {"enabled": True, "action": "raise"})
        self.assertEqual(AscendConfig(native=5).arguments, ((), {"native": 5}))
        error = ValueError("invalid options")
        self.wiring.config.parse_options.side_effect = error
        with self.assertRaises(ValueError) as caught:
            AscendConfig(kv_trace={"action": "invalid"})
        self.assertIs(caught.exception, error)
        native_run = Engine.run_engine_core.args[0]
        self.assertEqual(Engine.run_engine_core("engine", rank=2), "engine")
        native_run.assert_called_once_with("engine", rank=2)
        for target in (load, global_patch, schedule, run_worker):
            target.assert_called_once_with()
