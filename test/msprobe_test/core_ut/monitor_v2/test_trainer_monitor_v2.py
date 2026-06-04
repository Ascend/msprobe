import unittest
from unittest.mock import MagicMock, patch

import torch

from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.trainer import TrainerMonitorV2, _resolve_framework


class FakeMonitor(BaseMonitorV2):
    def __init__(self, collect_outputs=None):
        super().__init__()
        self.collect_outputs = list(collect_outputs or [])
        self.config_seen = None
        self.start_calls = []
        self.stop_calls = 0

    def set_config(self, config):
        super().set_config(config)
        self.config_seen = config

    def start(self, *args, **kwargs):
        self.start_calls.append((args, kwargs))

    def stop(self):
        self.stop_calls += 1

    def collect(self):
        if not self.collect_outputs:
            return None
        out = self.collect_outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeWriter:
    def __init__(self, out_dir, rank=0, async_write=False):
        self.out_dir = out_dir
        self.rank = rank
        self.async_write = async_write
        self.closed = 0
        self.writes = []

    def write_monitor_data(self, data):
        self.writes.append(data)
        return "written"

    def close(self):
        self.closed += 1


class OptimizerWithStep:
    def __init__(self):
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return "step-ok"


class TestTrainerMonitorV2(unittest.TestCase):
    def test_resolve_framework_when_alias_or_invalid_value_is_provided_then_pass(self):
        self.assertEqual(_resolve_framework({"framework": "torch"}, None), "pytorch")
        self.assertEqual(_resolve_framework({"framework": "ms"}, None), "mindspore")
        self.assertEqual(_resolve_framework({}, "pt"), "pytorch")
        with self.assertRaises(ValueError):
            _resolve_framework({}, "mxnet")

    @patch("msprobe.core.monitor_v2.trainer.CSVWriterV2", new=FakeWriter)
    @patch("msprobe.core.monitor_v2.trainer.MonitorFactory.create")
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.get_rank_id", return_value=7)
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.set_fmk")
    @patch(
        "msprobe.core.monitor_v2.trainer.load_json",
        return_value={
            "framework": "torch",
            "output_dir": "./out",
            "rank": 7,
            "step_interval": 2,
            "step_count_per_record": 2,
            "collect_times": 3,
            "monitors": {
                "module": {"enabled": True, "ops": ["min"]},
                "optimizer": {"enabled": False},
                "broken": "skip-me",
            },
        },
    )
    def test_start_when_config_and_optimizer_are_valid_then_pass(self, mock_load, mock_set_fmk, mock_rank, mock_create):
        mon = FakeMonitor(
            collect_outputs=[
                {"rows": [{"module_name": "layer", "stats": {"min": torch.tensor(1.0)}}]},
                {"rows": [{"module_name": "layer", "stats": {"min": torch.tensor(2.0)}}]},
            ]
        )
        mock_create.return_value = mon

        trainer = TrainerMonitorV2("config.json")
        self.assertEqual(trainer.framework, "pytorch")
        self.assertEqual(trainer._target_ranks, {7})
        self.assertEqual(len(trainer.monitors), 1)
        self.assertEqual(mon.config_seen, {"ops": ["min"]})
        self.assertEqual(getattr(mon, "slug"), "module")
        mock_set_fmk.assert_called_once_with("pytorch")

        optimizer = OptimizerWithStep()
        trainer.start(model="demo-model", optimizer=optimizer, extra="ctx")
        self.assertEqual(len(mon.start_calls), 1)
        _, kwargs = mon.start_calls[0]
        self.assertEqual(kwargs["rank"], 7)
        self.assertEqual(kwargs["output_dir"], "./out")
        self.assertEqual(kwargs["extra"], "ctx")
        self.assertEqual(kwargs["step_provider"](), 0)

        trainer.step()
        self.assertEqual(trainer.current_step, 0)
        self.assertEqual(len(trainer.writer.writes), 0)

        result = optimizer.step()
        self.assertEqual(result, "step-ok")
        self.assertEqual(optimizer.calls, 1)
        self.assertEqual(trainer.current_step, 1)
        self.assertEqual(len(trainer.writer.writes), 1)
        written = trainer.writer.writes[0]
        self.assertEqual(written["slug"], "module")
        self.assertEqual(written["monitor"], "module")
        self.assertEqual(written["rank"], 7)
        self.assertEqual(written["step"], 0)
        self.assertEqual(written["step_interval"], 2)
        self.assertEqual(written["step_count_per_record"], 2)
        self.assertEqual(written["start_step"], 0)

        trainer.stop()
        self.assertEqual(mon.stop_calls, 1)
        self.assertEqual(trainer.writer.closed, 1)
        self.assertFalse(trainer._step_patched)

    @patch("msprobe.core.monitor_v2.trainer.CSVWriterV2", new=FakeWriter)
    @patch("msprobe.core.monitor_v2.trainer.MonitorFactory.create")
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.get_rank_id", return_value=1)
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.set_fmk")
    @patch(
        "msprobe.core.monitor_v2.trainer.load_json",
        return_value={
            "framework": "pytorch",
            "rank": ["bad"],
            "start_step": 1,
            "stop_step": 3,
            "step_interval": 1,
            "step_count_per_record": "bad",
            "monitors": {"module": {"enabled": True}},
        },
    )
    def test_step_when_collect_fails_and_writer_closes_repeatedly_then_pass(
        self, mock_load, mock_set_fmk, mock_rank, mock_create
    ):
        mon = FakeMonitor(collect_outputs=[RuntimeError("collect failed"), None])
        mock_create.return_value = mon

        trainer = TrainerMonitorV2("config.json")
        self.assertEqual(trainer.step_count_per_record, 1)
        self.assertEqual(trainer._target_ranks, set())
        self.assertFalse(trainer._should_collect_step(0))
        self.assertTrue(trainer._should_collect_step(1))
        self.assertTrue(trainer._should_collect_step(2))
        self.assertFalse(trainer._should_collect_step(3))

        trainer.step()
        self.assertEqual(trainer.current_step, 1)
        self.assertEqual(len(trainer.writer.writes), 0)

        trainer._close_writer()
        trainer._close_writer()
        self.assertEqual(trainer.writer.closed, 1)

    @patch("msprobe.core.monitor_v2.trainer.CSVWriterV2", new=FakeWriter)
    @patch("msprobe.core.monitor_v2.trainer.MonitorFactory.create", return_value=None)
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.get_rank_id", return_value=3)
    @patch("msprobe.core.monitor_v2.trainer.FmkAdp.set_fmk")
    @patch(
        "msprobe.core.monitor_v2.trainer.load_json",
        return_value={"framework": "pytorch", "rank": [2], "monitors": {"module": {"enabled": True}}},
    )
    def test_start_when_current_rank_is_not_target_rank_then_pass(self, mock_load, mock_set_fmk, mock_rank, mock_create):
        trainer = TrainerMonitorV2("config.json")
        trainer.start(model="model")
        trainer.step()
        trainer.stop()
        self.assertEqual(trainer.current_step, 0)
        self.assertEqual(len(trainer.writer.writes), 0)

    @patch("msprobe.core.monitor_v2.trainer.load_json", return_value=[])
    def test_init_when_config_is_not_dict_then_fail(self, mock_load):
        with self.assertRaises(TypeError):
            TrainerMonitorV2("config.json")

    @patch("msprobe.core.monitor_v2.trainer.load_json", return_value={"framework": "pytorch", "format": "json"})
    def test_init_when_format_is_not_csv_then_fail(self, mock_load):
        with self.assertRaises(ValueError):
            TrainerMonitorV2("config.json")
