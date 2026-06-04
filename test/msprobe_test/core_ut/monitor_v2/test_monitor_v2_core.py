import csv
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.factory import MonitorFactory
from msprobe.core.monitor_v2.utils import (
    build_param2name,
    get_vpp_stage_from_tag,
    iter_model_chunks,
    iter_model_modules,
    iter_model_params,
)
from msprobe.core.monitor_v2.writer import CSVWriterV2


class DummyMonitor(BaseMonitorV2):
    def __init__(self):
        super().__init__()
        self.started = False
        self.stopped = False

    def start(self, *args, **kwargs):
        self.started = True

    def stop(self):
        self.stopped = True


class TestBaseMonitorV2(unittest.TestCase):
    def test_collect_when_rows_exist_then_pass(self):
        mon = DummyMonitor()
        mon.set_config({})
        self.assertEqual(mon._ops, mon.DEFAULT_OPS)

        mon.set_context(rank=3, output_dir="./tmp")
        self.assertEqual(mon._context["rank"], 3)
        self.assertEqual(mon._context["output_dir"], "./tmp")

        self.assertIsNone(mon.collect())
        mon._rows.append({"module_name": "layer", "stats": {"min": torch.tensor(1.0)}})
        out = mon.collect()
        self.assertEqual(len(out["rows"]), 1)
        self.assertEqual(mon._rows, [])
        self.assertIsNone(mon.collect())

    @patch("msprobe.core.monitor_v2.base.logger.warning")
    def test_set_config_when_ops_contain_invalid_values_then_pass(self, mock_warning):
        mon = DummyMonitor()
        mon.set_config({"ops": "mean"})
        self.assertEqual(mon._ops, ["mean"])

        mon.set_config({"ops": ["min", "bad"]})
        self.assertEqual(mon._ops, ["min"])
        self.assertTrue(mock_warning.called)

        mon.set_config({"ops": ["bad"]})
        self.assertEqual(mon._ops, mon.DEFAULT_OPS)


class TestMonitorV2Utils(unittest.TestCase):
    def test_iter_model_chunks_when_input_is_mixed_then_pass(self):
        self.assertEqual(list(iter_model_chunks(None)), [])

        single = object()
        self.assertEqual(list(iter_model_chunks(single)), [(single, "")])

        items = ["m0", None, "m2"]
        self.assertEqual(list(iter_model_chunks(items)), [("m0", "0:"), ("m2", "2:")])

    def test_build_param2name_when_model_or_optimizer_is_provided_then_pass(self):
        def fake_named_parameters(chunk):
            if chunk == "bad":
                raise RuntimeError("skip this chunk")
            return [("weight", f"{chunk}_param")]

        def fake_iter_named_modules(chunk):
            if chunk == "bad":
                raise RuntimeError("skip this chunk")
            return [("linear", f"{chunk}_module")]

        with patch("msprobe.core.monitor_v2.utils.FmkAdp.named_parameters", side_effect=fake_named_parameters):
            params = list(iter_model_params(["good", "bad"]))
        self.assertEqual(params, [("0:weight", "good_param")])

        with patch("msprobe.core.monitor_v2.utils.FmkAdp.iter_named_modules", side_effect=fake_iter_named_modules):
            modules = list(iter_model_modules(["good", "bad"]))
        self.assertEqual(modules, [("0:linear", "good_module")])

        with patch("msprobe.core.monitor_v2.utils.FmkAdp.named_parameters", return_value=[("p0", "param0")]):
            param2name = build_param2name(model="model", optimizer=None)
        self.assertEqual(param2name, {"param0": "p0"})

        optimizer = type("Opt", (), {"param_groups": [{"params": ["a", "b"]}]})()
        with patch("msprobe.core.monitor_v2.utils.FmkAdp.named_parameters", return_value=[]):
            param2name = build_param2name(model="model", optimizer=optimizer)
        self.assertEqual(param2name, {"a": "param_0", "b": "param_1"})

    def test_get_vpp_stage_from_tag_when_tag_has_or_lacks_prefix_then_pass(self):
        self.assertEqual(get_vpp_stage_from_tag("2:layer.weight"), 2)
        self.assertIsNone(get_vpp_stage_from_tag("layer.weight"))
        self.assertIsNone(get_vpp_stage_from_tag("x:layer.weight"))
        self.assertIsNone(get_vpp_stage_from_tag(""))


class TestMonitorFactory(unittest.TestCase):
    def test_create_when_registry_contains_valid_and_invalid_inputs_then_pass(self):
        registry = {"demo": {"dummy": DummyMonitor}}
        with patch.object(MonitorFactory, "_REGISTRY", registry):
            inst = MonitorFactory.create("DEMO", "DuMmY")
            self.assertIsInstance(inst, DummyMonitor)
            self.assertEqual(MonitorFactory.available("demo"), {"dummy": DummyMonitor})

            available = MonitorFactory.available("demo")
            available["extra"] = object()
            self.assertEqual(MonitorFactory.available("demo"), {"dummy": DummyMonitor})

            self.assertIsNone(MonitorFactory.create("demo", "missing"))
            self.assertIsNone(MonitorFactory.create("missing", "dummy"))


class TestCSVWriterV2(unittest.TestCase):
    def setUp(self):
        self.prev_fmk = FmkAdp.fmk
        FmkAdp.set_fmk(Const.PT_FRAMEWORK)

    def tearDown(self):
        FmkAdp.set_fmk(self.prev_fmk)

    def test_safe_slug_when_slug_and_step_formats_vary_then_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CSVWriterV2(tmpdir, rank=5)
            self.assertEqual(writer._safe_slug("../a b/c"), "a_b_c")
            self.assertEqual(
                writer._resolve_csv_path("mon", step="bad", interval=2, start_step=0),
                os.path.join(writer.rank_dir, "mon.csv"),
            )
            self.assertEqual(
                writer._resolve_csv_path("mon", step=5, interval=2, start_step=0),
                os.path.join(writer.rank_dir, "mon_step4-5.csv"),
            )
            self.assertEqual(
                writer._resolve_csv_path("mon", step=-1, interval=2, start_step=0),
                os.path.join(writer.rank_dir, "mon_step-1-0.csv"),
            )
            self.assertIsNone(writer.close())

    def test_write_monitor_data_when_rows_contain_missing_stats_then_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CSVWriterV2(tmpdir, rank=3)
            self.assertIsNone(writer.write_monitor_data({"rows": []}))

            csv_path = writer.write_monitor_data(
                {
                    "monitor": "module",
                    "slug": "module/stats",
                    "step": 6,
                    "step_count_per_record": 2,
                    "start_step": 0,
                    "rows": [
                        {
                            "vpp_stage": 0,
                            "module_name": "0:layer.weight",
                            "stats": {"min": torch.tensor(1.0), "max": torch.tensor(2.0)},
                        },
                        {
                            "vpp_stage": MonitorConst.DEFAULT_STAGE,
                            "module_name": "layer.bias",
                            "stats": {"min": torch.tensor(3.0)},
                        },
                    ],
                }
            )
            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["step"], "6")
            self.assertEqual(rows[0]["module_name"], "0:layer.weight")
            self.assertAlmostEqual(float(rows[0]["min"]), 1.0)
            self.assertTrue(math.isnan(float(rows[0]["max"])))
            self.assertTrue(math.isnan(float(rows[1]["max"])))

