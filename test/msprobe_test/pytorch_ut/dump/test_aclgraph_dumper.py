# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

try:
    from msprobe.lib import aclgraph_dump_ext as _aclgraph_dump_ext
    import msprobe.pytorch.aclgraph_dumper as aclgraph_dumper_module
    from msprobe.pytorch.aclgraph_dumper import AclGraphDumper
    IMPORT_OK = True
except Exception:
    _aclgraph_dump_ext = None
    aclgraph_dumper_module = None
    AclGraphDumper = None
    IMPORT_OK = False


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


class AddOnlyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


@unittest.skipUnless(IMPORT_OK, "aclgraph_dump_ext or aclgraph_dumper import failed, skip tests")
class TestAclGraphDumper(unittest.TestCase):
    def test_aclgraph_dump_ext_import_ok(self):
        self.assertTrue(hasattr(_aclgraph_dump_ext, "get_acl_stat_dict"))

    def test_init_validate_dump_path(self):
        mock_config = {"task": "statistics", "dump_path": "./dump", "statistics": {"list": []}}
        with patch.object(aclgraph_dumper_module, "create_directory") as mock_create_dir, \
                patch.object(aclgraph_dumper_module, "check_and_get_real_path") as mock_real_path, \
                patch.object(aclgraph_dumper_module, "load_json", return_value=mock_config):
            mock_real_path.side_effect = ["/tmp/config.json", "/tmp/aclgraph_dump"]
            dumper = AclGraphDumper(config_path="./config.json")

        self.assertEqual(dumper.dump_path, "/tmp/aclgraph_dump")
        self.assertEqual(mock_real_path.call_count, 2)
        mock_create_dir.assert_called_once_with("/tmp/aclgraph_dump")

    def test_start_and_step_return_none(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "mix", [])), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_patch"), \
                patch.object(AclGraphDumper, "_synchronize"), \
                patch.object(AclGraphDumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict", return_value={}), \
                patch.object(aclgraph_dumper_module, "save_json"):
            dumper = AclGraphDumper(config_path="./config.json")
            start_ret = dumper.start(MagicMock())
            step_ret = dumper.step()

        self.assertIsNone(start_ret)
        self.assertIsNone(step_ret)

    def test_step_writes_dump_json(self):
        stats = {
            "toy.forward.input.0": {
                "dtype": "Float",
                "shape": [2, 8],
                "max": 1.0,
                "min": -1.0,
                "mean": 0.0,
                "norm": 2.0,
            },
            "toy.forward.output.0": {
                "dtype": "Float",
                "shape": [2, 4],
                "max": 2.0,
                "min": -2.0,
                "mean": 0.1,
                "norm": 3.0,
            },
        }

        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "mix", [])), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_synchronize"), \
                patch.object(AclGraphDumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict", return_value=stats), \
                patch.object(aclgraph_dumper_module, "save_json") as mock_save_json:
            dumper = AclGraphDumper(config_path="./config.json")
            dumper._running = True
            dumper.step()

        save_path, dump_json = mock_save_json.call_args[0][0], mock_save_json.call_args[0][1]
        self.assertEqual(save_path, os.path.join("./dump/step0/rank0", "dump.json"))
        self.assertEqual(dump_json["task"], aclgraph_dumper_module.Const.STATISTICS)
        self.assertEqual(dump_json["level"], aclgraph_dumper_module.Const.LEVEL_MIX)
        self.assertEqual(dump_json["framework"], aclgraph_dumper_module.Const.PT_FRAMEWORK)
        self.assertIn("toy.forward", dump_json["data"])
        self.assertEqual(mock_save_json.call_args.kwargs["indent"], 2)

    def test_step_clear_only_without_dump(self):
        stats = {
            "toy.forward.input.0": {
                "dtype": "Float",
                "shape": [2, 8],
                "max": 1.0,
                "min": -1.0,
                "mean": 0.0,
                "norm": 2.0,
            }
        }

        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "mix", [])), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_synchronize"), \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict", return_value=stats) as mock_get_stats, \
                patch.object(aclgraph_dumper_module, "save_json") as mock_save_json:
            dumper = AclGraphDumper(config_path="./config.json")
            dumper._running = True
            step_before = dumper.step_id
            dumper.step(dump=False)

        mock_get_stats.assert_called_once_with(clear=True)
        mock_save_json.assert_not_called()
        self.assertEqual(dumper.step_id, step_before)

    def test_collect_acl_stat_called_after_start(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "mix", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        self.assertGreater(mock_acl_stat.call_count, 0)

    def test_load_msprobe_config(self):
        mock_config = {
            "task": "statistics",
            "dump_path": "./cfg_dump",
            "level": "L1",
            "statistics": {
                "list": ["linear", "mlp"]
            }
        }
        with patch.object(aclgraph_dumper_module, "check_and_get_real_path", return_value="/tmp/config.json"), \
                patch.object(aclgraph_dumper_module, "load_json", return_value=mock_config):
            dump_path, module_list, level, rank = AclGraphDumper._load_msprobe_config("./config.json")

        self.assertEqual(dump_path, "./cfg_dump")
        self.assertEqual(module_list, ["linear", "mlp"])
        self.assertEqual(level, "L1")
        self.assertIsNone(rank)

    def test_collect_with_list_filter(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", ["linear"], "mix", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        self.assertGreater(mock_acl_stat.call_count, 0)
        for call in mock_acl_stat.call_args_list:
            self.assertIn("linear", call.args[1])

    def test_collect_with_list_filter_ignore_case(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", ["Linear"], "mix", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        self.assertGreater(mock_acl_stat.call_count, 0)
        for call in mock_acl_stat.call_args_list:
            self.assertIn("linear", call.args[1])

    def test_collect_l1_api_stat_through_dispatch(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "L1", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        api_tags = [call.args[1] for call in mock_acl_stat.call_args_list if ".Aten." in call.args[1]]
        self.assertGreater(len(api_tags), 0)
        self.assertTrue(any(tag.startswith("Module.linear.Linear.Aten.") for tag in api_tags))

    def test_collect_mix_api_stat_with_api_only_filter(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", ["add"], "mix", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = AddOnlyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        api_tags = [call.args[1] for call in mock_acl_stat.call_args_list if ".Aten.add." in call.args[1]]
        self.assertGreater(len(api_tags), 0)
        self.assertTrue(all("add" in tag for tag in api_tags))

    def test_collect_with_dump_style_module_name_filter(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", ["Module.linear"], "mix", [])), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        tags = [call.args[1] for call in mock_acl_stat.call_args_list]
        self.assertGreater(len(tags), 0)
        self.assertTrue(any(tag.startswith("Module.linear.Linear.") for tag in tags))
        self.assertTrue(any(tag.startswith("Module.linear.Linear.Aten.") for tag in tags))

    def test_dispatch_collect_guard_only_applies_to_acl_stat(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "L1", [])):
            dumper = AclGraphDumper(config_path="./config.json")
            dumper._push_scope("linear")
            mode = aclgraph_dumper_module._AclTorchDispatchMode(dumper)
            collect_states = []
            func_states = []

            def fake_collect(*args, **kwargs):
                collect_states.append(dumper._is_dispatch_collecting())
                return True

            class FakeFunc:
                overloadname = "default"

                def __str__(self):
                    return "aten.add.Tensor"

                def __call__(self, *args, **kwargs):
                    func_states.append(dumper._is_dispatch_collecting())
                    return args[0]

            with patch.object(dumper, "_collect", side_effect=fake_collect):
                out = mode.__torch_dispatch__(FakeFunc(), (), args=(torch.randn(2, 8),), kwargs={})

        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(collect_states)
        self.assertTrue(all(collect_states))
        self.assertTrue(func_states)
        self.assertTrue(all(not state for state in func_states))

    def test_start_skips_non_target_rank(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_load_msprobe_config", return_value=("./dump", [], "mix", [1])), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_patch") as mock_patch, \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict") as mock_get_stats, \
                patch.object(aclgraph_dumper_module, "save_json") as mock_save_json:
            dumper = AclGraphDumper(config_path="./config.json")
            dumper.start(MagicMock())
            dumper.step()

        self.assertFalse(dumper._running)
        mock_patch.assert_not_called()
        mock_get_stats.assert_not_called()
        mock_save_json.assert_not_called()
